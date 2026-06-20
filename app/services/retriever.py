"""检索抽象层。

V2 要点：
- 启动时 ``describe_collection`` 探测**真实的 dense 字段名**（新 schema 是
  ``dense_vector``，旧 schema 可能是 ``vector``），避免硬编码失配。
- 只有当集合里真的存在 ``sparse_vector`` 字段、且 Milvus 支持 BM25 文本查询时，
  才走 sparse 召回；否则 **dense-only**，避免 Milvus Lite 上报
  ``search_data ... illegal``。
- ``manual_name`` 上的 TRIE 索引只在带 **标量 filter** 的 ``search`` 中才会被利用；
  可通过 :meth:`retrieve` 的 ``manual_name=`` 限定单本手册后再做向量/BM25 召回。
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

from app.core.config import settings
from app.services.multimodal.embeddings import JinaMultimodalEmbeddingClient
from app.services.multimodal.facts_store import ManualImageFactStore
from app.services.rag_skill.rerank import rerank_fused_hits
from app.utils.manual_lang import query_prefers_chinese_embedding


@dataclass
class ManualImageEvidence:
    """手册图片检索命中的结构化证据。"""

    image_id: str
    image_type: str = ""
    match_reason: str = ""
    prompt_text: str = ""
    score: float = 0.0
    parent_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """单条检索上下文单元。"""

    chunk_id: str
    text: str
    score: float
    manual_name: str = ""
    image_ids: list[str] = field(default_factory=list)
    image_evidence: list[ManualImageEvidence] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    """一次检索的调试元信息，供评测与观测使用。"""

    query: str
    top_k: int
    raw_count: int
    filtered_count: int = 0
    score_threshold: float = 0.0
    top1_score: float | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    filtered_chunk_ids: list[str] = field(default_factory=list)
    retrieved_manual_names: list[str] = field(default_factory=list)
    filtered_manual_names: list[str] = field(default_factory=list)
    source_queries: list[str] = field(default_factory=list)
    manual_name_decisions: list[str] = field(default_factory=list)
    image_vector_hits: list[str] = field(default_factory=list)
    ocr_entity_hits: list[str] = field(default_factory=list)
    selected_image_ids: list[str] = field(default_factory=list)


def retriever_context_filter(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """按分数阈值过滤检索结果。"""
    return [c for c in chunks if c.score > settings.retriever_context_filter_score_threshold]


# 常见 dense 字段命名，按优先级依次尝试匹配到 collection 实际字段
_DENSE_FIELD_CANDIDATES = ("dense_vector", "vector", "embedding")


def _milvus_manual_name_filter_expr(manual_name: str) -> str:
    """构造 ``manual_name == "..."`` 过滤表达式（转义 Milvus 字符串字面量）。"""
    v = manual_name.strip()
    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
    return f'manual_name == "{escaped}"'


class VectorRetriever:
    """Milvus 检索封装。

    - ``dense_field``: 在 ``__init__`` 中根据 collection 的实际 schema 决定；
      若探测失败，退化为 ``"dense_vector"`` 以保持与新 schema 兼容。
    - ``sparse_enabled``: 集合里含 ``sparse_vector`` 字段才为 True；
      否则 :meth:`retrieve` 直接跳过 sparse 召回。
    """

    def __init__(self) -> None:
        self.collection_name: str = settings.milvus_collection
        self._embed_zh = OllamaEmbeddings(
            model=settings.embed_model_zh,
            base_url=settings.ollama_base_url,
        )
        self._embed_en = OllamaEmbeddings(
            model=settings.embed_model_en,
            base_url=settings.ollama_base_url,
        )
        kwargs: dict = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        self.client = MilvusClient(**kwargs)
        if not self.client.has_collection(collection_name=self.collection_name):
            raise ValueError(f"Collection {self.collection_name} not found")

        self.dense_field, self.sparse_enabled, self.available_fields = self._probe_schema()
        self.image_collection_name = settings.multimodal_image_collection
        self.image_retrieval_enabled = self._probe_image_collection()
        self._jina_client: JinaMultimodalEmbeddingClient | None = None
        self._image_fact_store: ManualImageFactStore | None = None
        if not self.available_fields:
            print(
                "[WARN] describe_collection 未能获取字段列表，"
                f"默认使用 dense 字段名 '{self.dense_field}' 与 sparse_enabled={self.sparse_enabled}。"
            )

        # 启动时显式 load，提前暴露「没建索引 / load 不了」这类问题，避免每次 search
        # 才报 "collection not loaded"。失败不直接抛：留给后续 search 兜底并打印明确 WARN。
        self._ensure_loaded(force=True)

        print(
            f"[INFO] VectorRetriever: collection={self.collection_name} "
            f"dense_field={self.dense_field} sparse_enabled={self.sparse_enabled} "
            f"image_retrieval_enabled={self.image_retrieval_enabled}"
        )

    def _ensure_loaded(self, *, force: bool = False) -> bool:
        """尝试 load collection；成功返回 True，失败打印 WARN 并返回 False。"""
        try:
            self.client.load_collection(collection_name=self.collection_name)
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] load_collection 失败 (collection={self.collection_name}): {exc}. "
                "这通常意味着索引未建成功（例如 sparse_vector 字段存在但未建 "
                "SPARSE_INVERTED_INDEX，或 dense_vector 未建 HNSW）。"
                "请确认 `python scripts/build_index.py` 末尾 index_ok=True 后再重试。"
            )
            return False

    def _probe_schema(self) -> tuple[str, bool, list[str]]:
        """返回 (dense 字段名, 是否有 sparse_vector, 全部字段名)。"""
        try:
            desc = self.client.describe_collection(collection_name=self.collection_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] describe_collection 失败: {exc}")
            return "dense_vector", False, []

        fields_raw = desc.get("fields") if isinstance(desc, dict) else getattr(desc, "fields", None)
        names: list[str] = []
        if fields_raw:
            for f in fields_raw:
                if isinstance(f, dict):
                    names.append(str(f.get("name", "")))
                else:
                    names.append(str(getattr(f, "name", "")))

        dense_field = "dense_vector"
        for cand in _DENSE_FIELD_CANDIDATES:
            if cand in names:
                dense_field = cand
                break

        sparse_enabled = "sparse_vector" in names and settings.milvus_enable_bm25
        return dense_field, sparse_enabled, names

    def _probe_image_collection(self) -> bool:
        """探测图片 collection 是否存在；不存在时保持纯文本 RAG。"""
        if not settings.multimodal_image_retrieval_enabled:
            return False
        try:
            return self.client.has_collection(collection_name=settings.multimodal_image_collection)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 图片 collection 探测失败，跳过图片检索: {exc}")
            return False

    @staticmethod
    def _parse_image_ids(value: object) -> list[str]:
        """将 Milvus 中存储的 image_ids 统一转为 list[str]。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            if isinstance(decoded, list):
                return [str(v) for v in decoded]
            return [str(decoded)]
        return [str(value)]

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        *,
        manual_name: str | None = None,
        image_inputs: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """dense 主召回；若 ``sparse_enabled=True``，再与 sparse 结果做加权融合。

        若传入 ``manual_name``，两路 search 会带上 ``manual_name == "..."`` 过滤，
        从而用上 TRIE 索引并缩小候选集（典型「结构化约束 + 多尺度检索」的第一步）。
        """
        query = query.strip()
        if not query:
            return []

        filter_expr = (
            _milvus_manual_name_filter_expr(manual_name) if manual_name and manual_name.strip() else None
        )

        embedder = (
            self._embed_zh if query_prefers_chinese_embedding(query) else self._embed_en
        )
        model_name = (
            settings.embed_model_zh if query_prefers_chinese_embedding(query) else settings.embed_model_en
        )
        query_vector = self._embed_query(embedder, model_name, query)
        # 扩大召回池：检索 top_k * 3 候选，给 reranker 更多选择空间
        recall_limit = max(top_k * 3, 15)
        dense_hits = self._search_dense(
            query_vector=query_vector, limit=recall_limit, filter_expr=filter_expr
        )

        sparse_hits: list[dict] = []
        if self.sparse_enabled:
            sparse_hits = self._search_sparse_text(
                query=query, limit=recall_limit, filter_expr=filter_expr
            )

        if self.sparse_enabled:
            fused_hits = self._fuse_hits_by_rank(
                dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=max(15, top_k * 2)
            )
        else:
            fused_hits = dense_hits[:top_k]

        if not fused_hits:
            return []
        # 进行rerank
        fused_hits = self._rerank(fused_hits,query,top_k)

        list_results: list[RetrievedChunk] = []
        requested_manual_name = manual_name
        for hit in fused_hits:
            entity = hit.get("entity", {}) or {}
            chunk_id = str(entity.get("chunk_id", ""))
            text = str(entity.get("text", ""))
            chunk_manual_name = str(entity.get("manual_name", ""))
            image_ids = self._parse_image_ids(entity.get("image_ids"))
            score = float(hit.get("score", hit.get("distance", 0.0)))
            list_results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    manual_name=chunk_manual_name,
                    image_ids=image_ids,
                )
            )
        return self._maybe_add_image_evidence(
            chunks=list_results,
            query=query,
            top_k=top_k,
            manual_name=self._resolve_image_manual_scope(
                chunks=list_results,
                requested_manual_name=requested_manual_name,
            ),
            image_inputs=image_inputs or [],
        )

    def _get_jina_client(self) -> JinaMultimodalEmbeddingClient:
        if self._jina_client is None:
            self._jina_client = JinaMultimodalEmbeddingClient()
        return self._jina_client

    def _get_image_fact_store(self) -> ManualImageFactStore:
        if self._image_fact_store is None:
            self._image_fact_store = ManualImageFactStore(settings.manual_image_cache_path)
        return self._image_fact_store

    def _maybe_add_image_evidence(
        self,
        *,
        chunks: list[RetrievedChunk],
        query: str,
        top_k: int,
        manual_name: str | None,
        image_inputs: list[str],
    ) -> list[RetrievedChunk]:
        """多路图片召回并融合到文本 chunk。

        文本 RAG 是主路径；图片检索失败时只降级，不影响已有文本召回。
        """
        if not settings.multimodal_image_retrieval_enabled:
            return chunks
        mode = (settings.multimodal_image_retrieval_mode or "attached_only").strip().lower()
        if mode == "disabled":
            return chunks
        if mode == "attached_only":
            return self._attach_image_evidence_from_chunks(chunks)

        if not self.image_retrieval_enabled:
            return chunks
        if not manual_name:
            # 图片语义容易跨产品误召回；没有确定手册范围时，宁可只用文本 RAG。
            return chunks
        try:
            image_hits = self._retrieve_image_evidence(
                query=query,
                top_k=max(settings.multimodal_image_top_k, top_k),
                manual_name=manual_name,
                image_inputs=image_inputs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 多模态图片检索失败，已回退纯文本 RAG: {exc}")
            return chunks
        if not image_hits:
            return chunks
        return self._merge_image_evidence(chunks, image_hits)

    def _attach_image_evidence_from_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """只给文本已命中的 chunk 附加其自身图片事实，保证文本/图片证据一致。"""
        image_ids: list[str] = []
        for chunk in chunks:
            for image_id in [*chunk.image_ids, *_image_ids_from_text(chunk.text)]:
                image_id = image_id.strip()
                if image_id and image_id not in image_ids:
                    image_ids.append(image_id)
        if not image_ids:
            return chunks

        try:
            facts = self._get_image_fact_store().get_many(image_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 图片事实读取失败，已回退纯文本 RAG: {exc}")
            return chunks
        if not facts:
            return chunks

        for chunk in chunks:
            evidence = list(chunk.image_evidence)
            chunk_image_ids: list[str] = []
            for image_id in [*chunk.image_ids, *_image_ids_from_text(chunk.text)]:
                image_id = image_id.strip()
                if image_id and image_id not in chunk_image_ids:
                    chunk_image_ids.append(image_id)
            for image_id in chunk_image_ids:
                fact = facts.get(image_id)
                if fact is None:
                    continue
                evidence.append(
                    ManualImageEvidence(
                        image_id=fact.image_id,
                        image_type=fact.image_type,
                        match_reason="文本命中片段附带图片",
                        prompt_text=fact.to_prompt_text(),
                        score=chunk.score,
                        parent_chunk_ids=[chunk.chunk_id],
                    )
                )
            chunk.image_evidence = self._deduplicate_evidence(evidence)
        return chunks

    def _retrieve_image_evidence(
        self,
        *,
        query: str,
        top_k: int,
        manual_name: str | None,
        image_inputs: list[str],
    ) -> list[ManualImageEvidence]:
        hits: list[ManualImageEvidence] = []
        client = self._get_jina_client()
        text_vector = client.embed_text(query)
        if text_vector:
            hits.extend(
                self._search_semantic_image_vectors(
                    vectors=[text_vector],
                    top_k=top_k,
                    manual_name=manual_name,
                    reason_prefix="文本问题-图片语义召回",
                )
            )

        image_vectors: list[list[float]] = []
        for raw_image in image_inputs[: settings.vision_max_images]:
            vec = client.embed_image_input(raw_image)
            if vec:
                image_vectors.append(vec)
        if image_vectors:
            hits.extend(
                self._search_raw_image_vectors(
                    vectors=image_vectors,
                    top_k=top_k,
                    manual_name=manual_name,
                    reason_prefix="用户上传图片-手册图片相似召回",
                )
            )

        hits.extend(self._match_image_entities(query=query, top_k=settings.multimodal_entity_top_k, manual_name=manual_name))
        return self._deduplicate_image_hits(hits)[:top_k]

    def _search_semantic_image_vectors(
        self,
        *,
        vectors: list[list[float]],
        top_k: int,
        manual_name: str | None,
        reason_prefix: str,
    ) -> list[ManualImageEvidence]:
        """文本问题只检索 semantic_vector，表示“图片在手册上下文中表达的知识”。"""
        return self._search_image_vectors(
            vectors=vectors,
            anns_fields=("semantic_vector",),
            top_k=top_k,
            manual_name=manual_name,
            reason_prefix=reason_prefix,
        )

    def _search_raw_image_vectors(
        self,
        *,
        vectors: list[list[float]],
        top_k: int,
        manual_name: str | None,
        reason_prefix: str,
    ) -> list[ManualImageEvidence]:
        """用户上传图片时才使用 image_vector，避免文本问题被线条图像素相似度干扰。"""
        return self._search_image_vectors(
            vectors=vectors,
            anns_fields=("image_vector",),
            top_k=top_k,
            manual_name=manual_name,
            reason_prefix=reason_prefix,
        )

    def _search_image_vectors(
        self,
        *,
        vectors: list[list[float]],
        anns_fields: tuple[str, ...],
        top_k: int,
        manual_name: str | None,
        reason_prefix: str,
    ) -> list[ManualImageEvidence]:
        output_fields = [
            "image_id",
            "image_path",
            "manual_name",
            "parent_chunk_ids",
            "parent_context_text",
            "context_intent",
            "image_type",
            "semantic_text",
            "ocr_text",
            "visual_entities",
            "operation_steps",
            "warnings",
        ]
        filter_expr = _milvus_manual_name_filter_expr(manual_name) if manual_name else None
        hits: list[ManualImageEvidence] = []
        for anns_field in anns_fields:
            for vector in vectors:
                search_kw: dict = {
                    "collection_name": self.image_collection_name,
                    "anns_field": anns_field,
                    "data": [vector],
                    "limit": top_k,
                    "output_fields": output_fields,
                }
                if filter_expr:
                    search_kw["filter"] = filter_expr
                try:
                    results = self.client.search(**search_kw)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 图片向量检索失败 field={anns_field}: {exc}")
                    continue
                for hit in list(results[0]) if results and results[0] else []:
                    score = float(hit.get("score", hit.get("distance", 0.0)))
                    if score < settings.multimodal_image_min_score:
                        continue
                    evidence = self._image_hit_to_evidence(
                        hit.get("entity", {}) or {},
                        score=score,
                        match_reason=f"{reason_prefix}({anns_field})",
                    )
                    hits.append(evidence)
        return hits

    def _match_image_entities(
        self, *, query: str, top_k: int, manual_name: str | None
    ) -> list[ManualImageEvidence]:
        """OCR/实体精确匹配，保护故障码、按钮名、部件名这类不能靠相似度猜的字段。"""
        tokens = _entity_tokens(query)
        if not tokens:
            return []
        filter_expr = _milvus_manual_name_filter_expr(manual_name) if manual_name else ""
        output_fields = [
            "image_id",
            "manual_name",
            "parent_chunk_ids",
            "image_type",
            "parent_context_text",
            "context_intent",
            "semantic_text",
            "ocr_text",
            "visual_entities",
            "operation_steps",
            "warnings",
        ]
        try:
            rows = self.client.query(
                collection_name=self.image_collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=max(top_k * 25, 50),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 图片实体匹配失败: {exc}")
            return []
        hits: list[ManualImageEvidence] = []
        for row in rows or []:
            haystack = " ".join(
                str(row.get(field) or "")
                for field in (
                    "context_intent",
                    "ocr_text",
                    "visual_entities",
                    "operation_steps",
                    "warnings",
                    "semantic_text",
                )
            ).lower()
            matched = [token for token in tokens if token.lower() in haystack]
            if not matched:
                continue
            hits.append(
                self._image_hit_to_evidence(
                    row,
                    score=1.0 + min(len(matched), 4) * 0.05,
                    match_reason="OCR/实体精确匹配：" + ", ".join(matched[:4]),
                )
            )
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]

    @staticmethod
    def _image_hit_to_evidence(
        entity: dict,
        *,
        score: float,
        match_reason: str,
    ) -> ManualImageEvidence:
        image_id = str(entity.get("image_id", ""))
        parent_chunk_ids = VectorRetriever._parse_image_ids(entity.get("parent_chunk_ids"))
        prompt_parts: list[str] = []
        for label, field in (
            ("类型", "image_type"),
            ("图片意图", "context_intent"),
            ("OCR", "ocr_text"),
            ("视觉实体", "visual_entities"),
            ("操作步骤", "operation_steps"),
            ("警告/注意", "warnings"),
            ("父文本上下文", "parent_context_text"),
        ):
            value = str(entity.get(field) or "").strip()
            if value:
                prompt_parts.append(f"{label}: {value}")
        return ManualImageEvidence(
            image_id=image_id,
            image_type=str(entity.get("image_type") or ""),
            match_reason=match_reason,
            prompt_text="\n".join(prompt_parts),
            score=score,
            parent_chunk_ids=parent_chunk_ids,
        )

    @staticmethod
    def _resolve_image_manual_scope(
        *,
        chunks: list[RetrievedChunk],
        requested_manual_name: str | None,
    ) -> str | None:
        """确定图片检索范围。

        规则：显式/高置信 manual_name 优先；否则若文本 RAG top 结果高度集中在同一本
        手册，才允许图片检索。没有范围时跳过图片检索，减少跨产品错召回。
        """
        if requested_manual_name and requested_manual_name.strip():
            return requested_manual_name.strip()
        top = [c.manual_name for c in chunks[:5] if c.manual_name]
        if not top:
            return None
        counts: defaultdict[str, int] = defaultdict(int)
        for name in top:
            counts[name] += 1
        best, count = max(counts.items(), key=lambda kv: kv[1])
        if count >= 2 and count / len(top) >= 0.6:
            return best
        if len(top) == 1:
            return top[0]
        return None

    def _merge_image_evidence(
        self,
        chunks: list[RetrievedChunk],
        image_hits: list[ManualImageEvidence],
    ) -> list[RetrievedChunk]:
        by_chunk: defaultdict[str, list[ManualImageEvidence]] = defaultdict(list)
        for hit in image_hits:
            for chunk_id in hit.parent_chunk_ids:
                by_chunk[chunk_id].append(hit)

        existing_ids = {c.chunk_id for c in chunks}
        for chunk in chunks:
            evidence = list(chunk.image_evidence)
            evidence.extend(by_chunk.get(chunk.chunk_id, []))
            # 如果文本 chunk 自身带图，也把同图的图片 evidence 挂上。
            own_image_hits = [hit for hit in image_hits if hit.image_id in chunk.image_ids]
            evidence.extend(own_image_hits)
            chunk.image_evidence = self._deduplicate_evidence(evidence)

        missing_parent_ids = [cid for cid in by_chunk if cid not in existing_ids]
        parent_chunks = self._fetch_parent_chunks(missing_parent_ids)
        for parent in parent_chunks:
            parent.image_evidence = self._deduplicate_evidence(by_chunk.get(parent.chunk_id, []))
            chunks.append(parent)
        return chunks

    def _fetch_parent_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        safe_ids = [cid.replace("\\", "\\\\").replace('"', '\\"') for cid in chunk_ids[:20]]
        expr = "chunk_id in [" + ", ".join(f'"{cid}"' for cid in safe_ids) + "]"
        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["chunk_id", "text", "manual_name", "image_ids"],
                limit=len(safe_ids),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 图片父 chunk 回填失败: {exc}")
            return []
        result: list[RetrievedChunk] = []
        for row in rows or []:
            result.append(
                RetrievedChunk(
                    chunk_id=str(row.get("chunk_id") or ""),
                    text=str(row.get("text") or ""),
                    manual_name=str(row.get("manual_name") or ""),
                    image_ids=self._parse_image_ids(row.get("image_ids")),
                    score=0.01,
                )
            )
        return result

    @staticmethod
    def _deduplicate_evidence(items: list[ManualImageEvidence]) -> list[ManualImageEvidence]:
        by_id: dict[str, ManualImageEvidence] = {}
        for item in items:
            if not item.image_id:
                continue
            prev = by_id.get(item.image_id)
            if prev is None or item.score > prev.score:
                by_id[item.image_id] = item
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    @staticmethod
    def _deduplicate_image_hits(items: list[ManualImageEvidence]) -> list[ManualImageEvidence]:
        return VectorRetriever._deduplicate_evidence(items)

    def _search_dense(
        self, *, query_vector: list[float], limit: int, filter_expr: str | None = None
    ) -> list[dict]:
        output_fields = ["chunk_id", "text", "manual_name", "image_ids"]
        search_kw: dict = {
            "collection_name": self.collection_name,
            "anns_field": self.dense_field,
            "data": [query_vector],
            "limit": limit,
            "output_fields": output_fields,
        }
        if filter_expr:
            search_kw["filter"] = filter_expr
        try:
            results = self.client.search(**search_kw)
        except Exception as exc:  # noqa: BLE001
            # 典型恢复点：集合未 load。尝试一次 load + retry，再失败就放弃。
            msg = str(exc)
            if "not loaded" in msg and self._ensure_loaded():
                try:
                    results = self.client.search(**search_kw)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[WARN] dense 检索重试失败 (field={self.dense_field}): {exc2}")
                    return []
            else:
                print(f"[WARN] dense 检索失败 (field={self.dense_field}): {exc}")
                return []
        return list(results[0]) if results and results[0] else []

    @staticmethod
    def _embed_query(embedder: OllamaEmbeddings, model_name: str, query: str) -> list[float]:
        """生成 query embedding。

        某些 langchain-ollama / ollama Python 客户端组合会在 ``embed_query`` 上返回
        502，但同一 Ollama ``/api/embed`` HTTP 接口正常；这里保留封装优先，并提供
        直接 HTTP fallback，避免评测长跑被封装兼容性打断。
        """
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return embedder.embed_query(query)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))

        try:
            import requests as _req

            resp = _req.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/embed",
                json={"model": model_name, "input": [query]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings") or []
            if embeddings and isinstance(embeddings[0], list):
                return [float(v) for v in embeddings[0]]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Ollama embedding 失败 model={model_name}: {exc}"
            ) from exc

        raise RuntimeError(f"Ollama embedding 返回空结果 model={model_name}") from last_exc

    def _search_sparse_text(
        self, *, query: str, limit: int, filter_expr: str | None = None
    ) -> list[dict]:
        """走 BM25 Function 的 sparse 文本检索（仅在 sparse_enabled 时调用）。

        Milvus 2.5/2.6 + pymilvus 2.5+ 的正确入参格式是 ``data=[query_string]``，
        pymilvus 客户端会直接拒绝 ``[{"text": ...}]`` 这种 dict 形式
        (ParamError: search_data value is illegal)。
        """
        output_fields = ["chunk_id", "text", "manual_name", "image_ids"]
        search_kw: dict = {
            "collection_name": self.collection_name,
            "anns_field": "sparse_vector",
            "data": [query],
            "limit": limit,
            "output_fields": output_fields,
        }
        if filter_expr:
            search_kw["filter"] = filter_expr
        try:
            results = self.client.search(**search_kw)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "not loaded" in msg and self._ensure_loaded():
                try:
                    results = self.client.search(**search_kw)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[WARN] sparse BM25 检索重试失败: {exc2}")
                    return []
            else:
                print(f"[WARN] sparse BM25 检索失败: {exc}")
                return []
        return list(results[0]) if results and results[0] else []
    
    def _rerank(self, fused_hits: list[dict], query: str, top_k: int) -> list[dict]:
        """Cross-encoder 精排（见 ``rag_skill.rerank``）。"""
        return rerank_fused_hits(fused_hits, query, top_k)


    @staticmethod
    def _fuse_hits_by_rank(
        *, dense_hits: list[dict], sparse_hits: list[dict], top_k: int
    ) -> list[dict]:
        """按排名加权融合两路召回结果（dense 主，sparse 辅）。"""
        if not dense_hits and not sparse_hits:
            return []

        dense_weight = 0.6
        sparse_weight = 0.4

        fused_scores: defaultdict[str, float] = defaultdict(float)
        entities: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            entity = hit.get("entity", {}) or {}
            chunk_id = str(entity.get("chunk_id", ""))
            if not chunk_id:
                continue
            fused_scores[chunk_id] += dense_weight / rank
            entities[chunk_id] = entity

        for rank, hit in enumerate(sparse_hits, start=1):
            entity = hit.get("entity", {}) or {}
            chunk_id = str(entity.get("chunk_id", ""))
            if not chunk_id:
                continue
            fused_scores[chunk_id] += sparse_weight / rank
            entities.setdefault(chunk_id, entity)

        ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[
            :top_k
        ]
        return [{"entity": entities[cid], "score": fused_scores[cid]} for cid in ranked_ids]

    @staticmethod
    def build_trace(
        *,
        query: str,
        top_k: int,
        raw_chunks: list[RetrievedChunk],
        filtered_chunks: list[RetrievedChunk],
        source_queries: list[str] | None = None,
        manual_name_decisions: list[str] | None = None,
    ) -> RetrievalTrace:
        """根据召回前后结果组装统一 trace。"""
        return RetrievalTrace(
            query=query,
            top_k=top_k,
            raw_count=len(raw_chunks),
            filtered_count=len(filtered_chunks),
            score_threshold=settings.retriever_context_filter_score_threshold,
            top1_score=raw_chunks[0].score if raw_chunks else None,
            retrieved_chunk_ids=[c.chunk_id for c in raw_chunks],
            filtered_chunk_ids=[c.chunk_id for c in filtered_chunks],
            retrieved_manual_names=[c.manual_name for c in raw_chunks if c.manual_name],
            filtered_manual_names=[c.manual_name for c in filtered_chunks if c.manual_name],
            source_queries=source_queries or [query],
            manual_name_decisions=manual_name_decisions or [],
            image_vector_hits=[
                img.image_id
                for c in filtered_chunks
                for img in c.image_evidence
                if "向量召回" in img.match_reason
            ],
            ocr_entity_hits=[
                img.image_id
                for c in filtered_chunks
                for img in c.image_evidence
                if "OCR/实体" in img.match_reason
            ],
            selected_image_ids=[
                img.image_id
                for c in filtered_chunks
                for img in c.image_evidence
                if img.image_id
            ],
        )


def _entity_tokens(query: str) -> list[str]:
    """提取适合精确匹配的短实体，避免把整句都拿去 contains。"""
    text = query or ""
    tokens = re.findall(r"[A-Za-z]{2,}[-_/]?[A-Za-z0-9]*|\b[A-Z]?\d{2,4}\b|[\u4e00-\u9fff]{2,8}", text)
    stop = {"这个", "那个", "怎么", "如何", "为什么", "什么", "需要", "可以", "图片", "手册"}
    out: list[str] = []
    for token in tokens:
        if token in stop:
            continue
        if token not in out:
            out.append(token)
    return out[:12]


def _image_ids_from_text(text: str) -> list[str]:
    ids = re.findall(r"<IMG:([^>]+)>", text or "")
    out: list[str] = []
    for image_id in ids:
        image_id = image_id.strip()
        if image_id and image_id not in out:
            out.append(image_id)
    return out
