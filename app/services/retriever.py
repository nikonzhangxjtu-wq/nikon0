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
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

from app.core.config import settings
from app.services.rag_skill.rerank import rerank_fused_hits
from app.utils.manual_lang import query_prefers_chinese_embedding


@dataclass
class RetrievedChunk:
    """单条检索上下文单元。"""

    chunk_id: str
    text: str
    score: float
    manual_name: str = ""
    image_ids: list[str] = field(default_factory=list)


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
            f"dense_field={self.dense_field} sparse_enabled={self.sparse_enabled}"
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
        self, query: str, top_k: int = 4, *, manual_name: str | None = None
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
        query_vector = embedder.embed_query(query)
        dense_hits = self._search_dense(
            query_vector=query_vector, limit=max(top_k, 10), filter_expr=filter_expr
        )

        sparse_hits: list[dict] = []
        if self.sparse_enabled:
            sparse_hits = self._search_sparse_text(
                query=query, limit=max(top_k, 10), filter_expr=filter_expr
            )

        if self.sparse_enabled:
            fused_hits = self._fuse_hits_by_rank(
                dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=max(10,top_k)
            )
        else:
            fused_hits = dense_hits[:top_k]

        if not fused_hits:
            return []
        # 进行rerank
        fused_hits = self._rerank(fused_hits,query,top_k)

        list_results: list[RetrievedChunk] = []
        for hit in fused_hits:
            entity = hit.get("entity", {}) or {}
            chunk_id = str(entity.get("chunk_id", ""))
            text = str(entity.get("text", ""))
            manual_name = str(entity.get("manual_name", ""))
            image_ids = self._parse_image_ids(entity.get("image_ids"))
            score = float(hit.get("score", hit.get("distance", 0.0)))
            list_results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    manual_name=manual_name,
                    image_ids=image_ids,
                )
            )
        return list_results

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

        dense_weight = 0.7
        sparse_weight = 0.3

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
        )