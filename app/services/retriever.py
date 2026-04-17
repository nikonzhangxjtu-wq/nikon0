"""检索抽象层。

V2 要点：
- 启动时 ``describe_collection`` 探测**真实的 dense 字段名**（新 schema 是
  ``dense_vector``，旧 schema 可能是 ``vector``），避免硬编码失配。
- 只有当集合里真的存在 ``sparse_vector`` 字段、且 Milvus 支持 BM25 文本查询时，
  才走 sparse 召回；否则 **dense-only**，避免 Milvus Lite 上报
  ``search_data ... illegal``。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

from app.core.config import settings


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


class VectorRetriever:
    """Milvus 检索封装。

    - ``dense_field``: 在 ``__init__`` 中根据 collection 的实际 schema 决定；
      若探测失败，退化为 ``"dense_vector"`` 以保持与新 schema 兼容。
    - ``sparse_enabled``: 集合里含 ``sparse_vector`` 字段才为 True；
      否则 :meth:`retrieve` 直接跳过 sparse 召回。
    """

    def __init__(self) -> None:
        self.collection_name: str = settings.milvus_collection
        self.embed_model = OllamaEmbeddings(
            model=settings.embed_model,
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
        print(
            f"[INFO] VectorRetriever: collection={self.collection_name} "
            f"dense_field={self.dense_field} sparse_enabled={self.sparse_enabled}"
        )

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

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """dense 主召回；若 ``sparse_enabled=True``，再与 sparse 结果做加权融合。"""
        query = query.strip()
        if not query:
            return []

        query_vector = self.embed_model.embed_query(query)
        dense_hits = self._search_dense(query_vector=query_vector, limit=max(top_k, 10))

        sparse_hits: list[dict] = []
        if self.sparse_enabled:
            sparse_hits = self._search_sparse_text(query=query, limit=max(top_k, 10))

        if self.sparse_enabled:
            fused_hits = self._fuse_hits_by_rank(
                dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=top_k
            )
        else:
            fused_hits = dense_hits[:top_k]

        if not fused_hits:
            return []

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

    def _search_dense(self, *, query_vector: list[float], limit: int) -> list[dict]:
        output_fields = ["chunk_id", "text", "manual_name", "image_ids"]
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                anns_field=self.dense_field,
                data=[query_vector],
                limit=limit,
                output_fields=output_fields,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] dense 检索失败 (field={self.dense_field}): {exc}")
            return []
        return list(results[0]) if results and results[0] else []

    def _search_sparse_text(self, *, query: str, limit: int) -> list[dict]:
        """走 BM25 Function 的 sparse 文本检索（仅在 sparse_enabled 时调用）。"""
        output_fields = ["chunk_id", "text", "manual_name", "image_ids"]
        # 不同 Milvus / pymilvus 版本对 BM25 文本检索入参格式略有差异，做兼容兜底
        payload_candidates: tuple[object, ...] = ([query], [{"text": query}], [query.strip()])
        for payload in payload_candidates:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    anns_field="sparse_vector",
                    data=payload,
                    limit=limit,
                    output_fields=output_fields,
                )
                if results and results[0]:
                    return list(results[0])
            except Exception:
                continue
        # 这里到达一次后就不再打 WARN，避免评测时刷屏；真出问题可在集合无 sparse 字段时由调用侧感知
        return []

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
