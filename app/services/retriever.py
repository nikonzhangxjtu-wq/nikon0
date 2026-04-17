"""检索抽象层。

本文件提供接口与 V1 占位实现，便于你分步替换：
1）先跑通占位检索，验证端到端；
2）再接入 Milvus + llama-index 等真实检索。
"""

from __future__ import annotations
import json
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
from dataclasses import dataclass, field
from pymilvus import MilvusClient

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
    """过滤检索结果。"""
    filtered_chunks: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.score > settings.retriever_context_filter_score_threshold:
            filtered_chunks.append(chunk)
    return filtered_chunks

class VectorRetriever:
    
    collection_name: str = settings.milvus_collection
    embed_model: OllamaEmbeddings
    client: MilvusClient

    def __init__(self):
        self.embed_model = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
        kwargs: dict = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        self.client = MilvusClient(**kwargs)
        if not self.client.has_collection(collection_name=self.collection_name):
            raise ValueError(f"Collection {settings.milvus_collection} not found")

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
                # 非 JSON 字符串，兜底作为单个 ID
                return [value]
            if isinstance(decoded, list):
                return [str(v) for v in decoded]
            return [str(decoded)]
        return [str(value)]

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """向量检索并返回结构化 chunk 列表。"""
        query = query.strip()
        if not query:
            return []

        query_vector = self.embed_model.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "text", "manual_name", "image_ids"],
        )
        if not results or not results[0]:
            return []

        list_results: list[RetrievedChunk] = []
        # MilvusClient.search 对单 query 返回 List[List[dict]]，取第 1 组命中
        for hit in results[0]:
            entity = hit.get("entity", {})
            chunk_id = str(entity.get("chunk_id", ""))
            text = str(entity.get("text", ""))
            manual_name = str(entity.get("manual_name", ""))
            image_ids = self._parse_image_ids(entity.get("image_ids"))
            score = float(hit.get("distance", hit.get("score", 0.0)))
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
            retrieved_chunk_ids=[chunk.chunk_id for chunk in raw_chunks],
            filtered_chunk_ids=[chunk.chunk_id for chunk in filtered_chunks],
            retrieved_manual_names=[chunk.manual_name for chunk in raw_chunks if chunk.manual_name],
            filtered_manual_names=[chunk.manual_name for chunk in filtered_chunks if chunk.manual_name],
        )
        