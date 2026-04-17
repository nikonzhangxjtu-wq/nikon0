"""联合测试：模拟 Milvus + Ollama，串联 build_index 入库与 retriever 检索。

无需本机启动 Milvus/Ollama；验证写入字段与检索解析是否一致。

运行（项目根目录）::

    python -m app.test.test_retriever_build_index_integration
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualChunk
from app.services.retriever import VectorRetriever
from scripts import build_index


class FakeMilvusClient:
    """最小 Milvus 行为：与 build_index / retriever 调用约定一致。"""

    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        return collection_name in self._collections

    def drop_collection(self, collection_name: str, **kwargs) -> None:
        self._collections.pop(collection_name, None)

    def create_collection(self, collection_name: str, schema=None, **kwargs) -> None:
        self._collections[collection_name] = {"rows": []}

    def insert(self, collection_name: str, data, **kwargs) -> dict:
        rows = self._collections[collection_name]["rows"]
        if isinstance(data, dict):
            data = [data]
        rows.extend(data)
        return {"insert_count": len(data), "ids": []}

    def create_index(self, *args, **kwargs) -> None:
        return None

    def load_collection(self, *args, **kwargs) -> None:
        return None

    def search(
        self,
        collection_name: str,
        data,
        limit: int = 10,
        output_fields=None,
        **kwargs,
    ) -> list[list[dict]]:
        stored = self._collections.get(collection_name, {}).get("rows", [])
        anns_field = kwargs.get("anns_field", "dense_vector")
        hits: list[dict] = []
        for i, row in enumerate(stored[:limit]):
            entity = {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "manual_name": row["manual_name"],
                "image_ids": row["image_ids"],
            }
            # 仅模拟排序分值来源，便于覆盖 dense/sparse 两路逻辑
            if anns_field == "sparse_vector":
                hits.append({"entity": entity, "distance": 0.02 * (i + 1)})
            else:
                hits.append({"entity": entity, "distance": 0.01 * (i + 1)})
        return [hits]


def _mock_embeddings(dim: int = 8) -> MagicMock:
    m = MagicMock()
    m.embed_query.return_value = [0.1] * dim
    m.embed_documents.side_effect = lambda texts: [[0.1 + i * 0.01] * dim for i in range(len(texts))]
    return m


def _sample_chunks() -> list[ManualChunk]:
    return [
        ManualChunk(
            chunk_id="联合测试_0000",
            manual_name="联合测试手册",
            text="空调制冷剂说明与注意事项",
            image_ids=["pic_a", "pic_b"],
        ),
        ManualChunk(
            chunk_id="联合测试_0001",
            manual_name="联合测试手册",
            text="冰箱门封条清洁方法",
            image_ids=[],
        ),
    ]


def test_build_index_then_retrieve_roundtrip():
    fake = FakeMilvusClient()
    dim = 8
    mock_emb = _mock_embeddings(dim)

    svc = MagicMock()
    svc.parse_and_chunk.return_value = _sample_chunks()

    with (
        patch("scripts.build_index.OllamaEmbeddings", return_value=mock_emb),
        patch("scripts.build_index.ManualIngestionService", return_value=svc),
        patch("scripts.build_index._milvus_client", return_value=fake),
        patch("scripts.build_index._probe_dim", return_value=dim),
    ):
        build_index.main()

    stored = fake._collections[settings.milvus_collection]["rows"]
    assert len(stored) == 2
    assert stored[0]["chunk_id"] == "联合测试_0000"
    assert "pic_a" in stored[0]["image_ids"]
    ids0 = json.loads(stored[0]["image_ids"])
    assert ids0 == ["pic_a", "pic_b"]

    with (
        patch("app.services.retriever.MilvusClient", return_value=fake),
        patch("app.services.retriever.OllamaEmbeddings", return_value=mock_emb),
    ):
        retriever = VectorRetriever()
        chunks_out = retriever.retrieve("制冷剂", top_k=4)

    assert len(chunks_out) == 2
    assert chunks_out[0].chunk_id == "联合测试_0000"
    assert "制冷" in chunks_out[0].text or chunks_out[0].text
    assert chunks_out[0].image_ids == ["pic_a", "pic_b"]
    assert chunks_out[0].manual_name == "联合测试手册"
    assert isinstance(chunks_out[0].score, float)

    trace = retriever.build_trace(
        query="制冷剂",
        top_k=4,
        raw_chunks=chunks_out,
        filtered_chunks=chunks_out[:1],
    )
    assert trace.query == "制冷剂"
    assert trace.raw_count == 2
    assert trace.filtered_count == 1
    assert trace.retrieved_chunk_ids == ["联合测试_0000", "联合测试_0001"]
    assert trace.filtered_chunk_ids == ["联合测试_0000"]
    assert trace.top1_score == chunks_out[0].score


if __name__ == "__main__":
    test_build_index_then_retrieve_roundtrip()
    print("[OK] test_retriever_build_index_integration passed")
