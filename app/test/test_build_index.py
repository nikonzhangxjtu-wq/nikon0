"""测试 `scripts/build_index.py` 流水线（Mock Ollama / Milvus，无需本机服务）。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 项目根目录（app/test/ → parents[2]）
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llama_index.core import Document

from app.core.config import settings
from app.services.ingestion import ManualChunk
from scripts import build_index


def test_chunks_to_documents():
    chunks = [
        ManualChunk(
            chunk_id="手册_0000",
            manual_name="手册",
            text="正文",
            image_ids=["a", "b"],
        )
    ]
    docs = build_index._chunks_to_documents(chunks)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "正文"
    assert docs[0].metadata["chunk_id"] == "手册_0000"
    assert docs[0].metadata["manual_name"] == "手册"
    assert docs[0].metadata["image_ids"] == ["a", "b"]


def test_truncate_warns_once(capsys):
    warned = [False]
    long = "x" * 100
    build_index._truncate(long, 50, "text", warned)
    build_index._truncate(long, 50, "text", warned)
    err = capsys.readouterr().out
    assert "[WARN]" in err
    assert err.count("[WARN]") == 1


def test_main_pipeline_mocked():
    dim = 4
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.0] * dim
    mock_emb.embed_documents.side_effect = lambda texts: [[float(i)] * dim for i, _ in enumerate(texts)]

    chunk = ManualChunk(
        chunk_id="t_0000",
        manual_name="t",
        text="hello",
        image_ids=["id1"],
    )
    mock_svc = MagicMock()
    mock_svc.parse_and_chunk.return_value = [chunk]

    mock_client = MagicMock()
    mock_client.has_collection.return_value = False

    with (
        patch("scripts.build_index.OllamaEmbeddings", return_value=mock_emb),
        patch("scripts.build_index.ManualIngestionService", return_value=mock_svc),
        patch("scripts.build_index._milvus_client", return_value=mock_client),
    ):
        build_index.main()

    mock_client.create_collection.assert_called_once()
    mock_client.insert.assert_called_once()
    _, insert_kw = mock_client.insert.call_args
    assert insert_kw["collection_name"] == settings.milvus_collection
    rows = insert_kw["data"]
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "t_0000"
    assert rows[0]["manual_name"] == "t"
    assert len(rows[0]["dense_vector"]) == dim
    assert "id1" in rows[0]["image_ids"]

    mock_client.create_index.assert_called_once()
    mock_client.load_collection.assert_called_once_with(collection_name=settings.milvus_collection)


if __name__ == "__main__":
    test_chunks_to_documents()
    warned = [False]
    out = build_index._truncate("x" * 100, 50, "text", warned)
    assert len(out) == 50
    test_main_pipeline_mocked()
    print("[OK] test_build_index: all tests passed")
