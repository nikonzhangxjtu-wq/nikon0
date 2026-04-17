"""索引健康检查：切块 + 建库 +（可选）真实 Milvus 冒烟。

默认使用 Mock，避免依赖本机 Ollama/Milvus；需要真实环境时设置::

    RUN_LIVE_MILVUS=1

在项目根目录运行::

    python app/test/test_index_health.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.ingestion import ManualChunk, ManualIngestionService


class FakeMilvusClient:
    """记录建库与写入，便于观察流水线是否正常。"""

    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}
        self.last_schema: object | None = None

    def has_collection(self, collection_name: str, **kwargs) -> bool:
        return collection_name in self._collections

    def drop_collection(self, collection_name: str, **kwargs) -> None:
        self._collections.pop(collection_name, None)

    def create_collection(self, collection_name: str, schema=None, **kwargs) -> None:
        self.last_schema = schema
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


def _print_chunk_stats(chunks: list[ManualChunk], *, label: str) -> None:
    lengths = [len(c.text) for c in chunks]
    print(f"[chunk-stats:{label}] n={len(chunks)}")
    if not lengths:
        return
    print(
        f"[chunk-stats:{label}] len min={min(lengths)} max={max(lengths)} "
        f"avg={sum(lengths)/len(lengths):.1f}"
    )


def test_ingestion_on_sample_manual() -> None:
    """观察真实手册切块是否正常（需要 langchain-text-splitters）。"""
    try:
        import langchain_text_splitters  # noqa: F401
    except ModuleNotFoundError:
        print("[SKIP] 未安装 langchain-text-splitters，跳过真实切块观察。")
        return

    manual_path = _ROOT / "手册" / "冰箱手册.txt"
    if not manual_path.is_file():
        print(f"[SKIP] 未找到样例手册: {manual_path}")
        return

    svc = ManualIngestionService(str(_ROOT / "手册"))
    chunks = svc.parse_one_file(manual_path)
    assert chunks, "切块结果为空"
    _print_chunk_stats(chunks, label="冰箱手册")
    assert all(c.chunk_id for c in chunks)
    assert all(c.manual_name == manual_path.stem for c in chunks)


def test_milvus_schema_bm25_function_mocked() -> None:
    """观察 milvus_create.build_collection 是否挂上 BM25 function。

    说明：这里 **不连接 Milvus 服务端**，只走 pymilvus 的 schema 构建与
    ``create_collection(..., schema=...)`` 传参；失败与「服务端是否支持 BM25」无关。

    以前用 patch 替换 ``MilvusClient.create_schema`` 时，在部分 pymilvus 版本里
    ``create_schema`` 是 classmethod/descriptor，patch 方式不当会导致「字段都加在假
    schema 上，但后续逻辑未走到假对象的 ``add_function``」。这里改为使用真实的
    ``MilvusClient.create_schema``，并在 Fake client 上捕获最终 schema 再断言。
    """
    try:
        import pymilvus  # noqa: F401
    except ModuleNotFoundError:
        print("[SKIP] 未安装 pymilvus，跳过 schema/BM25 function 观察。")
        return

    from app.services import milvus_create

    fake = FakeMilvusClient()
    dim = 8

    try:
        milvus_create.build_collection(fake, vector_dim=dim)
    except AttributeError as exc:
        print(
            "[SKIP] 当前 pymilvus 的 CollectionSchema 可能不支持 add_function / BM25 "
            f"（{exc}）。建议 pymilvus >= 2.5 且与 Milvus 服务端版本匹配。"
        )
        return

    schema = fake.last_schema
    assert schema is not None, "create_collection 未收到 schema"

    field_names: list[str] = []
    if hasattr(schema, "fields"):
        raw = getattr(schema, "fields")
        if raw and hasattr(raw[0], "name"):
            field_names = [f.name for f in raw]
        elif raw and isinstance(raw[0], dict):
            field_names = [f.get("name", "") for f in raw]

    if field_names:
        assert "chunk_id" in field_names
        assert "dense_vector" in field_names
        assert "sparse_vector" in field_names
        assert "text" in field_names
        assert "image_ids" in field_names
        assert "manual_name" in field_names

    funcs = getattr(schema, "functions", None)
    if not funcs and hasattr(schema, "to_dict"):
        funcs = (schema.to_dict() or {}).get("functions") or []
    assert funcs, (
        "schema 上未登记任何 function（例如 BM25）。请确认 "
        "`app/services/milvus_create.py` 的 `build_collection` 已调用 "
        "`schema.add_function(...)`，且 pymilvus 版本支持 FunctionType.BM25。"
    )


def test_build_index_pipeline_mocked() -> None:
    """观察 build_index.main 写入字段是否与当前 schema 对齐。"""
    try:
        import pymilvus  # noqa: F401
    except ModuleNotFoundError:
        print("[SKIP] 未安装 pymilvus，跳过 build_index 流水线 mock 观察。")
        return

    try:
        from scripts import build_index
    except ModuleNotFoundError as exc:
        print(f"[SKIP] build_index 依赖缺失（{exc}），跳过流水线 mock 观察。")
        return

    dim = 4
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.0] * dim
    mock_emb.embed_documents.side_effect = lambda texts: [[float(i)] * dim for i, _ in enumerate(texts)]

    chunk = ManualChunk(
        chunk_id="health_0000",
        manual_name="health",
        text="hello world",
        image_ids=["id1"],
    )
    mock_svc = MagicMock()
    mock_svc.parse_and_chunk.return_value = [chunk]

    fake = FakeMilvusClient()
    with (
        patch("scripts.build_index.OllamaEmbeddings", return_value=mock_emb),
        patch("scripts.build_index.ManualIngestionService", return_value=mock_svc),
        patch("scripts.build_index._milvus_client", return_value=fake),
        patch("scripts.build_index._probe_dim", return_value=dim),
    ):
        build_index.main()

    stored = fake._collections[settings.milvus_collection]["rows"]
    assert len(stored) == 1
    row = stored[0]
    assert row["chunk_id"] == "health_0000"
    assert row["manual_name"] == "health"
    assert len(row["dense_vector"]) == dim
    assert "sparse_vector" not in row, "BM25 function 场景下 insert 不应手写 sparse_vector"
    assert "id1" in row["image_ids"]
    ids = json.loads(row["image_ids"])
    assert ids == ["id1"]


def test_optional_live_milvus_smoke() -> None:
    """可选：连真实 Milvus，只做最小读写探活（默认跳过）。"""
    if os.environ.get("RUN_LIVE_MILVUS", "") != "1":
        print("[SKIP] 未设置 RUN_LIVE_MILVUS=1，跳过真实 Milvus 冒烟。")
        return

    try:
        import pymilvus  # noqa: F401
    except ModuleNotFoundError:
        print("[SKIP] 未安装 pymilvus，无法做真实 Milvus 冒烟。")
        return

    from pymilvus import MilvusClient
    from app.services import milvus_create

    client = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)
    name = f"health_smoke_{os.getpid()}"
    prev = settings.milvus_collection
    try:
        settings.milvus_collection = name
        if client.has_collection(collection_name=name):
            client.drop_collection(collection_name=name)

        milvus_create.build_collection(client, vector_dim=4)
        client.insert(
            collection_name=name,
            data=[
                {
                    "chunk_id": "smoke_0000",
                    "dense_vector": [0.1, 0.2, 0.3, 0.4],
                    "text": "空调 滤网 清洁 步骤",
                    "image_ids": json.dumps([], ensure_ascii=False),
                    "manual_name": "smoke",
                }
            ],
        )
        milvus_create._create_vector_index(client)
        print(f"[live] collection={name} insert+index ok")
    finally:
        settings.milvus_collection = prev


if __name__ == "__main__":
    test_ingestion_on_sample_manual()
    test_milvus_schema_bm25_function_mocked()
    test_build_index_pipeline_mocked()
    test_optional_live_milvus_smoke()
    print("[OK] test_index_health passed")
