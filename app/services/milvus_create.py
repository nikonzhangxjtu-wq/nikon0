"""Milvus collection / index 定义。

关于 sparse 与 BM25：

- Milvus 2.5+ 提供「BM25 Function」：由服务端根据 `text` 字段自动生成 sparse 向量，
  检索时可以直接传文本。但在 **Milvus Lite（本地 .db 文件）** 上当前对 BM25
  Function 的支持有限，开启后容易出现
  ``fieldName(dense_vector) not found`` 或 ``search_data ... illegal`` 这类奇怪报错。
- 因此这里用 ``settings.milvus_enable_bm25`` 控制是否启用 BM25 Function / sparse
  字段；默认 False，在 Milvus Lite 上也能正常建库、检索（仅 dense 一路）。
"""

from __future__ import annotations
from gc import enable

from pymilvus import DataType, MilvusClient

from app.core.config import settings

TEXT_MAX_LEN = 16384
IMAGE_IDS_MAX_LEN = 8192
IMAGE_META_MAX_LEN = 8192
MEMORY_TEXT_MAX_LEN = 4096


def _collection_field_names(client: MilvusClient, collection_name: str) -> set[str]:
    """describe_collection 返回的字段名集合。"""
    desc = client.describe_collection(collection_name=collection_name)
    fields_raw = desc.get("fields") if isinstance(desc, dict) else getattr(desc, "fields", None)
    names: set[str] = set()
    if not fields_raw:
        return names
    for f in fields_raw:
        if isinstance(f, dict):
            n = f.get("name")
            if n:
                names.add(str(n))
        else:
            n = getattr(f, "name", None)
            if n:
                names.add(str(n))
    return names


def build_collection(client: MilvusClient, vector_dim: int) -> None:
    """(Re)create the target collection with the current schema.

    - 始终包含: chunk_id / dense_vector / text / image_ids / manual_name
    - 当 ``settings.milvus_enable_bm25`` 为 True 时, 额外添加 sparse_vector 字段,
      并通过 ``schema.add_function`` 挂上 BM25 Function(text -> sparse_vector).
    """
    name = settings.milvus_collection
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        is_primary=True,
    )
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
    
    # 🌟 修改点：增加 analyzer_params 显式指定分词器为 chinese
    schema.add_field(
        field_name="text", 
        datatype=DataType.VARCHAR, 
        max_length=TEXT_MAX_LEN,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"} 
    )
    
    schema.add_field(
        field_name="image_ids",
        datatype=DataType.VARCHAR,
        max_length=IMAGE_IDS_MAX_LEN,
    )
    schema.add_field(field_name="manual_name", datatype=DataType.VARCHAR, max_length=256)

    if settings.milvus_enable_bm25:
        # 仅在明确开启 BM25 Function 时添加 sparse 字段, 否则在 Milvus Lite 上容易
        # 出现「schema 注册但功能不完整」的情况.
        try:
            from pymilvus import Function, FunctionType

            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            bm25_fn = Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(bm25_fn)
        except Exception as exc:  # noqa: BLE001
            print(
                "[WARN] 当前 pymilvus/Milvus 不支持 BM25 Function, 已回退为 dense-only: "
                f"{exc}"
            )

    client.create_collection(collection_name=name, schema=schema)


def _create_vector_index(client: MilvusClient) -> None:
    """为当前 collection 建索引并 load。

    注意：``build_collection`` 里若 BM25 Function 导入失败会回退为 dense-only schema，
    此时集合里 **没有** ``sparse_vector`` 字段。这里必须以 **实际 schema** 为准，
    不能只看 ``settings.milvus_enable_bm25``，否则会出现
    ``cannot create index on non-existed field: sparse_vector``。
    """
    name = settings.milvus_collection
    field_names = _collection_field_names(client, name)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    if "sparse_vector" in field_names:
        try:
            # Milvus 2.6+ 对 BM25 Function 的输出 sparse 字段要求 metric_type="BM25",
            # 使用旧的 IP 会报 "index metric type of BM25 function output field must be BM25".
            # params 里 bm25_k1 / bm25_b 采用 Milvus 官方推荐默认值.
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"bm25_k1": 1.2, "bm25_b": 0.75},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] sparse_vector 索引添加失败, 忽略: {exc}")
    elif settings.milvus_enable_bm25:
        print(
            "[WARN] 已开启 MILVUS_ENABLE_BM25，但集合中无 sparse_vector 字段 "
            "（多为 pymilvus 过旧无法 import Function）。请升级 pymilvus 与 Milvus "
            "服务端版本匹配后再重建；当前仅创建 dense 索引。"
        )

    index_params.add_index(
        field_name="manual_name",
        index_type="TRIE",
    )

    client.create_index(collection_name=settings.milvus_collection, index_params=index_params)
    client.load_collection(collection_name=settings.milvus_collection)


def build_image_collection(client: MilvusClient, vector_dim: int | None = None) -> None:
    """(Re)create 手册图片 collection。

    图片与文本 chunk 是不同粒度的知识单元：图片 collection 只保存图片级证据和
    parent_chunk_ids，不修改原有 manual_chunks_v1，保证纯文本 RAG 主链路稳定。
    """
    name = settings.multimodal_image_collection
    dim = vector_dim or settings.multimodal_image_vector_dim
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="image_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        is_primary=True,
    )
    schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="semantic_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="manual_name", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="parent_chunk_ids", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="parent_context_text", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="context_intent", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="image_type", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="semantic_text", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="ocr_text", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="visual_entities", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="operation_steps", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    schema.add_field(field_name="warnings", datatype=DataType.VARCHAR, max_length=IMAGE_META_MAX_LEN)
    client.create_collection(collection_name=name, schema=schema)


def create_image_vector_index(client: MilvusClient) -> None:
    """为手册图片 collection 建立向量与手册名索引并 load。"""
    name = settings.multimodal_image_collection
    index_params = MilvusClient.prepare_index_params()
    for field_name in ("image_vector", "semantic_vector"):
        index_params.add_index(
            field_name=field_name,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )
    index_params.add_index(field_name="manual_name", index_type="TRIE")
    client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(collection_name=name)


def build_user_memory_collection(client: MilvusClient, vector_dim: int | None = None) -> None:
    """(Re)create 长期用户情景记忆 collection。

    该 collection 只保存用户个人化/情景化记忆，不重复手册静态知识。
    """
    name = settings.memory_episodic_collection
    dim = vector_dim or settings.vector_dim
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="memory_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        is_primary=True,
    )
    schema.add_field(field_name="user_key", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="memory_text", datatype=DataType.VARCHAR, max_length=MEMORY_TEXT_MAX_LEN)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="source_session", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="created_at", datatype=DataType.DOUBLE)
    schema.add_field(field_name="expire_ts", datatype=DataType.DOUBLE)
    client.create_collection(collection_name=name, schema=schema)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    index_params.add_index(field_name="user_key", index_type="TRIE")
    index_params.add_index(field_name="expire_ts", index_type="STL_SORT")
    client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(collection_name=name)


def build_user_memory_v2_collection(client: MilvusClient, vector_dim: int | None = None) -> None:
    """(Re)create v3 episodic memory collection。

    v2/v3 长期记忆保存的是“事件”，不是对话全文摘要；metadata 字段用于按用户、
    事件类型、产品、工单号过滤，dense_vector 只负责语义召回。
    """
    name = settings.memory_v3_episodic_collection
    dim = vector_dim or settings.vector_dim
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="event_id",
        datatype=DataType.VARCHAR,
        max_length=256,
        is_primary=True,
    )
    schema.add_field(field_name="user_key", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="event_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=MEMORY_TEXT_MAX_LEN)
    schema.add_field(field_name="product_model", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="case_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="issue_thread_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="created_at", datatype=DataType.DOUBLE)
    schema.add_field(field_name="expire_ts", datatype=DataType.DOUBLE)
    client.create_collection(collection_name=name, schema=schema)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )
    index_params.add_index(field_name="user_key", index_type="TRIE")
    index_params.add_index(field_name="event_type", index_type="TRIE")
    index_params.add_index(field_name="product_model", index_type="TRIE")
    index_params.add_index(field_name="case_id", index_type="TRIE")
    index_params.add_index(field_name="expire_ts", index_type="STL_SORT")
    client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(collection_name=name)
