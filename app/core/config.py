"""应用配置。

集中从环境变量与 `.env` 读取配置，其它模块不要散落读取 `os.environ`。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置（环境变量 + 可选 `.env`）。

    V1 保持字段少而清晰；只有真实代码路径用到的项再往里加。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    api_bearer_token: str = Field(default="replace_with_your_token", alias="API_BEARER_TOKEN")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    gen_model: str = Field(default="qwen2:latest", alias="GEN_MODEL")
    # 中英嵌入：常用做法是两侧都设为同一多语言模型（如 bge-m3），维度一致、建库可一次批量嵌入。
    # 若拆分不同模型，须保证输出维数相同，或与 Milvus dense_vector dim、VECTOR_DIM 一致。
    embed_model_en: str = Field(default="bge-m3:latest", alias="EMBED_MODEL_EN")
    embed_model_zh: str = Field(default="bge-m3:latest", alias="EMBED_MODEL_ZH")
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_db_name: str = Field(default="default", alias="MILVUS_DB_NAME")
    milvus_collection: str = Field(default="manual_chunks_v1", alias="MILVUS_COLLECTION")
    # 是否启用 Milvus BM25 Function（内置 sparse_vector + 文本查询）。
    # Milvus Lite（URI 为本地 .db 文件）目前对 BM25 Function 支持有限，保守默认关闭。
    milvus_enable_bm25: bool = Field(default=True, alias="MILVUS_ENABLE_BM25")
    vector_dim: int = Field(default=768, alias="VECTOR_DIM")
    manual_dir: str = Field(default="./手册", alias="MANUAL_DIR")
    generator_timeout: int = Field(default=10, alias="GENERATOR_TIMEOUT")
    # TODO：后续需要确定该阈值的具体大小
    retriever_context_filter_score_threshold: float = Field(default=0.3, alias="RETRIEVER_CONTEXT_FILTER_SCORE_THRESHOLD")

    # Cross-encoder 精排（需 langchain-community + sentence-transformers + torch）
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        alias="RERANK_MODEL_NAME",
    )
    # True：仅从 Hugging Face 缓存加载，不发起联网校验（需缓存已完整；否则加载会失败）
    rerank_hf_local_files_only: bool = Field(
        default=False,
        alias="RERANK_HF_LOCAL_FILES_ONLY",
    )


settings = Settings()
