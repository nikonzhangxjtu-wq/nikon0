"""Application configuration.

This module centralizes runtime configuration so the rest of the codebase
does not read environment variables directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`.

    Keep this class small and explicit in V1.
    Add fields only when they are used by real code paths.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    api_bearer_token: str = Field(default="replace_with_your_token", alias="API_BEARER_TOKEN")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    gen_model: str = Field(default="qwen2:latest", alias="GEN_MODEL")
    embed_model: str = Field(default="nomic-embed-text:latest", alias="EMBED_MODEL")

    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    milvus_db_name: str = Field(default="default", alias="MILVUS_DB_NAME")
    milvus_collection: str = Field(default="manual_chunks_v1", alias="MILVUS_COLLECTION")

    manual_dir: str = Field(default="./手册", alias="MANUAL_DIR")


settings = Settings()
