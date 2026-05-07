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
    gen_model: str = Field(default="deepseek-v4-flash", alias="GEN_MODEL")
    # 百炼 API（OpenAI 兼容）—— 设 BAILIAN_API_KEY 即启用；空则不启用仍走 Ollama
    bailian_api_key: str = Field(default="", alias="BAILIAN_API_KEY")
    bailian_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="BAILIAN_BASE_URL",
    )
    # 生成时最大输出 token 数（Ollama: num_predict / 百炼: max_tokens）
    gen_max_tokens: int = Field(default=1024, alias="GEN_MAX_TOKENS")
    # 简单 LLM 场景（路由仲裁、手册名识别、视觉摘要）仍走本地 Ollama
    simple_llm_model: str = Field(default="qwen2:latest", alias="SIMPLE_LLM_MODEL")
    # 多模态（用户上传图片）理解：开启后会尝试生成图片摘要并参与路由/检索/生成。
    vision_enabled: bool = Field(default=True, alias="VISION_ENABLED")
    # 为空时回退到 GEN_MODEL（若该模型不支持视觉输入，会自动降级为无图摘要）。
    vision_model: str = Field(default="", alias="VISION_MODEL")
    vision_max_images: int = Field(default=3, alias="VISION_MAX_IMAGES")
    vision_summary_max_chars: int = Field(default=320, alias="VISION_SUMMARY_MAX_CHARS")
    # True：低温 + 空回答重试，贴近赛题稳定性；本地调试可设 GEN_COMPETITION_MODE=false
    gen_competition_mode: bool = Field(default=True, alias="GEN_COMPETITION_MODE")
    gen_max_retries: int = Field(default=3, alias="GEN_MAX_RETRIES")
    gen_temperature_competition: float = Field(default=0.1, alias="GEN_TEMPERATURE_COMPETITION")
    gen_temperature_casual: float = Field(default=0.2, alias="GEN_TEMPERATURE_CASUAL")
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
    retriever_context_filter_score_threshold: float = Field(default=0.1, alias="RETRIEVER_CONTEXT_FILTER_SCORE_THRESHOLD")

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

    # unknown 领域时是否用 LLM 仲裁是否走 RAG（仅采纳 needs_rag + reason）
    router_unknown_llm_arbitrate: bool = Field(
        default=False,
        alias="ROUTER_UNKNOWN_LLM_ARBITRATE",
    )
    # 空字符串表示与 GEN_MODEL 相同
    router_arbiter_model: str = Field(default="", alias="ROUTER_ARBITER_MODEL")

    # ---- LLM Router（替代关键词启发式）----
    # True: 用 LLM 做路由分类；False: 回退到关键词启发式
    router_llm_enabled: bool = Field(default=True, alias="ROUTER_LLM_ENABLED")
    # 路由分类用模型（本地 qwen2 足够做二分类）
    router_llm_model: str = Field(default="qwen2:latest", alias="ROUTER_LLM_MODEL")

    # ---- ReAct Agent（多轮迭代检索）----
    # True: 用 ReAct 多轮检索；False: 走原单次检索
    react_enabled: bool = Field(default=True, alias="REACT_ENABLED")
    # 最多 SEARCH 几轮（含首轮），达到上限后强制进入最终生成
    react_max_iterations: int = Field(default=3, alias="REACT_MAX_ITERATIONS")
    # Agent 循环用模型（百炼 API 优先时用 deepseek-v4-flash；Ollama 回退时用 qwen2）
    react_agent_model: str = Field(default="deepseek-v4-flash", alias="REACT_AGENT_MODEL")

    # ---- Online Review Skill（联网口碑）----
    # True: 开启口碑 skill 分支；False: 忽略该分支并走普通 no-rag
    online_review_skill_enabled: bool = Field(
        default=True,
        alias="ONLINE_REVIEW_SKILL_ENABLED",
    )
    # 调用 provider 的默认返回上限
    online_review_top_k: int = Field(default=8, alias="ONLINE_REVIEW_TOP_K")
    # 口碑检索来源：local=本地评价表（默认）；mcp=HTTP MCP；none=不检索
    review_search_mode: str = Field(default="local", alias="REVIEW_SEARCH_MODE")
    # 可选：追加 JSON 评价表（数组），与内置数据合并；空则仅内置
    local_review_table_path: str = Field(default="", alias="LOCAL_REVIEW_TABLE_PATH")
    # MCP Review Provider：HTTP bridge endpoint（仅 review_search_mode=mcp 时使用）
    mcp_review_endpoint: str = Field(default="", alias="MCP_REVIEW_ENDPOINT")
    # 可选：MCP bridge 的 Bearer token
    mcp_review_api_key: str = Field(default="", alias="MCP_REVIEW_API_KEY")
    # MCP bridge 超时（秒）
    mcp_review_timeout_sec: int = Field(default=12, alias="MCP_REVIEW_TIMEOUT_SEC")
    # ---- Order Status Skill（订单进度）----
    # True：开启订单状态 skill 分支（MCP 订单查询）
    order_status_skill_enabled: bool = Field(
        default=True,
        alias="ORDER_STATUS_SKILL_ENABLED",
    )
    # 调用订单 provider 的默认返回上限
    order_status_top_k: int = Field(default=3, alias="ORDER_STATUS_TOP_K")
    # MCP Order Provider：HTTP bridge endpoint（留空则不启用，回退 Null provider）
    mcp_order_endpoint: str = Field(default="", alias="MCP_ORDER_ENDPOINT")
    # MCP server 中用于订单查询的 tool 名（默认 get_order_status）
    mcp_order_tool_name: str = Field(default="get_order_status", alias="MCP_ORDER_TOOL_NAME")
    # 可选：MCP order bridge 的 Bearer token
    mcp_order_api_key: str = Field(default="", alias="MCP_ORDER_API_KEY")
    # MCP order bridge 超时（秒）
    mcp_order_timeout_sec: int = Field(default=12, alias="MCP_ORDER_TIMEOUT_SEC")
    # ---- Case Intake Skill（售后受理）----
    case_intake_skill_enabled: bool = Field(
        default=True,
        alias="CASE_INTAKE_SKILL_ENABLED",
    )
    # True：工单收集状态持久化到 Redis（失败自动回退内存）
    case_intake_redis_enabled: bool = Field(
        default=True,
        alias="CASE_INTAKE_REDIS_ENABLED",
    )
    # 例：redis://127.0.0.1:6379/0；逻辑库 2 为 .../2（路径里是十进制库号，不是 /02）
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    redis_case_intake_key_prefix: str = Field(
        default="kf",
        alias="REDIS_CASE_INTAKE_KEY_PREFIX",
    )
    # Case intake 状态 TTL（秒），与会话长度对齐即可
    case_intake_redis_ttl_seconds: int = Field(
        default=3600,
        alias="CASE_INTAKE_REDIS_TTL_SECONDS",
    )
    # True：工单收集用 ReAct（本地 Ollama 多步 THOUGHT/ACTION）；False：仅用规则槽位（默认）
    case_intake_react_enabled: bool = Field(default=False, alias="CASE_INTAKE_REACT_ENABLED")
    # 每轮用户发言内 ReAct 最大 LLM 步数
    case_intake_react_max_steps: int = Field(default=4, alias="CASE_INTAKE_REACT_MAX_STEPS")
    # 为空则使用 SIMPLE_LLM_MODEL
    case_intake_react_model: str = Field(default="", alias="CASE_INTAKE_REACT_MODEL")

    # ---- 多轮对话记忆 ----
    # True：对话轮次持久化到 Redis（失败自动回退内存 ConversationStore）
    conversation_redis_enabled: bool = Field(
        default=True,
        alias="CONVERSATION_REDIS_ENABLED",
    )
    redis_conversation_key_prefix: str = Field(
        default="kf",
        alias="REDIS_CONVERSATION_KEY_PREFIX",
    )
    # 最多保留几轮历史
    conversation_max_turns: int = Field(default=5, alias="CONVERSATION_MAX_TURNS")
    # Session TTL（秒），过期后自动清理
    conversation_ttl_seconds: int = Field(default=3600, alias="CONVERSATION_TTL_SECONDS")
    # 用轻量 LLM 将省略指代的多轮问题改写为自包含检索查询
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")

    # Pipeline：路由置信度低于该值时记为低置信（写入 debug / prompt，不改变是否检索）
    pipeline_route_low_confidence_threshold: float = Field(
        default=0.45,
        alias="PIPELINE_ROUTE_LOW_CONFIDENCE_THRESHOLD",
    )
    # 检索后至少保留几条过滤后的 chunk 才走标准 rag_manual；1 表示仅「无结果」才闸门
    pipeline_min_filtered_chunks_for_rag: int = Field(
        default=1,
        alias="PIPELINE_MIN_FILTERED_CHUNKS_FOR_RAG",
    )


settings = Settings()
