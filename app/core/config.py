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
    # 简单 LLM 场景（路由仲裁、手册名识别等）：百炼 API 优先，未配置 key 时回退 Ollama。
    simple_llm_model: str = Field(default="deepseek-v4-flash", alias="SIMPLE_LLM_MODEL")
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
    # 仅当手册名识别置信度达到该阈值时，检索才加 manual_name 标量过滤；
    # 否则走全库检索，避免错选手册导致召回完全跑偏。
    manual_name_filter_min_confidence: float = Field(
        default=0.7,
        alias="MANUAL_NAME_FILTER_MIN_CONFIDENCE",
    )
    # 复杂/多子问触发多查询检索的最大查询数；不改变 Milvus schema。
    retrieval_multi_query_max_queries: int = Field(
        default=4,
        alias="RETRIEVAL_MULTI_QUERY_MAX_QUERIES",
    )
    retrieval_multi_query_min_signals: int = Field(
        default=2,
        alias="RETRIEVAL_MULTI_QUERY_MIN_SIGNALS",
    )

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
    # softmax temperature：< 1.0 放大分数差异，解决 sigmoid 压缩问题
    rerank_temperature: float = Field(
        default=0.5,
        alias="RERANK_TEMPERATURE",
    )
    # 关键词重叠得分在精排混合中的权重（0~1），0=纯语义，0.3=30%关键词+70%语义
    rerank_keyword_weight: float = Field(
        default=0.3,
        alias="RERANK_KEYWORD_WEIGHT",
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
    router_llm_model: str = Field(default="deepseek-v4-flash", alias="ROUTER_LLM_MODEL")

    # ---- ReAct Agent（多轮迭代检索）----
    # True: 用 ReAct 多轮检索；False: 走原单次检索
    react_enabled: bool = Field(default=True, alias="REACT_ENABLED")
    # 最多 SEARCH 几轮（含首轮），达到上限后强制进入最终生成
    react_max_iterations: int = Field(default=3, alias="REACT_MAX_ITERATIONS")
    # Agent 循环用模型（百炼 API 优先时用 deepseek-v4-flash；Ollama 回退时用 qwen2）
    react_agent_model: str = Field(default="deepseek-v4-flash", alias="REACT_AGENT_MODEL")

    # ---- Deprecated legacy skill switches ----
    # 旧 order_status/web_review 分支已从 Pipeline 断开；这些字段仅用于兼容本地
    # .env 中残留变量，避免配置加载失败，不再被运行时 workflow 读取。
    online_review_skill_enabled: bool = Field(
        default=False,
        alias="ONLINE_REVIEW_SKILL_ENABLED",
    )
    online_review_top_k: int = Field(default=8, alias="ONLINE_REVIEW_TOP_K")
    review_search_mode: str = Field(default="", alias="REVIEW_SEARCH_MODE")
    local_review_table_path: str = Field(default="", alias="LOCAL_REVIEW_TABLE_PATH")
    order_status_skill_enabled: bool = Field(
        default=False,
        alias="ORDER_STATUS_SKILL_ENABLED",
    )
    order_status_top_k: int = Field(default=3, alias="ORDER_STATUS_TOP_K")

    # ---- Case Intake Skill（售后受理）----
    case_intake_skill_enabled: bool = Field(
        default=True,
        alias="CASE_INTAKE_SKILL_ENABLED",
    )
    # local：直接调用本进程 CaseIntakeSkill；gateway：通过 MCP Gateway 调用 case-intake 服务。
    case_intake_provider: str = Field(default="local", alias="CASE_INTAKE_PROVIDER")
    mcp_gateway_endpoint: str = Field(default="http://127.0.0.1:18080/mcp", alias="MCP_GATEWAY_ENDPOINT")
    mcp_gateway_bearer_token: str = Field(default="", alias="MCP_GATEWAY_BEARER_TOKEN")
    mcp_gateway_timeout_sec: int = Field(default=15, alias="MCP_GATEWAY_TIMEOUT_SEC")
    mcp_case_intake_service_id: str = Field(default="case-intake", alias="MCP_CASE_INTAKE_SERVICE_ID")
    mcp_case_intake_collect_tool: str = Field(
        default="collect_case_intake",
        alias="MCP_CASE_INTAKE_COLLECT_TOOL",
    )
    mcp_case_intake_status_tool: str = Field(
        default="get_case_intake_status",
        alias="MCP_CASE_INTAKE_STATUS_TOOL",
    )
    mcp_case_intake_cancel_tool: str = Field(
        default="try_cancel_case_intake",
        alias="MCP_CASE_INTAKE_CANCEL_TOOL",
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

    # ---- Context Assembler / Token 压缩 ----
    # 这里先用轻量 token 估算，不引入真实 tokenizer；目标是稳定控制上下文规模。
    context_assembler_enabled: bool = Field(default=True, alias="CONTEXT_ASSEMBLER_ENABLED")
    context_total_token_budget: int = Field(default=6000, alias="CONTEXT_TOTAL_TOKEN_BUDGET")
    context_rag_token_budget: int = Field(default=3200, alias="CONTEXT_RAG_TOKEN_BUDGET")
    context_history_token_budget: int = Field(default=900, alias="CONTEXT_HISTORY_TOKEN_BUDGET")
    context_visual_token_budget: int = Field(default=500, alias="CONTEXT_VISUAL_TOKEN_BUDGET")
    context_memory_token_budget: int = Field(default=600, alias="CONTEXT_MEMORY_TOKEN_BUDGET")
    nikon0_context_llm_enabled: bool = Field(default=True, alias="NIKON0_CONTEXT_LLM_ENABLED")
    nikon0_context_llm_model: str = Field(default="", alias="NIKON0_CONTEXT_LLM_MODEL")
    nikon0_context_llm_timeout: int = Field(default=15, alias="NIKON0_CONTEXT_LLM_TIMEOUT")
    nikon0_context_llm_max_tokens: int = Field(default=512, alias="NIKON0_CONTEXT_LLM_MAX_TOKENS")
    nikon0_context_total_char_budget: int = Field(default=9000, alias="NIKON0_CONTEXT_TOTAL_CHAR_BUDGET")
    nikon0_enable_mock_skill: bool = Field(default=False, alias="NIKON0_ENABLE_MOCK_SKILL")

    # ---- Agent Memory ----
    memory_enabled: bool = Field(default=False, alias="MEMORY_ENABLED")
    memory_version: str = Field(default="v1", alias="MEMORY_VERSION")
    memory_v3_llm_judge_enabled: bool = Field(default=True, alias="MEMORY_V3_LLM_JUDGE_ENABLED")
    memory_v3_llm_judge_model: str = Field(default="", alias="MEMORY_V3_LLM_JUDGE_MODEL")
    memory_v3_llm_judge_max_tokens: int = Field(default=512, alias="MEMORY_V3_LLM_JUDGE_MAX_TOKENS")
    memory_v3_llm_judge_timeout: int = Field(default=20, alias="MEMORY_V3_LLM_JUDGE_TIMEOUT")
    memory_v3_episodic_collection: str = Field(
        default="user_memory_v2",
        alias="MEMORY_V3_EPISODIC_COLLECTION",
    )
    memory_v4_store: str = Field(default="redis", alias="MEMORY_V4_STORE")
    memory_v4_llm_judge_enabled: bool = Field(default=True, alias="MEMORY_V4_LLM_JUDGE_ENABLED")
    memory_session_summary_trigger_turns: int = Field(
        default=6,
        alias="MEMORY_SESSION_SUMMARY_TRIGGER_TURNS",
    )
    memory_user_profile_enabled: bool = Field(default=True, alias="MEMORY_USER_PROFILE_ENABLED")
    memory_user_profile_ttl_seconds: int = Field(
        default=90 * 24 * 3600,
        alias="MEMORY_USER_PROFILE_TTL_SECONDS",
    )
    memory_episodic_enabled: bool = Field(default=True, alias="MEMORY_EPISODIC_ENABLED")
    memory_episodic_collection: str = Field(
        default="user_memory_v1",
        alias="MEMORY_EPISODIC_COLLECTION",
    )
    memory_episodic_top_k: int = Field(default=3, alias="MEMORY_EPISODIC_TOP_K")
    memory_episodic_score_threshold: float = Field(
        default=0.35,
        alias="MEMORY_EPISODIC_SCORE_THRESHOLD",
    )
    memory_consolidation_async: bool = Field(default=True, alias="MEMORY_CONSOLIDATION_ASYNC")
    memory_consolidation_every_turns: int = Field(
        default=4,
        alias="MEMORY_CONSOLIDATION_EVERY_TURNS",
    )
    nikon0_memory_store: str = Field(default="memory", alias="NIKON0_MEMORY_STORE")
    nikon0_memory_redis_url: str = Field(default="", alias="NIKON0_MEMORY_REDIS_URL")
    nikon0_memory_mysql_dsn: str = Field(default="", alias="NIKON0_MEMORY_MYSQL_DSN")
    nikon0_memory_redis_prefix: str = Field(default="nikon0:memory", alias="NIKON0_MEMORY_REDIS_PREFIX")
    nikon0_memory_ttl_seconds: int = Field(default=86400, alias="NIKON0_MEMORY_TTL_SECONDS")
    nikon0_memory_write_gate_enabled: bool = Field(default=True, alias="NIKON0_MEMORY_WRITE_GATE_ENABLED")
    nikon0_memory_llm_planner_enabled: bool = Field(default=True, alias="NIKON0_MEMORY_LLM_PLANNER_ENABLED")
    nikon0_memory_llm_planner_model: str = Field(default="", alias="NIKON0_MEMORY_LLM_PLANNER_MODEL")
    nikon0_memory_llm_planner_timeout: int = Field(default=12, alias="NIKON0_MEMORY_LLM_PLANNER_TIMEOUT")
    nikon0_memory_llm_planner_max_tokens: int = Field(default=512, alias="NIKON0_MEMORY_LLM_PLANNER_MAX_TOKENS")
    nikon0_memory_persistence_strict: bool = Field(default=False, alias="NIKON0_MEMORY_PERSISTENCE_STRICT")

    # ---- Multimodal Manual Images V2 ----
    # True：启用手册图片 collection 召回；失败时自动回退纯文本 RAG。
    multimodal_image_retrieval_enabled: bool = Field(
        default=True,
        alias="MULTIMODAL_IMAGE_RETRIEVAL_ENABLED",
    )
    # attached_only: 只把文本 top-k chunk 自带图片挂为视觉证据；vector_search: 旧图片向量召回。
    multimodal_image_retrieval_mode: str = Field(
        default="attached_only",
        alias="MULTIMODAL_IMAGE_RETRIEVAL_MODE",
    )
    manual_image_dir: str = Field(default="./手册/插图", alias="MANUAL_IMAGE_DIR")
    manual_image_cache_path: str = Field(
        default="./app/data/manual_image_understanding_cache.json",
        alias="MANUAL_IMAGE_CACHE_PATH",
    )
    manual_image_report_path: str = Field(
        default="./app/data/manual_image_asset_report.json",
        alias="MANUAL_IMAGE_REPORT_PATH",
    )
    manual_image_vlm_model: str = Field(
        default="qwen3-vl-flash",
        alias="MANUAL_IMAGE_VLM_MODEL",
    )
    # build_image_index：是否打印/落盘 VLM 结构理解结果（2597 张时建议保留 jsonl、关闭逐条控制台输出）
    manual_image_vlm_log_enabled: bool = Field(default=True, alias="MANUAL_IMAGE_VLM_LOG_ENABLED")
    manual_image_vlm_log_console: bool = Field(default=True, alias="MANUAL_IMAGE_VLM_LOG_CONSOLE")
    manual_image_vlm_log_path: str = Field(
        default="./app/data/manual_image_vlm_index.jsonl",
        alias="MANUAL_IMAGE_VLM_LOG_PATH",
    )
    # 兼容计划文档里的通用变量名；MANUAL_IMAGE_VLM_MODEL 优先，VLM_MODEL 作为兜底。
    vlm_model: str = Field(
        default="",
        alias="VLM_MODEL",
    )
    multimodal_embed_provider: str = Field(default="jina_api", alias="MULTIMODAL_EMBED_PROVIDER")
    multimodal_embed_model: str = Field(default="jina-clip-v2", alias="MULTIMODAL_EMBED_MODEL")
    # 百炼多模态 embedding 输出维度（qwen3-vl-embedding 支持 1024 等；multimodal-embedding-v1 固定 1024）
    multimodal_embed_dimension: int = Field(default=1024, alias="MULTIMODAL_EMBED_DIMENSION")
    # 百炼多模态 embedding（DashScope API）配置。若 DASHSCOPE_API_KEY 为空则回退 BAILIAN_API_KEY。
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    dashscope_multimodal_embedding_endpoint: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        alias="DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT",
    )
    dashscope_embedding_timeout_sec: int = Field(default=60, alias="DASHSCOPE_EMBEDDING_TIMEOUT_SEC")
    multimodal_image_collection: str = Field(
        default="manual_images_v1",
        alias="MULTIMODAL_IMAGE_COLLECTION",
    )
    multimodal_image_vector_dim: int = Field(
        default=1024,
        alias="MULTIMODAL_IMAGE_VECTOR_DIM",
    )
    jina_api_key: str = Field(default="", alias="JINA_API_KEY")
    jina_embedding_endpoint: str = Field(
        default="https://api.jina.ai/v1/embeddings",
        alias="JINA_EMBEDDING_ENDPOINT",
    )
    jina_embedding_timeout_sec: int = Field(default=60, alias="JINA_EMBEDDING_TIMEOUT_SEC")
    multimodal_image_top_k: int = Field(default=4, alias="MULTIMODAL_IMAGE_TOP_K")
    multimodal_entity_top_k: int = Field(default=8, alias="MULTIMODAL_ENTITY_TOP_K")
    multimodal_image_min_score: float = Field(default=0.15, alias="MULTIMODAL_IMAGE_MIN_SCORE")


settings = Settings()
