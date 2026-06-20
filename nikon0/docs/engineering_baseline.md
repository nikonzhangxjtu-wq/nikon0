# Engineering Baseline

本文记录 nikon0 的工程化基线。参考 `/Users/nikonzhang/project/fastapi-langgraph-agent-production-ready-template/` 的生产化组织，但 nikon0 不把 LangGraph 作为核心依赖。

## 技术栈建议

```text
API: FastAPI
Schema: Pydantic
Config: pydantic-settings + .env.development/.env.production
DB: PostgreSQL，用于用户、会话、业务记录、trace 索引
Cache/Session: Redis，用于 session issue memory、限流、短期缓存
Vector: Milvus，仅用于必要的知识/图片索引 backend
Gateway: 自研 MCP Gateway
Observability: structlog + Prometheus/Grafana，后续可接 Langfuse/OpenTelemetry
Eval: 独立 eval runner + golden cases + trace replay
```

## 推荐目录

```text
nikon0/
  app/
    main.py
    api/
      v1/
        chat.py
        sessions.py
        health.py
    core/
      config.py
      logging.py
      middleware.py
      metrics.py
      security.py
    schemas/
      agent.py
      skill.py
      tool.py
      knowledge.py
      memory.py
    models/
      session.py
      trace.py
      business.py
    services/
      agent_runtime.py
      session_service.py
      trace_service.py
  agent/
  skills/
  tools/
  knowledge/
  memory/
  safety/
  eval/
  infra/
```

## API 第一版

```text
POST /api/v1/chat
GET  /api/v1/sessions/{session_id}
GET  /api/v1/sessions/{session_id}/trace/{trace_id}
POST /api/v1/hitl/approvals/{approval_id}
GET  /health
```

## 配置原则

所有运行环境差异都来自环境变量。

第一批配置：

```text
APP_ENV=development
PROJECT_NAME=nikon0
LOG_LEVEL=INFO
LOG_FORMAT=console

LLM_PROVIDER=openai_compatible
LLM_MODEL=deepseek-v4-flash
LLM_TOTAL_TIMEOUT=60

MCP_GATEWAY_URL=http://localhost:8090/mcp
MCP_GATEWAY_AUTH_TOKEN=alice

SESSION_STORE=redis
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=86400

NIKON0_MEMORY_STORE=redis_mysql
NIKON0_MEMORY_REDIS_URL=redis://localhost:6379/2
NIKON0_MEMORY_MYSQL_DSN=mysql+pymysql://nikon0:nikon0@localhost:3306/nikon0?charset=utf8mb4
NIKON0_MEMORY_REDIS_PREFIX=nikon0:memory
NIKON0_MEMORY_TTL_SECONDS=86400

NIKON0_CONTEXT_LLM_ENABLED=true
NIKON0_CONTEXT_LLM_MODEL=deepseek-v4-flash
NIKON0_CONTEXT_LLM_TIMEOUT=15
NIKON0_CONTEXT_LLM_MAX_TOKENS=512
NIKON0_CONTEXT_TOTAL_CHAR_BUDGET=9000

KNOWLEDGE_BACKENDS=structured,playbook,fulltext,rag
RAG_ENABLED=false

SAFETY_HITL_ENABLED=true
```

## Memory 持久化

`nikon0` 的 P0 记忆持久化采用 Redis + MySQL：

```text
Redis: hot session snapshot
MySQL: durable session snapshot + append-only StateUpdate event audit
```

运行时配置：

```text
NIKON0_MEMORY_STORE=redis_mysql
NIKON0_MEMORY_REDIS_URL=redis://localhost:6379/2
NIKON0_MEMORY_MYSQL_DSN=mysql+pymysql://nikon0:nikon0@localhost:3306/nikon0?charset=utf8mb4
NIKON0_MEMORY_REDIS_PREFIX=nikon0:memory
NIKON0_MEMORY_TTL_SECONDS=86400
```

MySQL 表由 `SqlMemoryPersistence` 启动时自动创建：

```text
nikon0_memory_sessions      当前 session memory snapshot
nikon0_state_update_events  每次 StateUpdate 的追加事件，供审计和 replay
```

未配置或连接失败时，`build_default_runtime()` 会回退到 `InMemorySessionIssueStore`，便于本地测试继续运行。

## Context Pack 第一阶段

`nikon0` 的上下文管理第一阶段采用统一 `ContextPack`：

```text
ContextRuntime
  -> assemble named sections
  -> apply deterministic budgets
  -> render governed_context
  -> record budget trace
```

当前 section：

```text
system_policy
workflow
memory
conversation
tool_observations
evidence
current_user
runtime
```

第一阶段边界：

```text
不做 LLM compaction
不把 raw tool result 长期塞进 prompt
prompt 统一消费 context_pack
```

第二步已加入 `EvidenceContextManager`：

```text
RAG evidence -> sort by confidence -> deduplicate -> raw excerpt trimming -> preserve metadata
```

证据上下文原则：

```text
不默认总结 RAG chunk
优先保留 raw_excerpt
保留 manual/page/chunk/product/version 等 source metadata
记录 retrieved / included / deduplicated evidence ids
超长证据先截取相关原文片段，不做自由摘要
```

第三步已加入 `ToolObservationManager`：

```text
tool result -> prompt observation
raw result -> trace/storage ref
```

工具上下文原则：

```text
不把 raw tool result 直接塞进 prompt
保留 tool/status/summary/data_keys/error
保留 raw_result_ref，例如 trace://{trace_id}/tool_results/{index}
大字段只进入 trace/DB，不进入模型窗口
错误工具调用也进入 observation，便于模型解释当前状态
```

第四步已加入 `ConversationCompactor`：

```text
long transcript -> issue-local summary + recent raw turns
```

对话上下文原则：

```text
短历史保留原文
长历史保留最近若干行原文
旧历史生成结构化 issue-local 摘要
摘要只抽取原文中显著事实，不自由编造
记录 context.conversation_compact trace 事件
后续可用 LLM compactor 替换摘要生成，但 Runtime 仍负责预算和校验
```

第五步已加入 `ContextReadPlanner`：

```text
current request + memory preview + transcript preview
  -> decide included sections
  -> ContextRuntime only assembles selected sections
```

读取规划原则：

```text
默认使用 DeterministicContextReadPlanner
闲聊不引入 evidence/workflow/tool_observations
商品手册/故障码/安装/清洁类问题引入 evidence
报修/退款/投诉/转人工/审批类流程引入 workflow + tool_observations
继续/刚才/那个问题等指代场景保留 memory + conversation
每次规划写入 context.read_plan trace 事件
```

LLM 规划器：

```text
LlmContextReadPlanner
prompt: nikon0/context/read_planner_prompt.py
输出严格 JSON: included_sections / reasons / confidence
LLM 输出异常或低置信时回退 deterministic planner
```

第六步已升级 `ContextBudgeter`：

```text
sections -> per-section budget -> total budget degradation -> budget report
```

预算治理原则：

```text
section priority 越小越受保护
降级顺序从低优先级 section 开始
current_user / system_policy 默认不丢弃
报告包含 section_priorities / degradation_order / degraded_sections / dropped_sections
预算裁剪只做形态治理，不改变业务事实
```

第七步已加入 LLM 增强组件：

```text
LlmConversationCompactor
  -> 生成 issue-local summary_lines
  -> 最近对话仍由 Runtime 保留原文
  -> LLM 失败回退 deterministic ConversationCompactor

LlmEvidenceSpanSelector
  -> LLM 只返回 start/end 字符下标
  -> Runtime 只截取原文 raw span
  -> 不接受自由 summary，不改写证据
  -> LLM 失败回退 deterministic span selection
```

默认运行：

```text
build_default_runtime()
  -> NIKON0_CONTEXT_LLM_ENABLED=true 时启用：
     LlmContextReadPlanner
     LlmConversationCompactor
     LlmEvidenceSpanSelector
  -> 任一 LLM 节点异常时回退 deterministic
```

关闭方式：

```text
NIKON0_CONTEXT_LLM_ENABLED=false
```

后续可以在 `ContextRuntime` 前后加入：

```text
LLMContextReadPlanner
Issue-local ConversationCompactor
EvidenceSpanSelector
ToolObservationNormalizer
```

## 生产化底线

- 每个请求有 `trace_id`。
- 每次 tool call 有输入、输出、耗时和错误码。
- 每次 memory update 有 reason 和 evidence。
- 每次高风险决策有 safety decision。
- 所有外部服务调用有 timeout。
- LLM 输出结构化 JSON 时必须 schema 校验。
- 禁止把 API key 写入代码或文档示例值。
- 测试必须覆盖 skill 选择、工具调用、记忆更新、安全拦截。
