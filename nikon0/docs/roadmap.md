# Roadmap

## Phase 0: 架构定型

目标：建立 nikon0 目录和文档，明确系统边界。

产物：

- `README.md`
- `docs/architecture.md`
- `docs/module_contracts.md`
- `docs/data_model.md`
- `docs/engineering_baseline.md`
- `docs/evaluation.md`

## Phase 1: 最小 Agent Runtime

目标：实现一轮 Agent 的完整生命周期，但只接一个 mock skill。

范围：

- `AgentRequest / AgentResponse`
- `AgentRuntime`
- `SkillRegistry`
- `ExecutionTrace`
- FastAPI `/chat`
- 单元测试和最小 eval case

成功标准：

```text
用户输入 -> AgentRuntime -> MockSkill -> Answer -> Trace
```

## Phase 2: 迁移当前客服能力为 Skill

目标：把旧 pipeline 能力挂到 nikon0，而不是重写业务。

迁移：

```text
rag_manual -> ProductSupportSkill
case_intake -> CaseIntakeSkill
memory v4 -> Session Issue Memory
MCP Gateway client -> ToolRuntime
```

成功标准：

```text
产品排障和工单收集能通过 nikon0 AgentRuntime 跑通。
```

## Phase 3: KnowledgeRuntime

目标：把手册知识从 RAG 框架里抽出来。

能力：

- structured manual query。
- playbook query。
- full-text query。
- RAG backend 兜底。
- attached-only 图片证据。

成功标准：

```text
高频手册问题优先命中结构化知识或 playbook，RAG 只作补充。
```

## Phase 4: ToolRuntime + MCP Gateway

目标：所有外部业务服务统一走 MCP Gateway。

第一批工具：

- create_case
- query_case_status
- query_order_status
- assess_refund_policy

成功标准：

```text
Agent 不直接依赖具体服务 SDK，只依赖 ToolRuntime。
```

## Phase 5: Safety / HITL

目标：控制企业服务风险。

拦截：

- 退款承诺。
- 投诉升级。
- 创建正式工单。
- 发送外部消息。
- 隐私暴露。

成功标准：

```text
高风险动作必须生成 approval request 或 handoff request。
```

## Phase 6: Evaluation Harness

目标：建立 Agent 行为评测，而不是只看答案。

评测集：

- skill selection。
- tool call。
- memory state。
- safety。
- end-to-end。

成功标准：

```text
每次重构都能用 eval 判断是否破坏 Agent 行为。
```

