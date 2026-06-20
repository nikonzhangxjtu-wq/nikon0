# nikon0

nikon0 是一个面向企业产品服务场景的自研 Agent 框架。它从当前多模态智能客服项目演进而来，但不再把 RAG 当成主架构，而是把 RAG、手册、图片理解、工单、订单、退款、投诉、人工介入都视为 Agent Runtime 可调度的能力。

## 定位

nikon0 的第一业务场景是 **企业产品服务 Agent**：

```text
用户提出产品咨询、故障排查、售后报修、订单/工单查询、退款/投诉等请求；
Agent 读取当前会话问题状态，选择合适 skill，调用企业工具或知识能力，
在安全边界内完成答复、工单流转和人工介入。
```

## 设计原则

- 不依赖 LangChain / LangGraph 作为核心框架，Agent Runtime 自研。
- RAG 只是 KnowledgeRuntime 的一个 backend，不是系统中心。
- Skill 可以是 MD 指令、代码能力，或二者结合。
- MCP Gateway 是外部服务统一入口，Agent 不直接绑定具体业务系统。
- 记忆只做当前会话内的 Issue State，不做陪伴式长期用户画像。
- 高风险动作必须经过 Safety / Human-in-the-loop。
- 所有关键行为都要可观测、可追溯、可评测。

## 第一版目录

```text
nikon0/
  README.md
  docs/
    architecture.md
    module_contracts.md
    data_model.md
    engineering_baseline.md
    evaluation.md
    roadmap.md
  agent/
  skills/
  tools/
  knowledge/
  memory/
  safety/
  eval/
  app/
  infra/
```

这些目录目前先作为架构边界保留。下一轮开发时，建议先实现 `agent/`、`skills/`、`tools/` 的最小闭环，再迁移当前客服能力。

## 推荐阅读顺序

1. [Architecture](docs/architecture.md)
2. [Module Contracts](docs/module_contracts.md)
3. [Data Model](docs/data_model.md)
4. [Engineering Baseline](docs/engineering_baseline.md)
5. [Evaluation](docs/evaluation.md)
6. [Roadmap](docs/roadmap.md)

