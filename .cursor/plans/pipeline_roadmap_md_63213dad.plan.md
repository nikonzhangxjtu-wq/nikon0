---
name: Pipeline roadmap MD
overview: 在项目里新增一份循序渐进的 Markdown 路线图，帮助你从「看懂 ChatPipeline」到「能独立跑通、调通、扩展」整条 RAG 链路。
todos:
  - id: add-docs-dir
    content: 创建 docs/（若不存在）
    status: pending
  - id: write-pipeline-roadmap
    content: 撰写 docs/pipeline-roadmap.md（分阶段 + mermaid + 命令清单）
    status: pending
  - id: optional-readme-link
    content: （可选）在 README 增加指向 pipeline-roadmap 的链接
    status: pending
isProject: false
---

# Pipeline 循序渐进路线图（Markdown 文档）

## 目标

新增一份可执行的阅读/动手顺序说明，覆盖 [app/services/pipeline.py](app/services/pipeline.py) 涉及的：**路由 → 检索 → 拼 prompt → 生成**，以及与索引、配置、接口的衔接。

## 产出物

- 新建文档：[docs/pipeline-roadmap.md](docs/pipeline-roadmap.md)（若仓库尚无 `docs/`，一并创建该目录）
- 文档结构建议（目录级大纲）：
  1. **总览**：用一张 mermaid 流程图概括 `ChatPipeline.run`（与当前代码一致：`router.decide` → `needs_rag` 分支 → `retrieve` / `build_`* / `generate` 或 fallback）
  2. **阶段 0 — 环境与入口**：`.env` / [app/core/config.py](app/core/config.py) 中与流水线相关的项（`GEN_MODEL`、`EMBED_MODEL`、`OLLAMA_BASE_URL`、`MILVUS_`*、`MANUAL_DIR`）；如何用 [app/main.py](app/main.py) 的 `POST /chat` 触发整条链（Bearer、`stream` 限制）
  3. **阶段 1 — 单步读懂依赖**：按阅读顺序列出文件与关注点
    - [app/services/router.py](app/services/router.py)：`RouteDecision`、`needs_rag` 与 `domain_hint` 的含义  
    - [app/services/retriever.py](app/services/retriever.py)：Milvus `search` 返回结构、`image_ids` JSON 与 `RetrievedChunk`  
    - [app/utils/prompt_builder.py](app/utils/prompt_builder.py)：`build_context_block` / `build_generation_prompt` 的输入输出  
    - [app/services/generator.py](app/services/generator.py)：`ChatOllama` 与 `settings.gen_model`
  4. **阶段 2 — 索引与检索契约**：对照 [scripts/build_index.py](scripts/build_index.py) 写入字段与 retriever 读取字段是否一致；建议执行顺序：`build_index` → 再测 `VectorRetriever`
  5. **阶段 3 — 自动化验证**：运行已有测试（不写新测也可在文档中列出命令）
    - `python -m app.test.test_retriever_build_index_integration`（Fake Milvus，验契约）  
    - 可选：`python -m app.test.test_build_index`（若环境依赖满足）
  6. **阶段 4 — 真机联调清单**：Ollama 模型、`scripts/build_index.py`、Milvus 集合存在、`POST /chat` 走 RAG 的示例问题（含说明书类关键词以命中 [router.py](app/services/router.py) 的 RAG 分支）
  7. **阶段 5 — 下一步扩展**（与 [pipeline.py](app/services/pipeline.py) 顶部 TODO 对齐）：检索阈值、客服政策库 RAG、`images` 接入、流式响应

## 写作约束

- 全文中文，步骤可勾选（checkbox 语法），每条附带「预期结果」便于自检
- 不修改业务代码；仅新增/更新上述 Markdown（除非你后续要求把链接写进 README）

## 实施步骤（待你确认后执行）

1. 创建 `docs/`（如不存在）
2. 写入 `docs/pipeline-roadmap.md`，包含上述章节与 mermaid 图
3. （可选）在 [README.md](README.md) 末尾加一行指向该文档的链接——**默认不做**，除非你希望文档可从 README 发现

