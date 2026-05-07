# Multimodal Customer Service Agent (V1 Scaffold)

This repository is a **learning-first scaffold** for your competition project.
It is intentionally designed so that:

- The framework is ready to run.
- Core modules are clearly separated.
- Critical implementation points are marked with `TODO`.
- You can fill in the actual logic step by step.

## 1. V1 Goal

Build a baseline system that:

1. Exposes a competition-compatible `POST /chat` API.
2. Supports the required request fields (`question`, `images`, `session_id`, `stream`).
3. Implements a minimal pipeline:
   - RAG gate (whether retrieval is needed)
   - optional domain hint (manual vs customer service)
   - retrieval call
   - answer generation call
4. Can batch-generate a submission file from `question_public.csv`.

## 2. Suggested Project Structure

```text
app/
  main.py
  core/
    config.py
  schemas/
    chat.py
  services/
    generator.py
    ingestion.py
    pipeline.py
    retriever.py
    router.py
    session_store.py
  utils/
    prompt_builder.py
scripts/
  build_index.py
  run_batch_submission.py
eval/
  dataset/dev_eval.jsonl
  run_eval.py
  README.md
```

离线对比优化效果：见 [eval/README.md](eval/README.md)，一键命令示例：

```bash
python -m eval.run_eval --version v0.1-baseline
python -m eval.run_eval --version v0.2-rerank --no-judge
```

## 3. Tech Stack (Your Selected Direction)

- API: `fastapi`, `uvicorn`
- Validation: `pydantic`
- LLM: `langchain`, `langchain-ollama` (for `qwen2`)
- Indexing/RAG orchestration: `llama-index`
- Vector DB: `pymilvus` (Milvus)
- Optional workflow orchestration (later): `langgraph`

## 4. Python + Package Versions

Recommended:

- Python: `3.10` (conda env)

Version pins (stable for a V1 scaffold):

- `fastapi==0.115.0`
- `uvicorn[standard]==0.30.6`
- `pydantic==2.8.2`
- `python-dotenv==1.0.1`
- `langchain==0.2.16`
- `langchain-core==0.2.38`
- `langchain-text-splitters==0.2.2`
- `langchain-ollama==0.1.3`
- `llama-index==0.11.16`
- `llama-index-vector-stores-milvus==0.2.3`
- `pymilvus==2.4.6`
- `pandas==2.2.2`
- `orjson==3.10.7`

## 5. Conda Setup (No Auto Installation, Manual Commands)

```bash
conda create -n kafu_v1 python=3.10 -y
conda activate kafu_v1

pip install \
  "fastapi==0.115.0" \
  "uvicorn[standard]==0.30.6" \
  "pydantic==2.8.2" \
  "python-dotenv==1.0.1" \
  "langchain==0.2.16" \
  "langchain-core==0.2.38" \
  "langchain-text-splitters==0.2.2" \
  "langchain-ollama==0.1.3" \
  "llama-index==0.11.16" \
  "llama-index-vector-stores-milvus==0.2.3" \
  "pymilvus==2.4.6" \
  "pandas==2.2.2" \
  "orjson==3.10.7"
```

## 6. Ollama Model Check

You already have:

- `qwen2:latest` (generation)
- `nomic-embed-text:latest` (embedding)

Keep Ollama service running before API tests.

## 7. Environment Variables (Recommended)

Create `.env` manually:

```env
APP_HOST=0.0.0.0
APP_PORT=8000

API_BEARER_TOKEN=replace_with_your_token

OLLAMA_BASE_URL=http://localhost:11434
GEN_MODEL=qwen2:latest
EMBED_MODEL=nomic-embed-text:latest

MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DB_NAME=default
MILVUS_COLLECTION=manual_chunks_v1

MANUAL_DIR=./手册
```

## 8. Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 9. Build Index (after you implement TODOs)

```bash
python scripts/build_index.py
```

## 10. Batch Submission (after you implement TODOs)

```bash
python scripts/run_batch_submission.py
```

This script should generate an `id,ret` file aligned to:

- `question_public.csv`
- `submission_example.csv` schema

## 11. How To Use This Scaffold

Follow this learning sequence:

1. `app/services/ingestion.py`
   - Parse manual files.
   - Implement chunking and `<PIC>` -> image id binding.
2. `app/services/retriever.py`
   - Connect to Milvus.
   - Implement query embedding + top-k retrieval.
3. `app/services/generator.py`
   - Connect `qwen2` through LangChain.
   - Strictly answer from retrieved context.
4. `app/services/pipeline.py`
   - Compose gate -> retrieve -> generate.
5. `scripts/run_batch_submission.py`
   - Produce your first scoreable submission.

You should own the core logic yourself. This is exactly why the scaffold keeps key sections explicit and editable.

## 12. 路由与 RAG 门控升级指南（自行实现参考）

当前 [`app/services/pipeline.py`](app/services/pipeline.py) 与 [`app/services/router.py`](app/services/router.py) 属于「单次判定 → 直接执行」：`needs_rag=True` 就会检索并走 RAG prompt，**检索后没有有效上下文时也不会切换策略**；`RouteDecision.confidence` 主要进 debug，**未参与生成与兜底**。升级目标是把流程改成三段式：

**路由判定 → 检索验证 → 失败兜底**

### 设计原则（把三件事拆开）

1. **意图判断**：更像说明书还是客服，是否存在冲突（同一句话里两类词都强）。
2. **是否尝试检索**：值不值得去 Milvus 召回（可与「最终是否按 RAG 回答」分离）。
3. **检索结果是否可用**：过滤后是否有 chunk、分数是否可信；不可用则走兜底，而不是带着「（无检索上下文）」仍按强 RAG 回答。

### 分层推进（由浅入深）

**第一层：路由信号更细（先改 `router.py`）**

- 在 manual / customer_service / unknown 之外，增加 **冲突意图**（例如 manual 与 CS 关键词都命中且接近，不要与「完全无命中」混成同一种 unknown）。
- **收紧**「客服 → 关闭 RAG」：例如要求 `cs_hits` 达到最小次数，或相对 `manual_hits` 有足够 margin，避免一个词误判就完全不检索。
- 让 `confidence` / `reason` 能区分：无信号、弱信号、冲突、单边强信号（不必一上来就接模型分类）。

**第二层：检索后二次门控（优先改 `pipeline.py`）**

- `needs_rag=True` 只表示「尝试检索」；检索并 `retriever_context_filter` 之后：
  - **有有效 `filter_context`** → `need_rag=True`，走现有 RAG 流程。
  - **无有效上下文** → 切换到兜底分支（例如 `need_rag=False` 或单独 fallback key），要求澄清、不编造手册细节。
- 可选：结合 `RetrievalTrace` / top1 分数等做「弱召回」判定（注意 trace 里分数含义与过滤前后一致性）。

**第三层：Prompt 消费状态（`PromptContext` + `registry`）**

- 扩展 [`app/utils/prompts/context.py`](app/utils/prompts/context.py)：例如 `route_confidence`、`has_retrieval_context`、`fallback_reason`（或等价字段）。
- 在 [`app/utils/prompts/registry.py`](app/utils/prompts/registry.py) 区分：
  - 原生 no-rag（纯客服话术）
  - **尝试过 RAG 但无可用证据**（语气更保守、引导补全订单/型号/现象）
- 若暂不想新增 builder，可先复用 `no_rag_generic`，仅在 builder 内根据 `fallback_reason` 拼接说明。

**第四层：置信度的用法（建议渐进）**

- 先作 **软约束**：低置信度只在 prompt 中要求多澄清、少断言，不改变主分支，风险低。
- 再引入 **硬阈值**：例如低置信 + 弱检索时强制兜底（需配合评测调参）。

### 建议实现顺序（自检清单）

1. [`app/services/router.py`](app/services/router.py)：冲突类型、CS 关闭 RAG 的门槛、`confidence` 语义。
2. [`app/services/pipeline.py`](app/services/pipeline.py)：空/弱上下文 → fallback；必要时区分 `attempt_rag` 与最终 `need_rag`。
3. [`app/utils/prompts/context.py`](app/utils/prompts/context.py) + [`app/utils/prompts/registry.py`](app/utils/prompts/registry.py) + 各 `builders/*.py`：把状态传到 prompt。
4. [`eval/run_eval.py`](eval/run_eval.py)：统计 `route_needs_rag=1` 且 `context_chunk_count=0`、低置信仍强 RAG、CS 误判导致未检索等，便于对比版本。

### 最小可用升级（时间紧时只做这三项）

1. `router` 区分 **conflict**（或至少区分「双命中平局」与「全无命中」）。
2. `pipeline`：**过滤后无 chunk 则不走 `rag_manual` 强答**，改为兜底/澄清。
3. `PromptContext` 增加 **`fallback_reason`**（或 `has_retrieval_context`），让生成侧知道「不是不想检索，而是没检索到」。

### 相关文件速查

| 模块 | 路径 |
|------|------|
| 流水线 | [`app/services/pipeline.py`](app/services/pipeline.py) |
| 路由 | [`app/services/router.py`](app/services/router.py) |
| 检索与过滤 | [`app/services/retriever.py`](app/services/retriever.py) |
| Prompt 上下文 | [`app/utils/prompts/context.py`](app/utils/prompts/context.py) |
| Prompt 选择 | [`app/utils/prompts/registry.py`](app/utils/prompts/registry.py) |
| 离线评测 | [`eval/run_eval.py`](eval/run_eval.py) |
