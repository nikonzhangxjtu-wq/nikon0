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
