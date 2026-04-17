# 离线评测（小集 + 一键跑分）

## 目录说明

| 路径 | 说明 |
|------|------|
| `eval/dataset/dev_eval.jsonl` | 小评测集（每行一个 JSON 对象） |
| `eval/dataset/public_eval_30.jsonl` | 从 `question_public.csv` 挑选的 30 条覆盖性样本 |
| `eval/run_eval.py` | 一键跑分：跑 pipeline、记延迟、可选 LLM 打分、写 CSV |
| `eval/results/` | 输出目录（默认已加入 `.gitignore`） |

## 评测集 JSONL 字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 是 | 样本唯一 id |
| `question` | 是 | 用户问题 |
| `category` | 否 | 如 `manual` / `customer_service`，仅便于分析 |
| `rubric_keywords` | 否 | 关键词列表；若回答中出现任一关键词则 `rubric_pass=1`（弱指标，可自改规则） |
| `should_use_rag` | 否 | 金标路由；用于比较是否应走检索 |
| `gold_manual_name` | 否 | 期望命中的手册名，用于粗粒度检索命中率 |
| `gold_chunk_keywords` | 否 | 期望命中 chunk 的关键词提示，当前主要用于人工分析 |
| `answer_must_include` | 否 | 回答应覆盖的要点列表，全部出现才记通过 |

可自行复制该文件为 `my_eval.jsonl` 扩充题量。

## 运行方式

在项目根目录执行（保证能 `import app`）：

```bash
python -m eval.run_eval --version v0.1-baseline
```

运行 30 条公开问题样本集：

```bash
python -m eval.run_eval --version v0.1-public30 --dataset eval/dataset/public_eval_30.jsonl
```

常用参数：

```text
--version       本次优化标签，会写入 CSV 每一行，便于对比
--dataset       默认 eval/dataset/dev_eval.jsonl
--output-dir    默认 eval/results
--no-judge      不调评分模型，只测延迟与 rubric（适合 CI 或断网）
--judge-model   覆盖默认的评分用模型（默认读取环境变量 GEN_MODEL）
--max-rows      只跑前 N 条，快速试跑
```

## 输出文件

1. **`detail_<run_id>.csv`**  
   逐条样本：除 `version`、`latency_ms`、`score`、`rubric_pass`、`route_reason`、`question`、`answer` 外，还会记录 `route_needs_rag`、`route_confidence`、`retrieved_chunk_ids`、`filtered_chunk_ids`、`top1_score`、`context_chars`、`context_chunk_count` 等 trace 字段。

2. **`summary.csv`**（追加写入）  
   每次运行一行汇总：`score_mean`、`latency_p50_ms`、`latency_p95_ms`、`rubric_pass_rate`、`rag_decision_acc`、`manual_hit_rate`、`answer_must_include_rate` 等，用来画「版本 vs 指标」曲线。

## 指标怎么读

- **主看**：`score_mean`（开启 judge 时）、`rag_decision_acc`、`manual_hit_rate`、`answer_must_include_rate`。  
- **辅助排查**：看明细里的 `retrieved_chunk_ids`、`filtered_chunk_ids`、`top1_score`、`context_chars`，区分问题出在路由、召回还是 prompt。  
- **辅看**：`latency_p95_ms`，避免优化检索后延迟失控。  
- **平台分**：仍建议阶段性提交验证，与离线集趋势对照。

## 评分模型（LLM-as-judge）

默认用与 `GEN_MODEL` 相同的 Ollama 模型、低温度，要求输出末行 `SCORE:3`（1～5）。  
若解析失败，该行 `score` 为空，可检查 judge 原始输出列。
