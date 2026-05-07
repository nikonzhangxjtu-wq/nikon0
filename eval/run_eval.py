"""一键离线评测：跑 ChatPipeline，记录延迟，可选 LLM 打分，输出 CSV。

用法（在项目根目录）::

    python -m eval.run_eval --version v0.2-rerank

详见 eval/README.md。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# 支持 ``python eval/run_eval.py`` 与 ``python -m eval.run_eval`` 两种方式
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import settings
from app.services.pipeline import ChatPipeline

JUDGE_SYSTEM = """你是多模态客服赛题的自动评委。只根据「用户问题」与「助手回答」文本打分，不要编造题目或回答中不存在的事实。
若题目暗示应有图片而回答中未体现图文配合，可按「图文结合较差」在 2～3 分段酌情降分；若仅有文本材料则主要依据文本完整性、结构与深度评分。

评分区间 1～5，含义与正式赛题一致：
1 分，质量差：回答未回应问题，结构混乱或缺失，或图片相关表述明显无关、无帮助（若从文本可判断）。
2 分，质量一般：回答部分回应问题但不完整；结构较弱；条理与覆盖子问题不足。
3 分，质量中等：回答回应了问题但缺乏深度；结构基本清晰但可优化；覆盖面尚可。
4 分，质量良好：回答清晰、较为全面；结构逻辑清晰、组织合理；对多子问题有较好覆盖。
5 分，质量优秀：回答详细、有深度；结构严谨连贯；若能从回答看出与题意高度贴合、信息组织优秀，可给满分。

输出要求：最后一行必须且只能为如下格式（数字 1～5），不要其它文字在同一行：
SCORE:4
"""

SCORE_RE = re.compile(r"SCORE:\s*([1-5])", re.IGNORECASE)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数，p 取 0～1，例如 0.95。"""
    if not sorted_vals:
        return 0.0
    xs = sorted(sorted_vals)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(xs[lo] + frac * (xs[hi] - xs[lo]))


def _safe_filename_part(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s).strip("_") or "run"


def _relpath(path: Path) -> str:
    """尽量输出相对仓库根的路径，便于阅读。"""
    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e
    return rows


def rubric_pass(answer: str, keywords: list[str] | None) -> str:
    """弱规则：任一关键词出现在回答中则记 1，否则 0；无关键词则空字符串。"""
    if not keywords:
        return ""
    for kw in keywords:
        if not kw:
            continue
        if kw in answer:
            return "1"
    return "0"


def judge_score(question: str, answer: str, model: str) -> tuple[str | None, str]:
    """调用同一 Ollama 上的评分模型，返回 (分数 '1'～'5' 或 None, 原始文本)。"""
    from langchain_ollama import ChatOllama

    client = ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
    user = f"用户问题：\n{question}\n\n助手回答：\n{answer}\n"
    try:
        msg = client.invoke(
            [
                ("system", JUDGE_SYSTEM),
                ("human", user),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"[judge_error] {exc}"

    raw = getattr(msg, "content", "") or ""
    m = SCORE_RE.findall(raw)
    if not m:
        return None, raw.strip()
    return m[-1], raw.strip()


@dataclass
class RowResult:
    version: str
    run_id: str
    created_at: str
    sample_id: str
    category: str
    question: str
    answer: str
    route_reason: str
    route_needs_rag: str
    route_domain_hint: str
    route_confidence: float
    route_strategy: str
    route_low_confidence: str
    post_retrieval_gate: str
    latency_ms: float
    score: str | None
    rubric_pass: str
    judge_raw: str
    should_use_rag: str
    gold_manual_name: str
    gold_chunk_keywords: str
    answer_must_include: str
    retrieved_chunk_ids: str
    filtered_chunk_ids: str
    retrieved_manual_names: str
    filtered_manual_names: str
    top1_score: str
    context_chars: int
    context_chunk_count: int
    images_count: int
    pic_marker_count: int


def _json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _manual_hit(
    gold_manual_name: str,
    filtered_manual_names: list[str],
    retrieved_manual_names: list[str],
) -> str:
    if not gold_manual_name:
        return ""
    all_names = filtered_manual_names or retrieved_manual_names
    return "1" if gold_manual_name in all_names else "0"


def _must_include_pass(answer: str, items: list[str] | None) -> str:
    if not items:
        return ""
    return "1" if all(item and item in answer for item in items) else "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评测：pipeline + 延迟 + 可选 LLM 打分")
    parser.add_argument("--version", required=True, help="本次版本标签，写入 CSV 便于对比")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "eval" / "dataset" / "public_eval_30.jsonl",
        help="JSONL 评测集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "eval" / "results",
        help="输出目录",
    )
    parser.add_argument("--no-judge", action="store_true", help="关闭 LLM 打分，仅测延迟与 rubric")
    parser.add_argument("--judge-model", default="", help="评分模型名，默认使用 GEN_MODEL")
    parser.add_argument("--max-rows", type=int, default=0, help="只跑前 N 条，0 表示全量")
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到评测集: {dataset_path}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    judge_model = args.judge_model or settings.gen_model
    judge_enabled = not args.no_judge

    samples = load_jsonl(dataset_path)
    if args.max_rows and args.max_rows > 0:
        samples = samples[: args.max_rows]

    pipeline = ChatPipeline()
    row_results: list[RowResult] = []

    for sample in samples:
        sid = str(sample.get("id", ""))
        q = str(sample.get("question", "")).strip()
        if not q:
            continue
        cat = str(sample.get("category", ""))
        rub_kw = sample.get("rubric_keywords")
        keywords = rub_kw if isinstance(rub_kw, list) else None
        should_use_rag = sample.get("should_use_rag")
        gold_manual_name = str(sample.get("gold_manual_name", ""))
        gold_chunk_keywords = sample.get("gold_chunk_keywords")
        gold_chunk_keywords_list = gold_chunk_keywords if isinstance(gold_chunk_keywords, list) else []
        answer_must_include = sample.get("answer_must_include")
        answer_must_include_list = answer_must_include if isinstance(answer_must_include, list) else []

        t0 = time.perf_counter()
        pr = pipeline.run(question=q, images=[])
        dt_ms = (time.perf_counter() - t0) * 1000.0

        score_val: str | None = None
        judge_raw = ""
        if judge_enabled:
            score_val, judge_raw = judge_score(q, pr.answer, judge_model)

        rp = rubric_pass(pr.answer, keywords)

        row_results.append(
            RowResult(
                version=args.version,
                run_id=run_id,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                sample_id=sid,
                category=cat,
                question=q,
                answer=pr.answer,
                route_reason=pr.route_reason,
                route_needs_rag="1" if pr.debug.route_needs_rag else "0",
                route_domain_hint=pr.debug.route_domain_hint,
                route_confidence=pr.debug.route_confidence,
                route_strategy=pr.debug.route_strategy,
                route_low_confidence="1" if pr.debug.route_low_confidence else "0",
                post_retrieval_gate=pr.debug.post_retrieval_gate or "",
                latency_ms=dt_ms,
                score=score_val,
                rubric_pass=rp,
                judge_raw=judge_raw[:4000] if judge_raw else "",
                should_use_rag="" if should_use_rag is None else ("1" if bool(should_use_rag) else "0"),
                gold_manual_name=gold_manual_name,
                gold_chunk_keywords=_json_compact(gold_chunk_keywords_list),
                answer_must_include=_json_compact(answer_must_include_list),
                retrieved_chunk_ids=_json_compact(
                    pr.debug.retrieval.retrieved_chunk_ids if pr.debug.retrieval else []
                ),
                filtered_chunk_ids=_json_compact(
                    pr.debug.retrieval.filtered_chunk_ids if pr.debug.retrieval else []
                ),
                retrieved_manual_names=_json_compact(
                    pr.debug.retrieval.retrieved_manual_names if pr.debug.retrieval else []
                ),
                filtered_manual_names=_json_compact(
                    pr.debug.retrieval.filtered_manual_names if pr.debug.retrieval else []
                ),
                top1_score="" if not pr.debug.retrieval or pr.debug.retrieval.top1_score is None else f"{pr.debug.retrieval.top1_score:.6f}",
                context_chars=pr.debug.context_chars,
                context_chunk_count=pr.debug.context_chunk_count,
                images_count=len(pr.images),
                pic_marker_count=pr.answer.count("<PIC>"),
            )
        )

    # --- 明细 CSV ---
    ver_safe = _safe_filename_part(args.version)
    detail_path = output_dir / f"detail_{ver_safe}_{run_id}.csv"
    detail_fields = [
        "version",
        "run_id",
        "created_at",
        "id",
        "category",
        "latency_ms",
        "route_reason",
        "route_needs_rag",
        "route_domain_hint",
        "route_confidence",
        "route_strategy",
        "route_low_confidence",
        "post_retrieval_gate",
        "should_use_rag",
        "gold_manual_name",
        "gold_chunk_keywords",
        "answer_must_include",
        "retrieved_chunk_ids",
        "filtered_chunk_ids",
        "retrieved_manual_names",
        "filtered_manual_names",
        "top1_score",
        "context_chars",
        "context_chunk_count",
        "images_count",
        "pic_marker_count",
        "score",
        "rubric_pass",
        "judge_raw",
        "question",
        "answer",
    ]
    with detail_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=detail_fields)
        w.writeheader()
        for r in row_results:
            w.writerow(
                {
                    "version": r.version,
                    "run_id": r.run_id,
                    "created_at": r.created_at,
                    "id": r.sample_id,
                    "category": r.category,
                    "latency_ms": f"{r.latency_ms:.2f}",
                    "route_reason": r.route_reason,
                    "route_needs_rag": r.route_needs_rag,
                    "route_domain_hint": r.route_domain_hint,
                    "route_confidence": f"{r.route_confidence:.4f}",
                    "route_strategy": r.route_strategy,
                    "route_low_confidence": r.route_low_confidence,
                    "post_retrieval_gate": r.post_retrieval_gate,
                    "should_use_rag": r.should_use_rag,
                    "gold_manual_name": r.gold_manual_name,
                    "gold_chunk_keywords": r.gold_chunk_keywords,
                    "answer_must_include": r.answer_must_include,
                    "retrieved_chunk_ids": r.retrieved_chunk_ids,
                    "filtered_chunk_ids": r.filtered_chunk_ids,
                    "retrieved_manual_names": r.retrieved_manual_names,
                    "filtered_manual_names": r.filtered_manual_names,
                    "top1_score": r.top1_score,
                    "context_chars": r.context_chars,
                    "context_chunk_count": r.context_chunk_count,
                    "images_count": r.images_count,
                    "pic_marker_count": r.pic_marker_count,
                    "score": r.score or "",
                    "rubric_pass": r.rubric_pass,
                    "judge_raw": r.judge_raw,
                    "question": r.question,
                    "answer": r.answer,
                }
            )

    # --- 汇总指标 ---
    latencies = [r.latency_ms for r in row_results]
    scores_f: list[float] = []
    for r in row_results:
        if r.score and r.score.isdigit():
            scores_f.append(float(r.score))

    rubric_den = sum(1 for r in row_results if r.rubric_pass != "")
    rubric_hits = sum(1 for r in row_results if r.rubric_pass == "1")
    rubric_rate = (rubric_hits / rubric_den) if rubric_den else ""
    rag_gold_rows = [r for r in row_results if r.should_use_rag != ""]
    rag_acc_den = len(rag_gold_rows)
    rag_acc_hits = sum(1 for r in rag_gold_rows if r.route_needs_rag == r.should_use_rag)
    manual_gold_rows = [r for r in row_results if r.gold_manual_name]
    manual_hit_hits = 0
    must_include_den = 0
    must_include_hits = 0
    for r in row_results:
        filtered_manual_names = json.loads(r.filtered_manual_names) if r.filtered_manual_names else []
        retrieved_manual_names = json.loads(r.retrieved_manual_names) if r.retrieved_manual_names else []
        if r.gold_manual_name:
            manual_hit_hits += int(
                _manual_hit(r.gold_manual_name, filtered_manual_names, retrieved_manual_names) == "1"
            )
        answer_must_include_items = json.loads(r.answer_must_include) if r.answer_must_include else []
        must_include = _must_include_pass(r.answer, answer_must_include_items)
        if must_include != "":
            must_include_den += 1
            must_include_hits += int(must_include == "1")

    summary_row = {
        "version": args.version,
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_samples": len(row_results),
        "judge_enabled": "1" if judge_enabled else "0",
        "judge_model": judge_model if judge_enabled else "",
        "score_mean": f"{sum(scores_f) / len(scores_f):.4f}" if scores_f else "",
        "score_n": len(scores_f),
        "latency_p50_ms": f"{_percentile(latencies, 0.50):.2f}" if latencies else "",
        "latency_p95_ms": f"{_percentile(latencies, 0.95):.2f}" if latencies else "",
        "latency_mean_ms": f"{sum(latencies) / len(latencies):.2f}" if latencies else "",
        "rubric_pass_rate": f"{rubric_hits / rubric_den:.4f}" if rubric_den else "",
        "rubric_denominator": rubric_den,
        "rag_decision_acc": f"{rag_acc_hits / rag_acc_den:.4f}" if rag_acc_den else "",
        "rag_decision_denominator": rag_acc_den,
        "manual_hit_rate": f"{manual_hit_hits / len(manual_gold_rows):.4f}" if manual_gold_rows else "",
        "manual_hit_denominator": len(manual_gold_rows),
        "answer_must_include_rate": f"{must_include_hits / must_include_den:.4f}" if must_include_den else "",
        "answer_must_include_denominator": must_include_den,
        "avg_images_count": f"{sum(r.images_count for r in row_results) / len(row_results):.2f}" if row_results else "",
        "avg_pic_marker_count": f"{sum(r.pic_marker_count for r in row_results) / len(row_results):.2f}" if row_results else "",
        "dataset": _relpath(dataset_path),
        "detail_csv": _relpath(detail_path),
    }

    summary_path = output_dir / "summary.csv"
    summary_fields = list(summary_row.keys())
    write_header = not summary_path.is_file()
    with summary_path.open("a", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=summary_fields)
        if write_header:
            w.writeheader()
        w.writerow(summary_row)

    print(f"[eval] 明细: {detail_path}")
    print(f"[eval] 汇总已追加: {summary_path}")
    print(
        "[eval] 摘要:",
        f"n={summary_row['n_samples']}",
        f"score_mean={summary_row['score_mean'] or 'n/a'}",
        f"p95_ms={summary_row['latency_p95_ms'] or 'n/a'}",
        f"rubric_rate={summary_row['rubric_pass_rate'] or 'n/a'}",
    )


if __name__ == "__main__":
    main()
