"""Chunk 级检索质量数据集构建 & 评估。

与 test_retrieval_quality.py 的关键区别：
- 旧数据集：ground truth = query 来源手册的**全部 chunk**（手册级评估）
- 新数据集：ground truth = **具体 1-3 个 chunk_id**（chunk 级评估）

用法:
    python -m app.test.build_chunk_level_dataset --build-only   # 仅构建数据集
    python -m app.test.build_chunk_level_dataset --eval-only ./eval/dataset/chunk_level_queries.json  # 评估
    python -m app.test.build_chunk_level_dataset  # 构建 + 评估
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.ingestion import ManualIngestionService, ManualChunk


# ── 正则 ──────────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"(?m)^#+\s*(.+?)(?:\s*#+\s*)?$")
_IMG_TAG_RE = re.compile(r"<IMG:[^>]+>")
_LATEX_RE = re.compile(r"\$\\[a-z]+\{[^}]*\}\$")
_SPECIAL_CHAR_RE = re.compile(r"[\\$^{}]")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

_CN_ACTION_RE = re.compile(
    r"(如何|怎么|怎样)(安装|拆卸|清洁|清洗|更换|设置|操作|使用|启动|关闭|"
    r"充电|调节|维护|保养|排除|解决|处理|重置|配对|连接|校准|组装|拆除|取出|放入|选择|调整|按下|按住|转动|拨动|推动|拉动|拧紧|松开)"
)
_EN_ACTION_RE = re.compile(
    r"(how to|how do I|troubleshoot(?:ing)?)\s+([a-z]{3,}(?:\s+[a-z]{3,}){0,8})",
    re.IGNORECASE,
)
_CN_STEP_RE = re.compile(r"(?:步骤\s*\d+|第\s*\d+\s*步)[：:.\s]*(.+)")
_EN_STEP_RE = re.compile(r"(?:Step\s*\d+|STEP\s*\d+)[：:.\s]*(.+)", re.IGNORECASE)

_CN_FAULT_RE = re.compile(r"(?:错误(?:码|代码)?|故障码|报警)[：:.\s]*([A-Z]?\d{1,4})")
_EN_FAULT_RE = re.compile(r"(?:error\s*code|fault\s*code)[：:.\s]*([A-Z]?\d{1,4})", re.IGNORECASE)
_E_CODE_RE = re.compile(r"\b(E\d{1,4})\b")

_DEFINITION_CN_RE = re.compile(r"(?:什么是|什么叫|什么叫做)[一-鿿\w]{2,20}")
_DEFINITION_EN_RE = re.compile(r"(?:what is|what are|what does)\s+[a-z]{3,}(?:\s+[a-z]{2,}){0,5}", re.IGNORECASE)

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^安全", r"^警告", r"^注意[事项]?$", r"^重要", r"^前言",
        r"^产品简介", r"^目录", r"^使用说明[书]?$", r"^简介", r"^概述",
        r"^总则", r"^适用范围?$", r"^Introduction$", r"^Safety",
        r"^Warning", r"^Caution", r"^Important", r"^Table of Contents",
        r"^Overview", r"^General", r"^Specifications?$", r"^Product",
        r"^User Manual", r"^Congratulations", r"^感谢", r"^欢迎",
        r"^Before", r"^在使用", r"^使用前", r"^保养$", r"^维护$",
        r"^技术参数$", r"^技术规格$", r"^NOTICE$", r"^NOTE$",
        r"^注$", r"^提示$", r"^备注$", r"^Tips?$",
        r"^\d+\s+[A-Z]",  # "2 Your phone" 章节编号
        r"^\d+\.\d*\s",   # "3.1 Something" 编号
    ]
]

# TOC / 页码引用特征
_TOC_PAGE_RE = re.compile(r"\.\s*\d{1,3}\b")  # "Item Check List. 3"
_HEADING_PAGE_RE = re.compile(r"^\d+\s")      # 纯数字开头


@dataclass
class ChunkQuery:
    query_id: str
    query_text: str
    source_manual: str
    ground_truth_chunk_ids: list[str]
    query_type: str
    language: str


@dataclass
class ChunkLevelDataset:
    queries: list[ChunkQuery]
    all_chunks: dict[str, ManualChunk]
    all_manuals: list[str]
    chunks_by_manual: dict[str, list[str]]


# ── 工具函数 ──────────────────────────────────────────────────────────


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _clean_text(text: str) -> str:
    """清除 <IMG:..>、LaTeX、特殊字符、Markdown # 标记。"""
    text = _IMG_TAG_RE.sub("", text)
    text = _LATEX_RE.sub("", text)
    text = _SPECIAL_CHAR_RE.sub("", text)
    text = text.replace("#", " ").replace("~", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _is_boilerplate(text: str) -> bool:
    t = text.strip()
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(t):
            return True
    if len(t) < 4:
        return True
    # 纯数字+符号开头（章节编号）
    if re.match(r"^\d+[\s.)]", t):
        return True
    # TOC / 页码引用特征："xxx. 3" 或 "xxx.. .74"
    if re.search(r"\.\s*\d{1,3}\b", t) and len(t.split()) <= 8:
        return True
    # 带 "reference page numbers" 的是 TOC
    if re.search(r"reference page numbers|page numbers are provided", t, re.IGNORECASE):
        return True
    # 纯英文 boilerplate 短句
    if re.match(r"^(Always|Never|Do not|Make sure|Ensure|Check)\b", t, re.IGNORECASE):
        if len(t) < 30:
            return True
    # troubleshooting/warranty 混合 — 实际是 warranty 不是 troubleshooting
    if t.lower().startswith("troubleshooting") and "warranty" in t.lower():
        return True
    return False


def _is_valid_query(text: str) -> bool:
    if not text or "<IMG:" in text or "<PIC>" in text:
        return False
    t = text.strip()
    if len(t) < 4 or len(t) > 100:
        return False
    alpha = sum(1 for c in t if c.isalpha() or '一' <= c <= '鿿')
    if alpha / max(len(t), 1) < 0.3:
        return False
    if re.match(r"^[\d\s.,;:()\-+]+$", t):
        return False
    # 乱码：以"至"开头或仅数字+标点
    if re.match(r"^至\d|^\d+[。，、]\s*\d", t):
        return False
    # 占位符文本
    if re.search(r"在此处粘贴|在此粘贴|placeholder|TODO|FIXME", t, re.IGNORECASE):
        return False
    return True


def _extract_heading(text: str) -> str | None:
    for m in _HEADING_RE.finditer(text):
        heading = _clean_text(m.group(1).strip())
        if not _is_boilerplate(heading) and _is_valid_query(heading):
            return heading
    return None


def _extract_action_query(text: str) -> str | None:
    # 中文
    m = _CN_ACTION_RE.search(text)
    if m:
        start, end = m.start(), min(len(text), m.end() + 30)
        phrase = text[start:end]
        phrase = re.split(r"[，。；！？\n,.;!?]", phrase)[0]
        phrase = _clean_text(phrase)
        if _is_valid_query(phrase) and not _is_boilerplate(phrase):
            return phrase
    # 英文
    m = _EN_ACTION_RE.search(text)
    if m:
        start, end = m.start(), min(len(text), m.end() + 40)
        phrase = text[start:end]
        phrase = re.split(r"[.,;!?\n]", phrase)[0]
        phrase = _clean_text(phrase)
        if _is_valid_query(phrase) and not _is_boilerplate(phrase):
            return phrase
    return None


def _extract_fault_query(text: str) -> str | None:
    for pat, prefix in [
        (_CN_FAULT_RE, "故障码 "), (_E_CODE_RE, "故障码 "), (_EN_FAULT_RE, "Error code "),
    ]:
        m = pat.search(text)
        if m:
            code = m.group(1).strip()
            if len(code) >= 2:
                return f"{prefix}{code}"
    return None


def _extract_step_query(text: str) -> str | None:
    clean = _clean_text(text)
    for regex in [_CN_STEP_RE, _EN_STEP_RE]:
        for m in regex.finditer(clean):
            step_text = m.group(1).strip()
            if _is_valid_query(step_text) and not _is_boilerplate(step_text):
                action = _CN_ACTION_RE.search(text)
                prefix = f"如何{action.group(2)}" if action else ""
                return f"{prefix}{step_text[:40]}" if prefix else step_text[:60]
    return None


def _extract_definition_query(text: str) -> str | None:
    # 中文 "什么是X"
    m = _DEFINITION_CN_RE.search(text)
    if m:
        phrase = _clean_text(m.group(0))
        if _is_valid_query(phrase) and not _is_boilerplate(phrase):
            return phrase
    # 英文 "What is X"
    m = _DEFINITION_EN_RE.search(text)
    if m:
        phrase = _clean_text(m.group(0))
        if _is_valid_query(phrase) and not _is_boilerplate(phrase):
            return phrase
    return None


# ── 数据集构建 ────────────────────────────────────────────────────────


def build_chunk_level_dataset(
    chunks: list[ManualChunk], *, max_per_manual: int = 3,
) -> ChunkLevelDataset:
    chunks_by_manual: dict[str, list[ManualChunk]] = defaultdict(list)
    chunk_by_id: dict[str, ManualChunk] = {}
    for c in chunks:
        chunks_by_manual[c.manual_name].append(c)
        chunk_by_id[c.chunk_id] = c

    all_manuals = sorted(chunks_by_manual.keys())
    chunk_ids_by_manual = {
        m: [c.chunk_id for c in clist] for m, clist in chunks_by_manual.items()
    }

    extractors = [
        ("heading", _extract_heading),
        ("action", _extract_action_query),
        ("fault", _extract_fault_query),
        ("action", _extract_step_query),
        ("definition", _extract_definition_query),
    ]

    queries: list[ChunkQuery] = []
    qid = 0
    seen_queries: set[str] = set()

    # 1. 构建所有候选 queries
    for manual_name in all_manuals:
        manual_chunks = chunks_by_manual[manual_name]
        manual_queries: list[ChunkQuery] = []
        used_chunks: set[str] = set()

        for qtype, extractor in extractors:
            for c in manual_chunks:
                if len(manual_queries) >= max_per_manual:
                    break
                if c.chunk_id in used_chunks:
                    continue
                result = extractor(c.text)
                if not result or result in seen_queries:
                    continue
                seen_queries.add(result)
                used_chunks.add(c.chunk_id)
                manual_queries.append(ChunkQuery(
                    query_id=f"cq{qid:04d}", query_text=result,
                    source_manual=manual_name,
                    ground_truth_chunk_ids=[c.chunk_id],
                    query_type=qtype,
                    language="zh" if _is_chinese(result) else "en",
                ))
                qid += 1

        queries.extend(manual_queries)

    # 2. 唯一性过滤：query 不应作为标题出现在太多其他 chunk 中
    #    构建所有 chunk 的 heading -> chunk_count 索引
    heading_count: dict[str, int] = defaultdict(int)
    for c in chunks:
        h = _extract_heading(c.text)
        if h:
            heading_count[h.lower()] += 1

    filtered: list[ChunkQuery] = []
    for q in queries:
        count = heading_count.get(q.query_text.lower(), 1)
        if count <= 2:  # 最多在 2 个 chunk 中作为标题出现
            filtered.append(q)
        else:
            # 这个 query 太通用，但可以尝试从同一 chunk 提取其他内容
            pass

    # 3. 确保每本手册至少有 1 条
    covered_manuals: set[str] = {q.source_manual for q in filtered}
    for manual_name in all_manuals:
        if manual_name in covered_manuals:
            continue
        # 从该手册的第一条非 boilerplate heading 中补一条
        for c in chunks_by_manual[manual_name]:
            heading = _extract_heading(c.text)
            if heading and heading not in seen_queries:
                if heading_count.get(heading.lower(), 1) <= 3:
                    seen_queries.add(heading)
                    filtered.append(ChunkQuery(
                        query_id=f"cq{qid:04d}", query_text=heading,
                        source_manual=manual_name,
                        ground_truth_chunk_ids=[c.chunk_id],
                        query_type="heading",
                        language="zh" if _is_chinese(heading) else "en",
                    ))
                    qid += 1
                    covered_manuals.add(manual_name)
                    break

    # 重新分配 query_id
    for i, q in enumerate(filtered):
        q.query_id = f"cq{i:04d}"

    return ChunkLevelDataset(
        queries=filtered, all_chunks=chunk_by_id,
        all_manuals=all_manuals, chunks_by_manual=chunk_ids_by_manual,
    )


# ── 验证 ──────────────────────────────────────────────────────────────


def validate_dataset(dataset: ChunkLevelDataset) -> dict:
    report: dict = {
        "total": len(dataset.queries), "warnings": [], "keyword_overlap_stats": {},
    }
    overlaps: list[float] = []
    for q in dataset.queries:
        target = dataset.all_chunks.get(q.ground_truth_chunk_ids[0])
        if not target:
            report["warnings"].append(f"{q.query_id}: 目标 chunk 不存在")
            continue
        q_words = set(re.findall(r"[一-鿿]+|[a-zA-Z]{2,}", q.query_text.lower()))
        c_words = set(re.findall(r"[一-鿿]+|[a-zA-Z]{2,}", target.text.lower()))
        if q_words:
            overlap = len(q_words & c_words) / len(q_words)
            overlaps.append(overlap)
            if overlap < 0.2:
                report["warnings"].append(
                    f"{q.query_id}: 低重叠 {overlap:.0%} q='{q.query_text[:50]}'"
                )

    report["keyword_overlap_stats"] = {
        "mean": sum(overlaps) / len(overlaps) if overlaps else 0,
        "min": min(overlaps) if overlaps else 0,
        "max": max(overlaps) if overlaps else 0,
        "low_overlap_count": sum(1 for o in overlaps if o < 0.2),
    }

    by_manual: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    for q in dataset.queries:
        by_manual[q.source_manual] += 1
        by_type[q.query_type] += 1
    report["by_manual"] = dict(by_manual)
    report["by_type"] = dict(by_type)
    return report


# ── 评估 ──────────────────────────────────────────────────────────────


def evaluate_chunk_level(dataset: ChunkLevelDataset, top_k: int = 5) -> dict:
    from app.services.retriever import VectorRetriever

    retriever = VectorRetriever()
    total_time_ms = 0.0
    chunk_h1 = chunk_h3 = chunk_h5 = 0
    manual_h1 = manual_h3 = 0
    mrr_sum = 0.0
    n = len(dataset.queries)
    per_query_results: list[dict] = []

    print(f"\n评估 {n} 条 chunk 级 query（每条约 3s）...\n")

    for i, q in enumerate(dataset.queries):
        t0 = time.perf_counter()
        try:
            retrieved = retriever.retrieve(q.query_text, top_k=top_k)
        except Exception as e:
            print(f"  [{q.query_id}] error: {e}")
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        total_time_ms += elapsed

        gt_ids = set(q.ground_truth_chunk_ids)
        h1 = retrieved[0].chunk_id in gt_ids if retrieved else False
        h3 = any(c.chunk_id in gt_ids for c in retrieved[:3])
        h5 = any(c.chunk_id in gt_ids for c in retrieved[:5])
        m1 = retrieved[0].manual_name == q.source_manual if retrieved else False
        m3 = any(c.manual_name == q.source_manual for c in retrieved[:3])

        chunk_h1 += int(h1)
        chunk_h3 += int(h3)
        chunk_h5 += int(h5)
        manual_h1 += int(m1)
        manual_h3 += int(m3)

        chunk_mrr = 0.0
        for rank, c in enumerate(retrieved, 1):
            if c.chunk_id in gt_ids:
                chunk_mrr = 1.0 / rank
                mrr_sum += chunk_mrr
                break

        per_query_results.append({
            "query_id": q.query_id,
            "query_text": q.query_text[:80],
            "source_manual": q.source_manual,
            "query_type": q.query_type,
            "gt_chunk_id": q.ground_truth_chunk_ids[0],
            "chunk_hit": h1,
            "chunk_hit_at_3": h3,
            "chunk_hit_at_5": h5,
            "manual_hit": m1,
            "chunk_mrr": chunk_mrr,
            "retrieved_top5_chunks": [c.chunk_id for c in retrieved[:5]],
            "retrieved_top5_manuals": [c.manual_name for c in retrieved[:5]],
            "retrieved_top5_scores": [round(c.score, 4) for c in retrieved[:5]],
            "latency_ms": round(elapsed, 1),
        })

        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            avg_lat = total_time_ms / (i + 1)
            eta = avg_lat * (n - i - 1) / 1000
            print(
                f"\r[{i+1}/{n}] chunk-hit@1={chunk_h1/(i+1):.1%} "
                f"mrr={mrr_sum/(i+1):.4f}  avg={avg_lat:.0f}ms  eta={eta:.0f}s",
                end="", flush=True,
            )

    print()
    return {
        "dataset": {
            "total_queries": n, "total_manuals": len(dataset.all_manuals),
            "manuals": dataset.all_manuals,
        },
        "chunk_level": {
            "chunk_hit_at_1": chunk_h1 / n if n else 0,
            "chunk_hit_at_3": chunk_h3 / n if n else 0,
            "chunk_hit_at_5": chunk_h5 / n if n else 0,
            "mrr": mrr_sum / n if n else 0,
        },
        "manual_level": {
            "manual_hit_at_1": manual_h1 / n if n else 0,
            "manual_hit_at_3": manual_h3 / n if n else 0,
        },
        "latency": {"total_mean_ms": total_time_ms / n if n else 0},
        "per_query": per_query_results,
    }


# ── 导出 ──────────────────────────────────────────────────────────────


def export_dataset(dataset: ChunkLevelDataset, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "description": "Chunk 级检索评估数据集 — ground truth 为具体 chunk_id",
        "total_queries": len(dataset.queries),
        "total_manuals": len(dataset.all_manuals),
        "queries": [
            {
                "query_id": q.query_id, "query_text": q.query_text,
                "source_manual": q.source_manual,
                "ground_truth_chunk_ids": q.ground_truth_chunk_ids,
                "query_type": q.query_type, "language": q.language,
            }
            for q in dataset.queries
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据集已导出: {path}")


def print_summary(dataset: ChunkLevelDataset, v: dict) -> None:
    print("=" * 65)
    print("Chunk 级评估数据集概览")
    print("=" * 65)
    print(f"  总 query: {len(dataset.queries)}  手册: {len(dataset.all_manuals)}  chunk: {len(dataset.all_chunks)}")
    zh = sum(1 for q in dataset.queries if q.language == "zh")
    print(f"  语言: zh={zh}  en={len(dataset.queries) - zh}")

    print("  按类型:", "  ".join(f"{t}={c}" for t, c in sorted(v.get("by_type", {}).items())))

    kos = v.get("keyword_overlap_stats", {})
    print(f"  关键词重叠: mean={kos.get('mean',0):.0%}  min={kos.get('min',0):.0%}  low={kos.get('low_overlap_count',0)}")

    warns = v.get("warnings", [])
    if warns:
        print(f"  ⚠ {len(warns)} 条警告:")
        for w in warns[:8]:
            print(f"    - {w}")
        if len(warns) > 8:
            print(f"    ... 还有 {len(warns) - 8} 条")
    print("=" * 65)


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Chunk 级检索评估数据集构建 & 评估")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--eval-only", type=str, metavar="PATH")
    p.add_argument("--max-per-manual", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--export", type=str, default="./eval/dataset/chunk_level_queries.json")
    p.add_argument("--result", type=str, default=None)
    args = p.parse_args()

    if args.eval_only:
        data = json.loads(Path(args.eval_only).read_text(encoding="utf-8"))
        print(f"加载数据集: {args.eval_only} ({len(data['queries'])} 条)")
        ingestion = ManualIngestionService()
        all_chunks = ingestion.parse_and_chunk()
        chunk_by_id = {c.chunk_id: c for c in all_chunks}
        queries = [ChunkQuery(
            query_id=d["query_id"], query_text=d["query_text"],
            source_manual=d["source_manual"],
            ground_truth_chunk_ids=d["ground_truth_chunk_ids"],
            query_type=d.get("query_type", "general"),
            language=d.get("language", "zh"),
        ) for d in data["queries"]]
        cm = defaultdict(list)
        for c in all_chunks:
            cm[c.manual_name].append(c.chunk_id)
        ds = ChunkLevelDataset(queries=queries, all_chunks=chunk_by_id,
                               all_manuals=sorted(cm), chunks_by_manual=dict(cm))
        result = evaluate_chunk_level(ds, top_k=args.top_k)
        rp = args.result or f"./eval/results/chunk_level_{time.strftime('%Y%m%dT%H%M%SZ')}.json"
        Path(rp).parent.mkdir(parents=True, exist_ok=True)
        Path(rp).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        cl, ml = result["chunk_level"], result["manual_level"]
        print(f"\n结果: {rp}")
        print(f"  Chunk Hit@1/3/5: {cl['chunk_hit_at_1']:.1%} / {cl['chunk_hit_at_3']:.1%} / {cl['chunk_hit_at_5']:.1%}")
        print(f"  Chunk MRR:       {cl['mrr']:.4f}")
        print(f"  Manual Top-1/3:  {ml['manual_hit_at_1']:.1%} / {ml['manual_hit_at_3']:.1%}")
        print(f"  平均延迟:          {result['latency']['total_mean_ms']:.0f}ms")
        print(f"  Manual Top-1 - Chunk Hit@1 = {ml['manual_hit_at_1'] - cl['chunk_hit_at_1']:.1%}")
        print(f"  → 同手册内 chunk 排序失误率")
        return

    print("加载 chunk...")
    all_chunks = ManualIngestionService().parse_and_chunk()
    print(f"  {len(all_chunks)} 个 chunk")

    print("构建数据集...")
    ds = build_chunk_level_dataset(all_chunks, max_per_manual=args.max_per_manual)
    v = validate_dataset(ds)
    print_summary(ds, v)
    export_dataset(ds, args.export)

    print("\n--- 前 12 条样本 ---")
    for q in ds.queries[:12]:
        t = ds.all_chunks.get(q.ground_truth_chunk_ids[0])
        preview = t.text[:120].replace("\n", " ") if t else "(missing)"
        print(f"  [{q.query_id}] {q.query_type:10} | {q.source_manual:25} | {q.query_text[:60]}")
        print(f"         GT={q.ground_truth_chunk_ids[0]}  chunk: {preview}...")

    if args.build_only:
        return

    print("\n开始评估...")
    result = evaluate_chunk_level(ds, top_k=args.top_k)
    rp = args.result or f"./eval/results/chunk_level_{time.strftime('%Y%m%dT%H%M%SZ')}.json"
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    Path(rp).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cl, ml = result["chunk_level"], result["manual_level"]
    print(f"\n结果: {rp}")
    print(f"  Chunk Hit@1/3/5: {cl['chunk_hit_at_1']:.1%} / {cl['chunk_hit_at_3']:.1%} / {cl['chunk_hit_at_5']:.1%}")
    print(f"  Chunk MRR:       {cl['mrr']:.4f}")
    print(f"  Manual Top-1/3:  {ml['manual_hit_at_1']:.1%} / {ml['manual_hit_at_3']:.1%}")
    print(f"  平均延迟:          {result['latency']['total_mean_ms']:.0f}ms")
    print(f"  Manual Top-1 - Chunk Hit@1 = {ml['manual_hit_at_1'] - cl['chunk_hit_at_1']:.1%}")


if __name__ == "__main__":
    main()
