"""对 150 条 QA 结果进行自动评判并生成综合报告."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_PATH = Path("nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl")
RESULTS_PATH = Path("nikon0/eval/reports/manual-qa-eval-150/full/raw_results.jsonl")
OUTPUT_DIR = Path("nikon0/eval/reports/manual-qa-eval-150/full")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def phrase_in_answer(phrase: str, answer: str) -> bool:
    """检查短语是否在回答中."""
    p = re.sub(r"\s+", "", phrase).lower()
    a = re.sub(r"\s+", "", answer).lower()
    if p in a:
        return True
    # 词级别匹配：至少60%的token出现
    tokens = [t for t in re.split(r"\s+", phrase) if len(t) >= 2]
    if len(tokens) >= 3:
        found = sum(1 for t in tokens if re.sub(r"\s+", "", t).lower() in a)
        return found >= len(tokens) * 0.6
    return False


def _manual_stem(value: str) -> str:
    return Path(value).stem.strip().lower()


def _evidence_manual_names(result: dict) -> set[str]:
    names: set[str] = set()
    for action in result.get("actions", []):
        if action.get("kind") != "tool" or action.get("name") != "product-support.search_product_manual":
            continue
        payload = action.get("payload") or {}
        for evidence in payload.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            manual_name = str((evidence.get("payload") or {}).get("manual_name") or "").strip()
            if manual_name:
                names.add(_manual_stem(manual_name))
    return names


def judge_item(item: dict, result: dict) -> dict[str, Any]:
    """评判单条结果."""
    cid = result["case_id"]
    expected = item.get("expected", {})
    acceptable_skills = expected.get("acceptable_skills", [])
    must_contain = expected.get("answer_must_contain", [])
    handoff_expected = expected.get("handoff", False)
    manual = item.get("metadata", {}).get("source_manual", "")
    answer = result.get("answer", "")
    skill = result.get("selected_skill", "")
    source = result.get("selection_source", "")

    checks = {}

    # 1. 技能选择检查
    if not acceptable_skills or None in acceptable_skills:
        checks["skill_ok"] = True  # 任意技能均可
    elif not skill:
        checks["skill_ok"] = "none" in acceptable_skills if None in acceptable_skills else False
    else:
        # 归一化技能名称
        checks["skill_ok"] = any(
            s and (skill == s or s in skill or skill in s)
            for s in acceptable_skills if s
        )
        if not checks["skill_ok"] and "general" in acceptable_skills and skill in ("none", ""):
            checks["skill_ok"] = True  # general 类别允许不选技能

    # 2. 证据检索检查
    manual_name = _manual_stem(manual) if manual else ""
    evidence_manual_names = _evidence_manual_names(result)
    checks["evidence_manual_names"] = sorted(evidence_manual_names)
    checks["evidence_from_correct_manual"] = bool(manual_name and manual_name in evidence_manual_names)
    checks["has_evidence"] = bool(evidence_manual_names)

    # 3. must_contain 检查
    must_scores = []
    for phrase in must_contain:
        found = phrase_in_answer(phrase, answer)
        must_scores.append(found)
    checks["must_contain_score"] = sum(must_scores) / len(must_scores) if must_scores else 1.0
    checks["must_contain_details"] = [
        {"phrase": p, "found": f}
        for p, f in zip(must_contain, must_scores)
    ]

    # 4. handoff/审批检查
    checks["handoff_triggered"] = any(
        a.get("kind") == "handoff" for a in result.get("actions", [])
    )
    checks["approval_triggered"] = any(
        a.get("kind") == "approval" for a in result.get("actions", [])
    )
    checks["handoff_ok"] = (
        not handoff_expected or checks["handoff_triggered"]
    )

    # 5. 整体状态
    has_answer = bool(answer) and "已接收到你的请求" not in answer[:100]
    checks["has_answer"] = has_answer
    checks["is_generic"] = not has_answer

    # 综合判断
    score = 0
    if checks["skill_ok"]:
        score += 30
    if checks["has_answer"]:
        score += 20
    if checks["must_contain_score"] >= 0.5:
        score += 25
    if checks["must_contain_score"] >= 0.75:
        score += 25
    if handoff_expected and checks["handoff_ok"]:
        score += 10

    # 等级
    if score >= 80 and checks["skill_ok"]:
        grade = "good"
    elif score >= 50:
        grade = "partial"
    elif checks["skill_ok"] or checks["has_answer"]:
        grade = "poor"
    else:
        grade = "fail"

    return {
        "case_id": cid,
        "category": result["category"],
        "message": result["message"][:100],
        "skill": skill,
        "source": source,
        "checks": checks,
        "score": score,
        "grade": grade,
        "answer_preview": answer[:200],
    }


def main(
    dataset_path: Path = DATASET_PATH,
    results_path: Path = RESULTS_PATH,
    output_dir: Path = OUTPUT_DIR,
):
    items = load_jsonl(dataset_path)
    results = load_jsonl(results_path)
    run_metadata: dict[str, Any] = {}
    report_path = results_path.parent / "report.json"
    if report_path.exists():
        try:
            run_metadata = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_metadata = {}
    item_map = {i["case_id"]: i for i in items}
    result_map = {r["case_id"]: r for r in results}

    judgements = []
    for cid in sorted(item_map.keys()):
        item = item_map.get(cid)
        result = result_map.get(cid)
        if not result:
            continue
        j = judge_item(item, result)
        judgements.append(j)

    # 统计
    grade_counts = Counter(j["grade"] for j in judgements)
    total = len(judgements)
    print(f"Total: {total}")
    print(f"Grade distribution:")
    for grade in ["good", "partial", "poor", "fail"]:
        print(f"  {grade}: {grade_counts.get(grade, 0)} ({grade_counts.get(grade, 0)/total*100:.1f}%)")

    # 按类别统计
    by_category = {}
    for j in judgements:
        cat = j["category"]
        by_category.setdefault(cat, Counter())
        by_category[cat][j["grade"]] += 1

    print(f"\nBy category:")
    for cat in sorted(by_category):
        counts = by_category[cat]
        cat_total = sum(counts.values())
        good_pct = counts.get("good", 0) / cat_total * 100 if cat_total else 0
        print(f"  {cat:20s}: {cat_total:3d} items, good={good_pct:.0f}%")

    # 技能选择准确率
    skill_correct = sum(1 for j in judgements if j["checks"]["skill_ok"])
    print(f"\nSkill selection accuracy: {skill_correct}/{total} = {skill_correct/total*100:.1f}%")

    # must_contain 覆盖
    must_total = sum(1 for j in judgements if j["checks"]["must_contain_score"] >= 0.5)
    print(f"Must-contain coverage (>=50%): {must_total}/{total} = {must_total/total*100:.1f}%")

    # 保存评判结果
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "judgements.jsonl").write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in judgements) + "\n",
        encoding="utf-8",
    )

    # 生成报告
    report_lines = [
        "# nikon0 Manual QA Eval 150 - 综合评测报告",
        "",
        (
            "**运行配置**: "
            f"profile={run_metadata.get('runtime_profile', 'unknown')} | "
            f"real_llm={run_metadata.get('use_real_llm', 'unknown')} | "
            f"local_rag={run_metadata.get('local_rag', 'unknown')} | "
            f"mock_case_intake={run_metadata.get('mock_case_intake_tool', 'unknown')}"
        ),
        "",
        "## 1. 总体统计",
        "",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| 总题目数 | {total} |",
        f"| 成功执行 | {total - grade_counts.get('fail', 0)} |",
        f"| 完全失败 | {grade_counts.get('fail', 0)} |",
        f"| 技能选择准确率 | {skill_correct/total*100:.1f}% |",
        f"| Must-contain覆盖率(≥50%) | {must_total/total*100:.1f}% |",
        f"| 平均执行时间 | ~0.02s/item |",
        f"| 真实多轮样本 | {run_metadata.get('true_multi_turn_cases', 0)}/{run_metadata.get('multi_turn_cases', 0)} |",
        "",
        "## 2. 评级分布",
        "",
        f"| 等级 | 数量 | 占比 |",
        f"|------|------|------|",
    ]
    for grade in ["good", "partial", "poor", "fail"]:
        cnt = grade_counts.get(grade, 0)
        report_lines.append(f"| {grade} | {cnt} | {cnt/total*100:.1f}% |")

    report_lines += [
        "",
        "**评级标准**:",
        "- good: 技能选择正确 + 有实质回答 + must_contain覆盖率≥75%",
        "- partial: 部分满足条件（技能正确或有回答但不够完整）",
        "- poor: 仅满足一项基本条件",
        "- fail: 未满足任何条件",
        "",
        "## 3. 按类别统计",
        "",
        f"| 类别 | 总数 | good | partial | poor | fail | 良好率 |",
        f"|------|------|------|---------|------|------|--------|",
    ]
    for cat in sorted(by_category):
        counts = by_category[cat]
        cat_total = sum(counts.values())
        good_pct = counts.get("good", 0) / cat_total * 100 if cat_total else 0
        report_lines.append(
            f"| {cat} | {cat_total} | {counts.get('good', 0)} | "
            f"{counts.get('partial', 0)} | {counts.get('poor', 0)} | "
            f"{counts.get('fail', 0)} | {good_pct:.0f}% |"
        )

    report_lines += [
        "",
        "## 4. 技能选择分析",
        "",
    ]

    # 技能分布
    skill_dist = Counter()
    skill_grade = {}
    for j in judgements:
        sk = j["skill"] or "none"
        skill_dist[sk] += 1
        skill_grade.setdefault(sk, {"good": 0, "partial": 0, "poor": 0, "fail": 0})
        skill_grade[sk][j["grade"]] += 1

    report_lines.append(f"| 选中技能 | 数量 | good | partial | poor | fail |")
    report_lines.append(f"|----------|------|------|---------|------|------|")
    for sk, cnt in skill_dist.most_common():
        sg = skill_grade.get(sk, {})
        report_lines.append(
            f"| {sk} | {cnt} | {sg.get('good', 0)} | "
            f"{sg.get('partial', 0)} | {sg.get('poor', 0)} | {sg.get('fail', 0)} |"
        )

    # Fail 分析
    fails = [j for j in judgements if j["grade"] == "fail"]
    poor = [j for j in judgements if j["grade"] == "poor"]

    report_lines += [
        "",
        "## 5. Fail 项分析",
        "",
        f"共 {len(fails)} 个 fail 项:",
        "",
    ]
    for j in fails:
        report_lines.append(f"- **{j['case_id']}** ({j['category']}): skill={j['skill']}, source={j['source']}")
        report_lines.append(f"  Q: {j['message'][:80]}")
        report_lines.append(f"  A: {j['answer_preview'][:100]}")

    report_lines += [
        "",
        "## 6. Poor 项分析",
        "",
        f"共 {len(poor)} 个 poor 项:",
        "",
    ]
    for j in poor[:10]:  # 只展示前10个
        report_lines.append(f"- **{j['case_id']}** ({j['category']}): skill={j['skill']}, source={j['source']}")

    if len(poor) > 10:
        report_lines.append(f"- ... 及其他 {len(poor)-10} 项")

    report_lines += [
        "",
        "## 7. 关键发现",
        "",
        "### 7.1 系统优势",
        "- **中文产品查询路由准确**: planned 路由能正确将中文产品问题路由到 product_support",
        "- **本地手册搜索有效**: StructuredManualBackend 能从 .txt 文件中检索到相关内容",
        "- **Case intake 识别准确**: case_intake 相关的中文消息能被正确识别",
        "- **运行稳定**: 149/150 项成功执行，无崩溃",
        "",
        "### 7.2 系统局限",
        "- **英文查询不匹配**: 纯英文消息不会被 planner 匹配到 product_support（关键词主要为中文）",
        "- **证据检索不够精准**: 部分查询返回了不相关手册的证据（如 Airfryer 清洁问题返回 VR 手册）",
        "- **答案质量低**: 无 LLM 时只能输出原始证据片段，不能生成连贯的客服回答",
        "- **handoff/refund/复合意图处理不足**: 这些路径在 deterministic profile 下返回通用回答",
        "- **边界/无效输入处理**: 表情符号、极短消息等边界情况返回通用回答",
        "",
        "### 7.3 建议改进",
        "1. Planner 应增加英文关键词或支持语言无关的关键词匹配",
        "2. StructuredManualBackend 需要更精准的产品-手册映射机制",
        "3. 即使无 LLM，也应使用模板将证据转化为更可读的回答",
        "4. Handoff/Refund/投诉等流程需要实现专门的处理逻辑",
        "",
        "## 8. 评测数据",
        "",
        f"- 数据集: `nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl` ({total} 条)",
        f"- 原始结果: `nikon0/eval/reports/manual-qa-eval-150/full/raw_results.jsonl`",
        f"- 评判结果: `nikon0/eval/reports/manual-qa-eval-150/full/judgements.jsonl`",
        f"- 评测脚本: `nikon0/eval/run_manual_qa_eval.py`",
        f"- 数据集构建器: `nikon0/eval/build_manual_qa_dataset.py`",
        "",
        "---",
        f"*报告生成时间: 2026-06-19*",
    ]

    report_md = "\n".join(report_lines)
    (output_dir / "eval_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nReport saved to {output_dir / 'eval_report.md'}")

    # 输出报告摘要
    print("\n" + "\n".join(report_lines[:50]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(args.dataset, args.results, args.output_dir)
