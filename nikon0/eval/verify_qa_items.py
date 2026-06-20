"""验证 QA eval 题目是否真实基于手册内容。

对每条 QA item：
1. 打开它引用的手册文件
2. 搜索 must_include 中的每个短语
3. 搜索 golden_answer 中的关键事实
4. 报告 verified / fabricated / partial_match
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MANUAL_DIR = Path("/Users/nikonzhang/compeletion/手册")


def load_manual_text(manual_path: str | Path) -> str:
    """读取手册 JSON 数组格式的文本内容."""
    path = Path(manual_path)
    if not path.exists():
        # 尝试在 MANUAL_DIR 下查找
        path = MANUAL_DIR / path.name
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # 第一个元素是文本
        return ""
    except json.JSONDecodeError:
        # 部分手册 JSON 含有无效转义，提取第一个字符串元素
        # 格式: ["...text...", [...]] 或 ["...text...", [...]]
        match = re.search(r'\A\s*\[\s*"', raw)
        if not match:
            return ""
        start = match.end() - 1  # 指向开头的 "
        # 找到对应的闭合引号（跳过转义）
        i = start + 1
        while i < len(raw):
            if raw[i] == '\\':
                i += 2
            elif raw[i] == '"':
                # 检查后面是否是 , 或 ]
                after = raw[i+1:].lstrip()
                if after.startswith(',') or after.startswith(']'):
                    return raw[start+1:i]
                i += 1
            else:
                i += 1
        return ""


def normalize_text(text: str) -> str:
    """标准化文本以便匹配：去除多余空白."""
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    """检查短语是否在文本中（模糊匹配，忽略空白差异）.

    同时检查短语的关键词（长度>=2的token）是否有60%以上出现在文本中。
    """
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)
    if phrase_norm.lower() in text_norm.lower():
        return True
    # 词级别模糊匹配：短语中至少60%的关键词出现在文本中
    tokens = [t for t in re.split(r"\s+", phrase_norm) if len(t) >= 2]
    if len(tokens) >= 3:
        found = sum(1 for t in tokens if t.lower() in text_norm.lower())
        return found >= len(tokens) * 0.6
    return False


def verify_qa_item(item: dict) -> dict:
    """验证单条 QA item。

    返回:
      {
        "id": str,
        "manual": str,
        "checks": [{"what": str, "phrase": str, "found": bool, "found_in": str}],
        "status": "verified" | "partial_match" | "fabricated" | "no_manual"
      }
    """
    case_id = item.get("case_id", "unknown")
    manual_rel = item.get("metadata", {}).get("source_manual", "")

    if not manual_rel:
        return {
            "id": case_id,
            "manual": "",
            "checks": [],
            "status": "no_manual",
            "error": "metadata.source_manual is empty",
        }

    manual_text = load_manual_text(manual_rel)
    if not manual_text:
        return {
            "id": case_id,
            "manual": str(manual_rel),
            "checks": [],
            "status": "no_manual",
            "error": f"manual file not found: {manual_rel}",
        }

    checks = []

    # 检查 must_include
    for phrase in item.get("expected", {}).get("answer_must_contain", []):
        found = phrase_in_text(phrase, manual_text)
        checks.append({
            "what": "answer_must_contain",
            "phrase": phrase,
            "found": found,
            "found_in": manual_rel if found else "",
        })

    # 检查 golden_answer 中的关键句（按句号分拆）
    golden = item.get("golden_answer", "")
    facts = [s.strip() for s in golden.replace("；", "。").split("。") if len(s.strip()) >= 4]
    for fact in facts[:5]:  # 最多检查前5句
        found = phrase_in_text(fact[:80], manual_text)  # 取前80字符匹配
        checks.append({
            "what": "golden_answer_fact",
            "phrase": fact[:80],
            "found": found,
            "found_in": manual_rel if found else "",
        })

    total = len(checks)
    found_count = sum(1 for c in checks if c["found"])

    if total == 0:
        status = "no_checks"
    elif found_count == total:
        status = "verified"
    elif found_count >= total * 0.5:
        status = "partial_match"
    else:
        status = "fabricated"

    return {
        "id": case_id,
        "manual": str(manual_rel),
        "checks": checks,
        "total_checks": total,
        "found_count": found_count,
        "status": status,
    }


def verify_dataset(dataset_path: str | Path) -> dict:
    """验证整个数据集."""
    results = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = verify_qa_item(item)
            results.append(result)

    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    return {
        "total": len(results),
        "status_counts": status_counts,
        "results": results,
    }


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl"
    report = verify_dataset(dataset_path)

    print(f"Total items: {report['total']}")
    print(f"Status distribution:")
    for status, count in sorted(report["status_counts"].items()):
        print(f"  {status}: {count}")

    fabricated = [r for r in report["results"] if r["status"] in ("fabricated", "no_manual")]
    if fabricated:
        print(f"\n--- Items needing attention ({len(fabricated)}) ---")
        for r in fabricated:
            print(f"  [{r['status']}] {r['id']}: {r.get('error', '')}")
            for c in r.get("checks", []):
                if not c["found"]:
                    print(f"    MISSING: [{c['what']}] {c['phrase'][:60]}...")

    print("\nDone.")
