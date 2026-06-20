"""压缩结果自检。"""

from __future__ import annotations

from app.services.context.budget import keywords
from app.services.context.evidence_extractor import _has_valid_step_sequence
from app.services.context.types import CriticalFacts, EvidenceBlock


def verify_evidence(question: str, facts: CriticalFacts, blocks: list[EvidenceBlock]) -> list[str]:
    reasons: list[str] = []
    text = "\n".join(b.text for b in blocks)

    for term in _required_terms(question, facts):
        if term and term not in text:
            reasons.append(f"missing_entity:{term}")

    if _is_step_question(question):
        step_blocks = [b for b in blocks if b.block_type == "step_block"]
        if not step_blocks:
            reasons.append("missing_step_block")
        elif not any(_has_valid_step_sequence(b.text) for b in step_blocks):
            reasons.append("broken_step_sequence")

    if any(k in question for k in ("故障码", "指示灯", "状态", "含义")):
        table_blocks = [b for b in blocks if b.block_type == "table_like"]
        if not table_blocks:
            reasons.append("missing_table_block")

    return reasons


def _required_terms(question: str, facts: CriticalFacts) -> list[str]:
    out: list[str] = []
    out.extend(facts.product_models)
    out.extend(facts.fault_codes)
    out.extend(facts.visual_entities[:5])
    # 只把较长关键词纳入硬校验，避免“如何/怎么”这类泛词制造误报。
    out.extend(k for k in keywords(question) if len(k) >= 3 and any(ch.isdigit() for ch in k))
    seen: set[str] = set()
    result: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _is_step_question(question: str) -> bool:
    return any(k in (question or "") for k in ("如何", "怎么", "步骤", "安装", "清洁", "清洗", "更换", "拆卸", "启动", "关闭", "设置", "排查", "操作"))
