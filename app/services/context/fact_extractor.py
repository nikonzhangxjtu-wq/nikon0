"""关键事实抽取。

这些事实进入 [关键事实] 区块，不参与普通文本裁剪。
"""

from __future__ import annotations

import re

from app.services.context.types import CriticalFacts

_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ORDER_RE = re.compile(r"(?<!\d)\d{8,20}(?!\d)")
_FAULT_RE = re.compile(r"\b[A-Z]{1,4}\d{1,4}\b|[Ee]\d{1,4}|故障码[:：]?\s*([\w-]+)")
_MODEL_MARKER_RE = re.compile(r"(?:型号|model|Model|设备|产品)\s*[:：]?\s*([A-Za-z0-9_-]{2,32})")
_VISUAL_ENT_RE = re.compile(r"关键实体[:：]\s*([^\n]+)|OCR文字[:：]\s*([^\n]+)|产品类型[:：]\s*([^\n]+)")

_GOAL_KEYWORDS = (
    "退款",
    "退货",
    "换货",
    "补寄",
    "发票",
    "投诉",
    "报修",
    "维修",
    "赔偿",
    "安装",
    "清洁",
    "更换",
    "拆卸",
    "启动",
    "关闭",
    "设置",
    "故障排除",
)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = (item or "").strip(" ，,。；;：:")
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_critical_facts(
    *,
    question: str,
    context_block: str = "",
    visual_context: str = "",
    conversation_history: str = "",
    memory_context: str = "",
) -> CriticalFacts:
    blob = "\n".join(
        [
            question or "",
            visual_context or "",
            conversation_history or "",
            memory_context or "",
            context_block or "",
        ]
    )

    order_ids = _unique(_ORDER_RE.findall(blob))
    phones = _unique(_PHONE_RE.findall(blob))
    product_models = _unique(_MODEL_MARKER_RE.findall(blob))
    fault_codes = _unique([m.group(1) or m.group(0) for m in _FAULT_RE.finditer(blob)])

    goals = [kw for kw in _GOAL_KEYWORDS if kw in (question or "")]

    visual_entities: list[str] = []
    for match in _VISUAL_ENT_RE.findall(visual_context or ""):
        for group in match:
            if not group:
                continue
            visual_entities.extend(re.split(r"[,，、;；\s]+", group))

    missing_slots: list[str] = []
    for line in (context_block or "").splitlines():
        if line.startswith("missing:"):
            missing_slots.extend(re.split(r"[,，、\s]+", line.split(":", 1)[1]))

    return CriticalFacts(
        order_ids=order_ids,
        phones=phones,
        product_models=product_models,
        fault_codes=fault_codes,
        user_goals=_unique(goals),
        visual_entities=_unique(visual_entities),
        missing_slots=_unique(missing_slots),
    )


def render_facts(facts: CriticalFacts) -> str:
    if facts.count() <= 0:
        return ""
    rows: list[str] = ["[关键事实]"]
    mapping = (
        ("订单号", facts.order_ids),
        ("联系电话", facts.phones),
        ("产品/型号", facts.product_models),
        ("故障码/状态码", facts.fault_codes),
        ("用户诉求", facts.user_goals),
        ("视觉/OCR实体", facts.visual_entities),
        ("缺失字段", facts.missing_slots),
    )
    for label, values in mapping:
        if values:
            rows.append(f"{label}: {', '.join(values)}")
    return "\n".join(rows)
