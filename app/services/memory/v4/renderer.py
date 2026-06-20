"""v4 issue summary 渲染。"""

from __future__ import annotations

from app.services.memory.v4.types import IssueSummary, IssueThread


class IssueSummaryRenderer:
    def render(self, threads: list[IssueThread]) -> IssueSummary:
        if not threads:
            return IssueSummary(rendered_context="", thread_ids=[])
        lines = ["[当前问题状态]"]
        thread_ids = []
        for thread in threads:
            thread_ids.append(thread.thread_id)
            lines.extend(_render_thread(thread))
        return IssueSummary(
            rendered_context="\n".join(lines),
            thread_ids=thread_ids,
            trace={"thread_count": len(threads)},
        )


def _render_thread(thread: IssueThread) -> list[str]:
    lines = []
    if len(thread.facts) and thread.product_model:
        lines.append(f"- 产品: {thread.product_model}")
    lines.append(f"- 问题类型: {_issue_type_label(thread.issue_type)}")
    facts = _active_facts(thread)
    if facts.get("fault_code"):
        lines.append(f"- 故障码: {', '.join(facts['fault_code'])}")
    if facts.get("symptom"):
        lines.append(f"- 现象: {', '.join(facts['symptom'])}")
    if facts.get("attempted_action"):
        lines.append(f"- 已尝试: {', '.join(facts['attempted_action'])}")
    if facts.get("user_goal"):
        lines.append(f"- 用户诉求: {', '.join(facts['user_goal'])}")
    if facts.get("missing_slot"):
        lines.append(f"- 缺失信息: {', '.join(facts['missing_slot'])}")
    if facts.get("case_id"):
        lines.append(f"- 工单号: {', '.join(facts['case_id'])}")
    turn_ids = []
    for ev in thread.evidence_refs.values():
        if ev.turn_id and ev.turn_id not in turn_ids:
            turn_ids.append(ev.turn_id)
    if turn_ids:
        lines.append(f"- 证据: {', '.join(turn_ids[-4:])}")
    return lines


def _active_facts(thread: IssueThread) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for fact in thread.facts.values():
        if fact.status != "active":
            continue
        grouped.setdefault(fact.kind, [])
        if fact.value not in grouped[fact.kind]:
            grouped[fact.kind].append(fact.value)
    return grouped


def _issue_type_label(value: str) -> str:
    return {
        "fault": "故障排查",
        "howto": "操作咨询",
        "repair": "报修",
        "refund": "退款",
        "complaint": "投诉",
    }.get(value, "未知")
