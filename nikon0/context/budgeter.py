"""Context budgeter with explicit priorities and degradation reporting."""

from __future__ import annotations

from nikon0.context.pack import ContextBudgetReport, ContextPack, ContextSection


class ContextBudgeter:
    """Apply section budgets and total budget using section priorities.

    Lower `ContextSection.priority` means more important and therefore more
    protected. Degradation happens from highest priority number to lowest.
    """

    def __init__(
        self,
        *,
        total_char_budget: int = 9000,
        section_budgets: dict[str, int] | None = None,
        min_section_chars: int = 40,
    ) -> None:
        self.total_char_budget = max(100, int(total_char_budget))
        self.section_budgets = section_budgets or {}
        self.min_section_chars = max(20, int(min_section_chars))

    def apply(self, sections: list[ContextSection]) -> ContextPack:
        report = ContextBudgetReport(
            total_budget=self.total_char_budget,
            section_budgets=dict(self.section_budgets),
            section_priorities={section.name: section.priority for section in sections},
        )
        budgeted: list[ContextSection] = []
        for section in sections:
            section_budget = self.section_budgets.get(section.name, section.char_budget or self.total_char_budget)
            content, truncated = trim_tail(section.content, section_budget)
            updated = section.model_copy(
                update={
                    "content": content,
                    "truncated": truncated,
                    "token_estimate": _estimate_tokens(content),
                    "char_budget": section_budget,
                }
            )
            budgeted.append(updated)
            if truncated:
                report.truncated_sections.append(section.name)
                report.degraded_sections.append(section.name)

        degraded = self._degrade_to_total_budget(budgeted, report)
        report.used_chars = sum(len(section.content) for section in degraded)
        report.section_chars = {section.name: len(section.content) for section in degraded}
        report.degradation_order = [
            section.name
            for section in sorted(budgeted, key=lambda item: item.priority, reverse=True)
        ]
        return ContextPack(sections=degraded, budget_report=report)

    def _degrade_to_total_budget(
        self,
        sections: list[ContextSection],
        report: ContextBudgetReport,
    ) -> list[ContextSection]:
        by_name = {section.name: section for section in sections}
        total = sum(len(section.content) for section in sections)
        if total <= self.total_char_budget:
            return sections

        for section in sorted(sections, key=lambda item: item.priority, reverse=True):
            if total <= self.total_char_budget:
                break
            current = by_name[section.name]
            if section.name in {"current_user", "system_policy"}:
                continue
            overflow = total - self.total_char_budget
            target_len = max(self.min_section_chars, len(current.content) - overflow)
            if target_len >= len(current.content):
                continue
            if target_len <= self.min_section_chars and current.priority >= 50:
                total -= len(current.content)
                by_name[section.name] = current.model_copy(update={"content": "", "truncated": True})
                if section.name not in report.dropped_sections:
                    report.dropped_sections.append(section.name)
                if section.name not in report.degraded_sections:
                    report.degraded_sections.append(section.name)
                continue
            content, _ = trim_tail(current.content, target_len)
            total -= len(current.content) - len(content)
            by_name[section.name] = current.model_copy(
                update={"content": content, "truncated": True, "token_estimate": _estimate_tokens(content)}
            )
            if section.name not in report.degraded_sections:
                report.degraded_sections.append(section.name)
            if section.name not in report.truncated_sections:
                report.truncated_sections.append(section.name)

        return [section for section in (by_name[item.name] for item in sections) if section.content.strip()]


def trim_tail(content: str, budget: int) -> tuple[str, bool]:
    if budget <= 0:
        return "", bool(content)
    if len(content) <= budget:
        return content, False
    marker = "\n[truncated: kept most recent/relevant tail]\n"
    if budget <= len(marker) + 20:
        return content[-budget:], True
    return marker + content[-(budget - len(marker)):], True


def _estimate_tokens(content: str) -> int:
    return max(1, len(content) // 4) if content else 0
