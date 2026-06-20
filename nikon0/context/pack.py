"""Context pack schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ContextSection(BaseModel):
    name: str
    content: str
    priority: int = 100
    source: str = "runtime"
    token_estimate: int = 0
    char_budget: int | None = None
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBudgetReport(BaseModel):
    total_budget: int
    used_chars: int = 0
    section_budgets: dict[str, int] = Field(default_factory=dict)
    section_chars: dict[str, int] = Field(default_factory=dict)
    section_priorities: dict[str, int] = Field(default_factory=dict)
    degradation_order: list[str] = Field(default_factory=list)
    degraded_sections: list[str] = Field(default_factory=list)
    dropped_sections: list[str] = Field(default_factory=list)
    truncated_sections: list[str] = Field(default_factory=list)


class ContextPack(BaseModel):
    sections: list[ContextSection] = Field(default_factory=list)
    budget_report: ContextBudgetReport

    def section(self, name: str) -> ContextSection:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(name)

    def section_map(self) -> dict[str, str]:
        return {section.name: section.content for section in self.sections}

    def render(self) -> str:
        lines = ["[Context Pack]"]
        for section in self.sections:
            lines.append(f"\n[{section.name}]")
            lines.append(section.content or "(empty)")
        return "\n".join(lines)
