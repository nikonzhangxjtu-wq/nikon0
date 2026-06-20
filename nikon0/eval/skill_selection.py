"""Skill selection evaluation harness."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest


class SkillSelectionCase(BaseModel):
    case_id: str
    message: str
    acceptable_skills: list[str | None]
    session_id: str | None = None
    user_id: str | None = None


class SkillSelectionCaseResult(BaseModel):
    case_id: str
    passed: bool
    selected_skill: str | None
    acceptable_skills: list[str | None]
    source: str
    reason: str
    trace_id: str


class SkillSelectionReport(BaseModel):
    total: int
    passed: int
    accuracy: float
    results: list[SkillSelectionCaseResult] = Field(default_factory=list)


class SkillSelectionHarness:
    """Runs golden skill-selection cases through a runtime instance."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run_cases(self, cases: list[SkillSelectionCase]) -> SkillSelectionReport:
        results: list[SkillSelectionCaseResult] = []
        for index, case in enumerate(cases, start=1):
            response = await self.runtime.run(
                AgentRequest(
                    session_id=case.session_id or f"eval-{case.case_id}-{index}",
                    user_id=case.user_id,
                    message=case.message,
                )
            )
            selection = response.debug.get("skill_selection") or {}
            selected_skill = selection.get("selected_skill")
            passed = selected_skill in case.acceptable_skills
            results.append(
                SkillSelectionCaseResult(
                    case_id=case.case_id,
                    passed=passed,
                    selected_skill=selected_skill,
                    acceptable_skills=case.acceptable_skills,
                    source=str(selection.get("source") or "none"),
                    reason=str(selection.get("reason") or ""),
                    trace_id=response.trace_id,
                )
            )
        passed_count = sum(1 for result in results if result.passed)
        total = len(results)
        return SkillSelectionReport(
            total=total,
            passed=passed_count,
            accuracy=(passed_count / total) if total else 0.0,
            results=results,
        )
