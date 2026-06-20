"""Skill that exercises ToolRuntime without external dependencies."""

from __future__ import annotations

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import SkillManifest, SkillMatch, SkillResult, ToolCallRequest


class ToolEchoSkill:
    name = "tool_echo"
    description = "Calls mock.echo to verify ToolRuntime lifecycle."
    risk_level = "low"
    manifest = SkillManifest(
        name=name,
        title="Tool Echo",
        description=description,
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={
            "type": "object",
            "properties": {"tool_result": {"type": "object"}},
        },
        capabilities=["tool_runtime_smoke_test"],
        required_tools=["mock.echo"],
        risk_level="low",
    )

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        message = context.request.message.strip().lower()
        if "tool echo" in message or "工具回声" in context.request.message:
            return SkillMatch(
                matched=True,
                confidence=0.95,
                reason="matched tool echo verification phrase",
            )
        return SkillMatch(matched=False, confidence=0.0, reason="no tool echo signal")

    async def run(self, context: AgentContext) -> SkillResult:
        if context.tool_results:
            last_result = context.tool_results[-1]
            return SkillResult(
                status="success",
                answer_draft=f"已完成工具调用验证，工具返回：{last_result.get('data', {})}",
                risk_level="low",
            )
        return SkillResult(
            status="success",
            answer_draft="已完成工具调用验证。",
            tool_calls=[
                ToolCallRequest(
                    service_id="mock",
                    tool_name="echo",
                    arguments={"message": context.request.message},
                    risk_level="low",
                )
            ],
            risk_level="low",
        )
