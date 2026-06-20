"""共享 fixtures for nikon0 深度测试套件."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.capability import (
    Evidence,
    FallbackPolicy,
    SkillManifest,
    SkillMatch,
    SkillResult,
    StateUpdate,
    StickyPolicy,
    ToolCallRequest,
    ToolCallResult,
    ToolSpec,
)
from nikon0.app.schemas.knowledge import KnowledgeRequest, KnowledgeResult
from nikon0.app.schemas.memory import IssueFact, IssueThread, SessionIssueMemory
from nikon0.app.schemas.planner import PlannerResult
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.memory.session import InMemorySessionIssueStore
from nikon0.skills.base import ManifestDrivenSkillSelector, SkillRegistry
from nikon0.skills.case_intake import CaseIntakeSkill
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.tools.case_intake import ExtractCaseSlotsTool
from nikon0.tools.runtime import HookRunner, ToolRegistry, ToolRuntime


# ── fake / mock 组件 ──────────────────────────────────────────────


class FakeRecorderTool:
    """记录每次调用的工具，用于验证工具是否被正确调用."""
    spec = ToolSpec(
        service_id="test", tool_name="recorder",
        description="Records every call for verification.", risk_level="low",
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        self.calls.append({
            "service_id": request.service_id,
            "tool_name": request.tool_name,
            "arguments": dict(request.arguments),
        })
        return ToolCallResult(
            ok=True, service_id=request.service_id, tool_name=request.tool_name,
            data={"recorded": True, "call_index": len(self.calls)},
        )


class FakeFailingTool:
    """总是失败的工具，可控制失败次数."""
    spec = ToolSpec(
        service_id="test", tool_name="failing",
        description="Always fails.", risk_level="low",
    )

    def __init__(self, fail_count: int = 999, error_code: str = "simulated_error") -> None:
        self.fail_count = fail_count
        self.error_code = error_code
        self.calls = 0

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        self.calls += 1
        return ToolCallResult(
            ok=False,
            service_id=request.service_id, tool_name=request.tool_name,
            error_code=self.error_code,
            error_message=f"Simulated failure #{self.calls}",
        )


class FakeSlowTool:
    """模拟慢响应."""
    spec = ToolSpec(
        service_id="test", tool_name="slow",
        description="Slow tool.", risk_level="low",
    )

    def __init__(self, delay: float = 5.0) -> None:
        self.delay = delay

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        await asyncio.sleep(0.01)  # 测试中不真实等待
        return ToolCallResult(
            ok=True, service_id=request.service_id, tool_name=request.tool_name,
            data={"slow": True},
        )


class FakeLargeDataTool:
    """返回大量数据的工具."""
    spec = ToolSpec(
        service_id="test", tool_name="large_data",
        description="Returns very large data.", risk_level="low",
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        size = int(request.arguments.get("size", 10000))
        return ToolCallResult(
            ok=True, service_id=request.service_id, tool_name=request.tool_name,
            data={"text": "X" * size, "items": list(range(size))},
        )


# ── fake skill 组件 ────────────────────────────────────────────────


class FakeMultiToolSkill:
    """产出多个并发工具调用的 skill."""
    name = "multi_tool"
    description = "Produces multiple tool calls to test loop behavior."
    risk_level = "low"
    manifest = SkillManifest(
        name=name, title="Multi Tool", description=description,
        required_tools=["test.recorder"],
        risk_level="low",
    )

    def __init__(self, tool_count: int = 3) -> None:
        self.tool_count = tool_count

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        return SkillMatch(matched=True, confidence=0.9, reason="test multi tool")

    async def run(self, context: AgentContext) -> SkillResult:
        if context.tool_results:
            return SkillResult(
                status="success",
                answer_draft=f"已完成{self.tool_count}个工具调用",
                risk_level="low",
            )
        return SkillResult(
            status="success",
            answer_draft="",
            tool_calls=[
                ToolCallRequest(
                    service_id="test", tool_name="recorder",
                    arguments={"index": i},
                )
                for i in range(self.tool_count)
            ],
            risk_level="low",
        )


class FakeConditionalSkill:
    """根据上下文条件决定产出的 skill."""
    name = "conditional"
    description = "Conditional skill."
    risk_level = "low"
    manifest = SkillManifest(
        name=name, title="Conditional", description=description,
        risk_level="low",
    )

    def __init__(self, *, should_fail: bool = False, answer: str = "conditional answer"):
        self.should_fail = should_fail
        self.answer = answer

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        return SkillMatch(matched=True, confidence=0.8, reason="test")

    async def run(self, context: AgentContext) -> SkillResult:
        if self.should_fail:
            raise RuntimeError("conditional skill explosion")
        return SkillResult(status="success", answer_draft=self.answer, risk_level="low")


class FakeStateMutatingSkill:
    """产生复杂 state_update 的 skill."""
    name = "state_mutator"
    description = "Mutates session state in complex ways."
    risk_level = "low"
    manifest = SkillManifest(
        name=name, title="State Mutator", description=description,
        risk_level="low",
    )

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        return SkillMatch(matched=True, confidence=0.9, reason="test")

    async def run(self, context: AgentContext) -> SkillResult:
        return SkillResult(
            status="success",
            answer_draft="state mutated",
            state_updates=[
                StateUpdate(key="nested", value={
                    "level1": {"level2": {"level3": "deep_value"}},
                    "array": [1, 2, 3, {"key": "val"}],
                }, reason="deeply nested state"),
                StateUpdate(key="very_long_key_" + "x" * 200, value="v", reason="long key"),
                StateUpdate(key="special_chars", value="\x00\n\r\t\b", reason="special chars"),
                StateUpdate(key="unicode", value="🎉💾中文한국어", reason="unicode"),
            ],
            risk_level="low",
        )


class FakeEmptySkill:
    """产出空 answer 和空 state_updates 的 skill."""
    name = "empty"
    description = "Produces empty results."
    risk_level = "low"
    manifest = SkillManifest(
        name=name, title="Empty", description=description,
        risk_level="low",
    )

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        return SkillMatch(matched=True, confidence=0.95, reason="test")

    async def run(self, context: AgentContext) -> SkillResult:
        return SkillResult(status="success", answer_draft="", risk_level="low")


# ── 帮助函数 ──────────────────────────────────────────────────────


def make_context(message: str = "test", session_id: str = "test-session") -> AgentContext:
    trace = ExecutionTrace(trace_id="test-trace", session_id=session_id, user_message=message)
    return AgentContext(
        request=AgentRequest(session_id=session_id, message=message),
        trace=trace,
    )


def make_runtime(**overrides) -> AgentRuntime:
    defaults = {
        "skill_registry": SkillRegistry([]),
        "tool_runtime": ToolRuntime(registry=ToolRegistry([])),
        "memory_store": InMemorySessionIssueStore(),
    }
    defaults.update(overrides)
    return AgentRuntime(**defaults)


def run(runtime: AgentRuntime, message: str, session_id: str = "s1") -> dict[str, Any]:
    """同步辅助：运行一次 AgentRuntime.run() 并返回 debug dict。"""
    response = asyncio.run(
        runtime.run(AgentRequest(session_id=session_id, message=message))
    )
    return {
        "answer": response.answer,
        "risk_level": response.risk_level,
        "trace_id": response.trace_id,
        "actions": [a.model_dump() for a in response.actions],
        "debug": response.debug,
    }


# ── manual 目录 fixture ───────────────────────────────────────────


@pytest.fixture
def manual_dir(tmp_path: Path) -> Path:
    d = tmp_path / "manuals"
    d.mkdir()
    (d / "AC900手册.txt").write_text(
        "AC900 显示 E2 表示滤网堵塞或风道异常。处理步骤：关闭电源，清洁滤网，检查风道，重新启动。",
        encoding="utf-8",
    )
    (d / "BC200手册.txt").write_text(
        "BC200 显示 E2 表示温度传感器故障。请联系售后服务中心进行检修。",
        encoding="utf-8",
    )
    (d / "安全指南.txt").write_text(
        "操作前务必断电。不要在潮湿环境中使用电器。高压部件只能由专业人员维修。",
        encoding="utf-8",
    )
    return d
