"""边界测试 - Skill 路由选择模块.

覆盖：sticky 状态损坏、model 选择回退、工具验证边界、置信度阈值、多 skill 竞争.
"""
from __future__ import annotations

import asyncio

import pytest

from nikon0.agent.runtime import AgentRuntime
from nikon0.app.schemas.agent import AgentRequest
from nikon0.app.schemas.capability import (
    FallbackPolicy,
    SkillManifest,
    SkillMatch,
    SkillResult,
    StickyPolicy,
)
from nikon0.skills.base import ManifestDrivenSkillSelector, SkillRegistry
from nikon0.skills.product_support import ProductSupportSkill
from nikon0.tools.runtime import ToolRegistry, ToolRuntime

from tests_deep.conftest import (
    FakeConditionalSkill,
    FakeEmptySkill,
    run,
    make_runtime,
)


class TestSkillRoutingEdgeCases:
    """Skill 路由选择的边界和异常场景."""

    def test_all_skills_reject_produces_general_handle(self):
        """所有 skill 都拒绝时应走 general_handle."""
        class RejectAllSkill:
            name = "reject_all"
            description = "Rejects everything."
            risk_level = "low"
            manifest = SkillManifest(name=name, title="Reject All", description=description)

            async def can_handle(self, context):
                return SkillMatch(matched=False, confidence=0.0, reason="always reject")

            async def run(self, context):
                return SkillResult(status="success", answer_draft="should not run")

        runtime = make_runtime(skill_registry=SkillRegistry([RejectAllSkill()]))
        result = run(runtime, "任意消息")
        assert "nikon0 已接收到你的请求" in result["answer"]
        assert result["debug"]["plan"]["needs_general_handle"] is True

    def test_model_selector_chooses_unknown_skill_gracefully(self):
        """Model 选择不存在的 skill 时应优雅降级."""
        class UnknownSelector(ManifestDrivenSkillSelector):
            async def select(self, context, manifests):
                return self.build_selection(
                    selected_skill="nonexistent_skill",
                    reason="model hallucinated a skill name",
                    confidence=0.95,
                    manifests=manifests,
                )

        runtime = make_runtime(
            skill_registry=SkillRegistry(
                [ProductSupportSkill()],
                selector=UnknownSelector(),
            )
        )
        result = run(runtime, "AC900 显示 E2 怎么处理")
        assert result["debug"]["skill_selection"]["selected_skill"] is None
        assert result["debug"]["skill_selection"]["source"] == "none"
        assert "unknown skill" in result["debug"]["skill_selection"]["reason"].lower()

    def test_model_selector_low_confidence_falls_back_to_planned(self):
        """Model 选择置信度过低时应回退到 planned."""
        class LowConfidenceSelector(ManifestDrivenSkillSelector):
            async def select(self, context, manifests):
                return self.build_selection(
                    selected_skill="product_support",
                    reason="guessing",
                    confidence=0.30,  # 低于 min_confidence=0.55
                    manifests=manifests,
                )

        runtime = make_runtime(
            skill_registry=SkillRegistry(
                [ProductSupportSkill()],
                selector=LowConfidenceSelector(),
            )
        )
        result = run(runtime, "AC900 显示 E2 怎么处理")
        # 低置信度下，应回退到 planned（planner 推荐 product_support）
        assert result["debug"]["trace"]["selected_skills"] == ["product_support"]

    def test_sticky_skill_continues_across_turns(self):
        """Sticky policy 应在 continue_when 状态下保持."""
        from tests_deep.conftest import FakeRecorderTool

        tool = FakeRecorderTool()
        runtime = make_runtime(
            skill_registry=SkillRegistry([FakeConditionalSkill()]),
            tool_runtime=ToolRuntime(registry=ToolRegistry([tool])),
        )
        # 设置 sticky 状态
        state = runtime.memory_store.load("sticky-s1")
        state.flat_state["conditional"] = {"status": "collecting"}
        runtime.memory_store._state["sticky-s1"] = state

        # 修改 conditional 的 manifest 以启用 sticky
        skill = runtime.skill_registry.get("conditional")
        skill.manifest = skill.manifest.model_copy(update={
            "sticky_policy": StickyPolicy(
                enabled=True, continue_when=["collecting"],
                exit_when=["ready"], max_turns=5,
            )
        })

        result = run(runtime, "继续收集信息", session_id="sticky-s1")
        assert result["debug"]["skill_selection"]["source"] == "sticky"
        assert result["debug"]["skill_selection"]["selected_skill"] == "conditional"

    def test_sticky_overstay_blocked_after_max_turns(self):
        """超过 max_turns 的 sticky 应被阻止."""
        runtime = make_runtime(
            skill_registry=SkillRegistry([FakeConditionalSkill()]),
        )
        skill = runtime.skill_registry.get("conditional")
        skill.manifest = skill.manifest.model_copy(update={
            "sticky_policy": StickyPolicy(
                enabled=True, continue_when=["collecting"],
                max_turns=2,
            )
        })

        state = runtime.memory_store.load("overstay-s1")
        state.flat_state["conditional"] = {"status": "collecting"}
        state.flat_state["_sticky_turns"] = {"conditional": 5}  # 已超过 max_turns
        runtime.memory_store._state["overstay-s1"] = state

        result = run(runtime, "继续", session_id="overstay-s1")
        # 不应是 sticky source
        assert result["debug"]["skill_selection"]["source"] != "sticky"

    def test_missing_required_tools_rejects_skill(self):
        """缺少必需工具的 skill 应被拒绝."""
        class ToolDependentSkill:
            name = "tool_heavy"
            description = "Needs a tool."
            risk_level = "low"
            manifest = SkillManifest(
                name=name, title="Tool Heavy", description=description,
                required_tools=["nonexistent.tool_name"],
            )

            async def can_handle(self, context):
                return SkillMatch(matched=True, confidence=0.9, reason="test")

            async def run(self, context):
                return SkillResult(status="success", answer_draft="should not run")

        runtime = make_runtime(
            skill_registry=SkillRegistry([ToolDependentSkill()]),
        )
        result = run(runtime, "trigger tool heavy")
        assert result["debug"]["skill_selection"]["selected_skill"] is None
        assert "missing required tools" in result["debug"]["skill_selection"]["reason"]

    def test_rule_fallback_selection_when_model_unavailable(self):
        """Model 不可用时，回退到 rule_fallback 选择."""
        class FailingSelector(ManifestDrivenSkillSelector):
            async def select(self, context, manifests):
                raise RuntimeError("model service unavailable")

        runtime = make_runtime(
            skill_registry=SkillRegistry(
                [ProductSupportSkill()],
                selector=FailingSelector(),
            )
        )
        result = run(runtime, "AC900 显示 E2 怎么处理")
        # model 失败，应回退到 planned（planner 推荐 product_support）
        assert "product_support" in result["debug"]["trace"]["selected_skills"]

    def test_empty_skill_produces_minimal_answer(self):
        """产出空 answer 的 skill 被正确处理."""
        runtime = make_runtime(skill_registry=SkillRegistry([FakeEmptySkill()]))
        result = run(runtime, "trigger empty")
        assert result["answer"] == "已完成处理。"

    def test_skill_without_manifest_gets_default(self):
        """没有 manifest 的 skill 获得默认 manifest."""
        class NoManifestSkill:
            name = "no_manifest"
            description = "No manifest."
            risk_level = "medium"

            async def can_handle(self, context):
                return SkillMatch(matched=True, confidence=0.7, reason="test")

            async def run(self, context):
                return SkillResult(status="success", answer_draft="ok", risk_level="medium")

        registry = SkillRegistry([NoManifestSkill()])
        manifests = registry.manifests()
        assert len(manifests) == 1
        assert manifests[0].name == "no_manifest"
        assert manifests[0].risk_level == "medium"  # 继承自 skill.risk_level

    def test_skill_registry_get_case_insensitive(self):
        """Skill name 查找应大小写不敏感."""
        registry = SkillRegistry([FakeConditionalSkill()])
        assert registry.get("conditional") is not None
        assert registry.get("CONDITIONAL") is not None
        assert registry.get("Conditional") is not None

    def test_selector_build_selection_structure(self):
        """build_selection 产出的结构完整."""
        class TestSelector(ManifestDrivenSkillSelector):
            pass

        selector = TestSelector()
        manifests = (FakeConditionalSkill().manifest, FakeEmptySkill().manifest)
        selection = selector.build_selection(
            selected_skill="conditional",
            reason="test selection",
            confidence=0.8,
            manifests=manifests,
        )
        assert selection.selected_skill == "conditional"
        assert len(selection.candidates) == 2
        assert len(selection.rejected) == 1
        assert selection.rejected[0].name == "empty"
        assert selection.source == "model"
