"""Tests for case-intake workflow protocol decisions."""

from __future__ import annotations

from nikon0.workflows.runtime import WorkflowRuntime, default_workflow_runtime


def test_repair_protocol_collects_missing_case_slots() -> None:
    runtime = default_workflow_runtime()

    decision = runtime.decide(
        message="我的设备坏了，想报修",
        slots={"issue": "我的设备坏了，想报修"},
        intent="repair",
    )

    assert decision.workflow_name == "repair_intake"
    assert decision.risk_level == "low"
    assert decision.requires_approval is False
    assert decision.handoff_required is False
    assert decision.next_tool == "case-intake.collect_case_intake"
    assert set(decision.missing_slots) == {"product_model", "contact_phone"}


def test_refund_protocol_requires_approval_without_auto_commitment() -> None:
    runtime = default_workflow_runtime()

    decision = runtime.decide(
        message="我要退款，麻烦马上处理",
        slots={"issue": "我要退款，麻烦马上处理"},
        intent="refund",
    )

    assert decision.workflow_name == "refund_intake"
    assert decision.risk_level == "high"
    assert decision.requires_approval is True
    assert decision.handoff_required is False
    assert decision.next_tool == "case-intake.collect_case_intake"
    assert "order_id" in decision.missing_slots


def test_complaint_protocol_requires_handoff() -> None:
    runtime = default_workflow_runtime()

    decision = runtime.decide(
        message="我要投诉升级并转人工",
        slots={"issue": "我要投诉升级并转人工"},
        intent="complaint",
    )

    assert decision.workflow_name == "complaint_escalation"
    assert decision.risk_level == "high"
    assert decision.requires_approval is False
    assert decision.handoff_required is True
    assert decision.next_tool == "case-intake.collect_case_intake"


def test_cancel_protocol_short_circuits_active_intake() -> None:
    runtime = default_workflow_runtime()

    decision = runtime.decide(
        message="算了，取消工单",
        slots={},
        intent="repair",
        is_cancel=True,
    )

    assert decision.workflow_name == "case_intake_cancel"
    assert decision.next_tool == "case-intake.try_cancel_case_intake"
    assert decision.stop_when == ["cancelled"]
