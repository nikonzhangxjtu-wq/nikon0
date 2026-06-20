from __future__ import annotations

from nikon0.eval.multi_agent_eval import MultiAgentExpected, MultiAgentEvalCase, score_multi_agent_response
from nikon0.app.schemas.agent import AgentResponse


def test_multi_agent_eval_scores_stage_order_and_trace_completeness() -> None:
    case = MultiAgentEvalCase(
        case_id="composite",
        turns=["洗碗机漏水，我要退款"],
        expected=MultiAgentExpected(
            agent_stages=["support", "service"],
            required_trace_stages=["agent.delegation_plan", "agent.replan"],
        ),
    )
    response = AgentResponse(
        answer="请提供订单号",
        trace_id="trace",
        debug={
            "multi_agent": {"agent_stages": ["support", "service"]},
            "trace": {"events": [{"stage": "agent.delegation_plan"}, {"stage": "agent.replan"}]},
        },
    )

    scored = score_multi_agent_response(case, response)

    assert scored.passed is True
    assert scored.agent_stages == ["support", "service"]


def test_multi_agent_eval_detects_wrong_stage_order() -> None:
    case = MultiAgentEvalCase(
        case_id="composite",
        turns=["洗碗机漏水，我要退款"],
        expected=MultiAgentExpected(agent_stages=["support", "service"]),
    )
    response = AgentResponse(answer="", trace_id="trace", debug={"multi_agent": {"agent_stages": ["service", "support"]}})

    scored = score_multi_agent_response(case, response)

    assert scored.passed is False
    assert "agent stages" in scored.failures[0]
