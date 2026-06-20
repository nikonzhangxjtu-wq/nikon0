from __future__ import annotations

import asyncio

from nikon0.memory.write_agent import MemoryWriteAgent, MemoryWriteRequest


class _Client:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(self, messages) -> str:
        return self.response


def test_memory_write_agent_returns_candidate_with_stage_metadata() -> None:
    agent = MemoryWriteAgent(
        _Client(
            '{"candidates":[{"update_key":"product_support","fields":{"issue_summary":"洗碗机漏水"},'
            '"confidence":0.88,"reason":"diagnosis","evidence_ids":["manual:1"]}]}'
        )
    )
    request = MemoryWriteRequest(
        source_agent="support",
        execution_stage="diagnosis",
        message="洗碗机漏水",
        evidence_ids=["manual:1"],
    )

    result = asyncio.run(agent.propose(request))

    assert result.valid is True
    assert result.candidates[0].source_agent == "support"
    assert result.candidates[0].execution_stage == "diagnosis"
    assert result.candidates[0].update.value["issue_summary"] == "洗碗机漏水"
    assert result.candidates[0].idempotency_key


def test_memory_write_agent_marks_invalid_json_as_failure() -> None:
    agent = MemoryWriteAgent(_Client("invalid"))

    result = asyncio.run(agent.propose(MemoryWriteRequest(source_agent="service", execution_stage="service_workflow", message="我要退款")))

    assert result.valid is False
    assert result.failure_reason
