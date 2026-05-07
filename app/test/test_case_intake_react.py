"""Case intake ReAct：打桩 Ollama，验证 EXIT / 追问路径。"""

from __future__ import annotations

from unittest.mock import patch

import app.services.skills.case_intake_skill as cis_mod
from app.services.skills.case_intake_redis_store import MemoryCaseIntakeStore
from app.services.skills.case_intake_skill import CaseIntakeSkill


def test_case_intake_react_exit_via_llm() -> None:
    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_react_exit"
    skill.run(question="电钻坏了帮我报修", session_id=sid, conversation_history="", enrichment="")
    assert skill.has_pending_intake(sid) is True

    llm = (
        "THOUGHT: 用户明确表示不想继续填单。\n"
        "ACTION: EXIT\n"
        "SLOTS_JSON: {}\n"
        "USER_FACING: 好的，已为你中止工单收集。\n"
    )
    with (
        patch.object(cis_mod.settings, "case_intake_react_enabled", True),
        patch("app.services.skills.case_intake_react._call_ollama", return_value=llm),
    ):
        r = skill.run(
            question="先停一下，我不想继续填工单了",
            session_id=sid,
            conversation_history="",
            enrichment="",
        )

    assert r.exited is True
    assert r.completed is False
    assert "中止" in r.reply_text
    assert skill.has_pending_intake(sid) is False
    assert len(r.react_trace) >= 1


def test_case_intake_react_ask_after_extract_mocked() -> None:
    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_react_ask"
    skill.run(question="我要报修，机器异响", session_id=sid, conversation_history="", enrichment="")

    responses = [
        (
            "THOUGHT: 先写入电话\n"
            "ACTION: EXTRACT\n"
            'SLOTS_JSON: {"contact_phone":"13900001111"}\n'
            "USER_FACING: ->\n"
        ),
        (
            "THOUGHT: 仍缺型号\n"
            "ACTION: ASK\n"
            "SLOTS_JSON: {}\n"
            "USER_FACING: 请补充产品型号（例如 DW-100）。\n"
        ),
    ]
    call_i = {"n": 0}

    def _fake_llm(_prompt: str) -> str:
        i = call_i["n"]
        call_i["n"] = i + 1
        return responses[i]

    with (
        patch.object(cis_mod.settings, "case_intake_react_enabled", True),
        patch("app.services.skills.case_intake_react._call_ollama", side_effect=_fake_llm),
    ):
        r = skill.run(
            question="电话是13900001111",
            session_id=sid,
            conversation_history="",
            enrichment="",
        )

    assert r.completed is False
    assert r.exited is False
    assert "型号" in r.reply_text
    assert len(r.react_trace) >= 2
