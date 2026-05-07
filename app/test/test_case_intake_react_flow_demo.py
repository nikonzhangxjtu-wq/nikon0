"""演示「两轮对话 + 每轮内多步 ReAct」工单收集全流程（Ollama 已打桩）。

看打印::

    python -m pytest app/test/test_case_intake_react_flow_demo.py -s

流程说明：
- 第 1 轮用户：报修 + 现象 → ReAct 内部 2 次 LLM（先 EXTRACT 写 issue，再 ASK 追问型号/电话）
- 第 2 轮用户：补型号 + 电话 → ReAct 内部 2 次 LLM（EXTRACT 写入槽位，DONE 结单）
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import app.services.skills.case_intake_skill as cis_mod
from app.core.config import settings
from app.services.skills.case_intake_redis_store import MemoryCaseIntakeStore
from app.services.skills.case_intake_skill import CaseIntakeSkill


def _banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


def _ollama_http_reachable() -> bool:
    try:
        import requests

        base = (settings.ollama_base_url or "").rstrip("/")
        if not base:
            return False
        resp = requests.get(f"{base}/api/tags", timeout=4)
        return bool(resp.ok)
    except Exception:
        return False


def test_case_intake_react_two_turn_full_flow_prints() -> None:
    """打桩 Ollama，打印两轮对话及每轮 ReAct trace。"""
    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_react_flow_demo"

    # 按调用顺序弹出，共 4 次 /api/chat（第 1 轮 2 步 + 第 2 轮 2 步）
    ollama_queue: list[str] = [
        (
            "THOUGHT: 先把用户描述的问题现象写入槽位。\n"
            "ACTION: EXTRACT\n"
            'SLOTS_JSON: {"issue":"电钻运行时有异响，申请报修"}\n'
            "USER_FACING: ->\n"
        ),
        (
            "THOUGHT: repair 仍缺 product_model 与 contact_phone，向用户追问。\n"
            "ACTION: ASK\n"
            "SLOTS_JSON: {}\n"
            "USER_FACING: 为尽快派工，请补充产品型号（如 DW-900）和 11 位手机号。\n"
        ),
        (
            "THOUGHT: 用户已给出型号与电话，写入槽位。\n"
            "ACTION: EXTRACT\n"
            'SLOTS_JSON: {"product_model":"DW-900","contact_phone":"13800138000"}\n'
            "USER_FACING: ->\n"
        ),
        (
            "THOUGHT: 必填槽位已齐，可以闭环工单。\n"
            "ACTION: DONE\n"
            "SLOTS_JSON: {}\n"
            "USER_FACING: ->\n"
        ),
    ]

    def _fake_ollama(prompt: str) -> str:
        _banner("Mock Ollama 本轮收到的 prompt（节选）")
        print((prompt[:520] + "\n...") if len(prompt) > 520 else prompt)
        if not ollama_queue:
            raise RuntimeError("ollama_queue 已空，调用次数与预期不符")
        raw = ollama_queue.pop(0)
        _banner("Mock Ollama 返回（固定脚本）")
        print(raw.strip())
        return raw

    ctx = (
        patch.object(cis_mod.settings, "case_intake_react_enabled", True),
        patch("app.services.skills.case_intake_react._call_ollama", side_effect=_fake_ollama),
    )

    # ---------- 第 1 轮 ----------
    _banner("【第 1 轮】用户发言 → skill.run（内部进入 ReAct 子循环）")
    q1 = "师傅你好，我手电钻异响，想报修一下"
    print(f"用户: {q1!r}")

    with ctx[0], ctx[1]:
        r1 = skill.run(question=q1, session_id=sid, conversation_history="", enrichment="")

    print(f"\n→ completed={r1.completed} exited={r1.exited}")
    print(f"→ reply_text:\n{r1.reply_text}")
    print(f"→ missing_slots: {r1.missing_slots}")
    print(f"→ react_trace ({len(r1.react_trace)} 条):")
    for i, line in enumerate(r1.react_trace, 1):
        print(f"    {i}. {line}")
    print("→ context_block（节选）:")
    print(r1.context_block[:450] + ("..." if len(r1.context_block) > 450 else ""))

    assert r1.completed is False
    assert r1.exited is False
    assert skill.has_pending_intake(sid) is True
    assert len(r1.react_trace) == 2

    # ---------- 第 2 轮 ----------
    _banner("【第 2 轮】用户补充型号与电话 → 再次 skill.run")
    q2 = "型号 DW-900，我手机 13800138000"
    print(f"用户: {q2!r}")

    with ctx[0], ctx[1]:
        r2 = skill.run(question=q2, session_id=sid, conversation_history="", enrichment="")

    print(f"\n→ completed={r2.completed} exited={r2.exited}")
    print(f"→ reply_text:\n{r2.reply_text}")
    print(f"→ missing_slots: {r2.missing_slots}")
    print(f"→ react_trace ({len(r2.react_trace)} 条):")
    for i, line in enumerate(r2.react_trace, 1):
        print(f"    {i}. {line}")

    assert r2.completed is True
    assert r2.exited is False
    assert skill.has_pending_intake(sid) is False
    assert len(r2.react_trace) == 2
    assert "DW-900" in r2.reply_text or "DW-900" in str(r2.ticket_payload)
    assert not ollama_queue, "应恰好消耗 4 次 Ollama 调用"

    _banner("【结束】工单已闭环，同 session 可重新发起新单")
    print("pending_intake =", skill.has_pending_intake(sid))


@pytest.mark.ollama
def test_case_intake_react_two_turn_full_flow_real_ollama() -> None:
    """真实调用 Ollama：多轮推进，直到工单闭环（容忍模型在中间继续追问）。"""
    if not _ollama_http_reachable():
        pytest.skip(
            f"Ollama 不可达: {settings.ollama_base_url!r}；"
            "请先启动 `ollama serve`，并确保模型已可用。"
        )

    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_react_flow_real_ollama"

    _banner("【真实 Ollama】第 1 轮：先报修，触发追问")
    q1 = "手电钻不转了，想申请报修"
    print(f"用户: {q1!r}")
    with patch.object(cis_mod.settings, "case_intake_react_enabled", True):
        r1 = skill.run(question=q1, session_id=sid, conversation_history="", enrichment="")

    print(f"→ completed={r1.completed} exited={r1.exited}")
    print(f"→ reply_text:\n{r1.reply_text}")
    print(f"→ missing_slots: {r1.missing_slots}")
    print(f"→ react_trace ({len(r1.react_trace)}): {list(r1.react_trace)}")
    assert r1.completed is False
    assert r1.exited is False
    assert skill.has_pending_intake(sid) is True

    _banner("【真实 Ollama】第 2 轮：补齐型号+电话，完成工单")
    q2 = "型号 DW-900，联系电话 13800138000"
    print(f"用户: {q2!r}")
    with patch.object(cis_mod.settings, "case_intake_react_enabled", True):
        r2 = skill.run(question=q2, session_id=sid, conversation_history="", enrichment="")

    print(f"→ completed={r2.completed} exited={r2.exited}")
    print(f"→ reply_text:\n{r2.reply_text}")
    print(f"→ missing_slots: {r2.missing_slots}")
    print(f"→ react_trace ({len(r2.react_trace)}): {list(r2.react_trace)}")
    assert r2.exited is False

    # 真实模型存在策略波动：有时第 2 轮会继续追问 attempted_actions。
    # 为保证“闭环流程”测试稳定，这里按最多 2 轮补充信息推进到完成。
    final_result = r2
    if not final_result.completed:
        _banner("【真实 Ollama】第 3 轮：补充已尝试操作，推动结单")
        q3 = (
            "我已经重启过并检查钻头安装，也试过更换电池，问题仍然不转。"
            "如果信息已齐请直接完成工单。"
        )
        print(f"用户: {q3!r}")
        with patch.object(cis_mod.settings, "case_intake_react_enabled", True):
            final_result = skill.run(
                question=q3,
                session_id=sid,
                conversation_history="",
                enrichment="",
            )
        print(f"→ completed={final_result.completed} exited={final_result.exited}")
        print(f"→ reply_text:\n{final_result.reply_text}")
        print(f"→ missing_slots: {final_result.missing_slots}")
        print(
            f"→ react_trace ({len(final_result.react_trace)}): "
            f"{list(final_result.react_trace)}"
        )

    assert final_result.completed is True
    assert final_result.exited is False
    assert skill.has_pending_intake(sid) is False
