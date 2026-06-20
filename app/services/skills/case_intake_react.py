"""工单收集 ReAct：单轮用户输入内多步 THOUGHT→ACTION→Observation，可中途 EXIT。

与检索 ReAct 不同：这里的「工具」是槽位合并与状态检查，不访问向量库。
Observation 由执行器生成，写回下一轮 prompt。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from app.core.config import settings
from app.services.llm_clients import chat_text
from app.services.skills.case_intake_types import CaseIntakeResult, CaseState


class CaseIntakeAgentAction(str, Enum):
    EXTRACT = "EXTRACT"  # 从本轮文本抽取/修正槽位（可带 SLOTS_JSON）
    ASK = "ASK"  # 信息仍不足，向用户追问（本轮结束）
    DONE = "DONE"  # 槽位已齐，闭环工单
    EXIT = "EXIT"  # 用户想中止收集，清空草稿


_RE_THOUGHT = re.compile(r"THOUGHT:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_RE_ACTION = re.compile(
    r"ACTION:\s*(EXTRACT|ASK|DONE|EXIT)",
    re.IGNORECASE,
)


_SYSTEM = (
    "你是售后工单分诊助手，使用 ReAct 方式推进「信息收集」。\n"
    "每轮用户发言你最多输出一次决策，格式必须严格如下（4 行）：\n"
    "THOUGHT: <一句话，≤80字，说明当前判断>\n"
    "ACTION: EXTRACT | ASK | DONE | EXIT\n"
    "SLOTS_JSON: <JSON 对象，仅当 ACTION=EXTRACT 时填写；否则写 {}>\n"
    "USER_FACING: <ACTION 为 ASK/EXIT 时给用户的短回复一句中文；其它写 ->\n"
    "\n"
    "ACTION 含义：\n"
    "- EXTRACT：从用户话里抽取或更新槽位，把结果放进 SLOTS_JSON（键只能出现下文允许的 key）。\n"
    "- ASK：仍缺关键字段，向用户追问（USER_FACING 为追问内容）。\n"
    "- DONE：关键字段已齐，可以生成工单（若实际未齐会被系统纠正为 ASK）。\n"
    "- EXIT：用户明确不想继续填工单、要放弃（USER_FACING 为告别说明）。\n"
    "\n"
    "允许槽位键：product_model, issue, contact_phone, order_id, attempted_actions（字符串值）。\n"
    "repair 意图必填：product_model, issue, contact_phone；refund 必填：order_id, issue, contact_phone。\n"
    "若用户说「算了/不报了/取消」等，应使用 EXIT。\n"
)


def _model_name() -> str:
    m = (settings.case_intake_react_model or "").strip()
    return m or settings.simple_llm_model


def _parse_action(raw: str) -> tuple[CaseIntakeAgentAction, str, dict[str, str], str]:
    text = (raw or "").strip()
    am = _RE_ACTION.search(text)
    if am:
        try:
            action = CaseIntakeAgentAction(am.group(1).upper())
        except ValueError:
            action = CaseIntakeAgentAction.EXTRACT
    else:
        if "EXIT" in text.upper() or "取消" in text:
            action = CaseIntakeAgentAction.EXIT
        elif "DONE" in text.upper() or "完成" in text:
            action = CaseIntakeAgentAction.DONE
        elif "ASK" in text.upper() or "追问" in text:
            action = CaseIntakeAgentAction.ASK
        else:
            action = CaseIntakeAgentAction.EXTRACT

    thought_m = _RE_THOUGHT.search(text)
    thought = (thought_m.group(1).strip() if thought_m else "")[:200]

    slots: dict[str, str] = {}
    sj = _extract_json_object(text)
    if isinstance(sj, dict):
        for k, v in sj.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            key = k.strip()
            if key in {"product_model", "issue", "contact_phone", "order_id", "attempted_actions"}:
                val = v.strip()
                if val:
                    slots[key] = val[:500]

    uf = ""
    if "USER_FACING:" in text:
        tail = text.split("USER_FACING:", 1)[1].strip()
        uf = tail.split("\n", 1)[0].strip()
    if uf in {"", "->", "-"}:
        uf = ""

    return action, thought, slots, uf


def _extract_json_object(text: str) -> Any:
    t = text
    if "```" in t:
        for part in t.split("```"):
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and "}" in p:
                t = p
                break
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _call_llm(prompt: str) -> str:
    return chat_text(
        model=_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=320,
        timeout=30,
    )


def _call_ollama(prompt: str) -> str:
    """兼容旧测试/调试脚本的函数名；实际已改为百炼优先、Ollama 兜底。"""
    return _call_llm(prompt)


def _observation_for_state(skill: Any, state: CaseState) -> str:
    req = skill._required_slots_for_intent(state.intent)  # noqa: SLF001
    missing = [k for k in req if not (state.slots or {}).get(k)]
    filled = {k: (state.slots or {}).get(k, "") for k in req if (state.slots or {}).get(k)}
    return (
        f"intent={state.intent}\n"
        f"已填关键槽位: {filled}\n"
        f"仍缺: {missing if missing else '（无，可 DONE）'}"
    )


def run_case_intake_react(
    skill: Any,
    *,
    session_id: str,
    slot_text: str,
    state: CaseState,
) -> tuple[CaseState, CaseIntakeResult | None, list[str]]:
    """执行 ReAct 子循环。若返回 ``CaseIntakeResult`` 则 ``CaseIntakeSkill.run`` 应直接采用并返回。

    返回 ``(state, None, trace)`` 表示未提前结束，应继续走规则补齐/追问逻辑。
    """
    trace: list[str] = []
    sid = session_id
    max_steps = max(1, settings.case_intake_react_max_steps)
    observation = "（首轮）请阅读用户与上下文，决定 EXTRACT/ASK/DONE/EXIT。"

    for step in range(1, max_steps + 1):
        prompt = "\n".join(
            [
                _SYSTEM,
                "",
                f"【第 {step}/{max_steps} 步】",
                _observation_for_state(skill, state),
                "",
                "用户与上下文（节选）：",
                slot_text[:4000],
                "",
                "上一轮 Observation：",
                observation,
            ]
        ).strip()

        try:
            raw = _call_ollama(prompt)
        except Exception as exc:  # noqa: BLE001
            trace.append(f"step={step} llm_error:{exc}")
            return state, None, trace

        action, thought, llm_slots, user_facing = _parse_action(raw)
        trace.append(f"step={step} action={action.value} thought={thought[:60]!r}")

        if action == CaseIntakeAgentAction.EXIT:
            skill._store.delete(sid)  # noqa: SLF001
            msg = user_facing or "好的，已中止当前工单信息收集。需要时可随时重新发起报修/退款描述。"
            return state, CaseIntakeResult(
                completed=False,
                exited=True,
                reply_text=msg,
                missing_slots=[],
                ticket_payload=skill._build_payload(state, completed=False),  # noqa: SLF001
                context_block="[工单收集状态]\nstatus: aborted\n（用户已中止）",
                react_trace=tuple(trace),
            ), trace

        if action == CaseIntakeAgentAction.EXTRACT:
            if llm_slots:
                state.slots.update(llm_slots)
            from app.services.skills.case_intake_skill import CaseIntakeSkill as _CIS

            state.slots.update(skill._extract_slots(slot_text))  # noqa: SLF001
            state.intent = _CIS._detect_intent(slot_text, state.intent)
            skill._store.save(sid, state)  # noqa: SLF001
            observation = _observation_for_state(skill, state)
            continue

        if action == CaseIntakeAgentAction.DONE:
            req = skill._required_slots_for_intent(state.intent)  # noqa: SLF001
            missing = [k for k in req if not state.slots.get(k)]
            if missing:
                observation = f"系统检查：仍缺 {missing}，不能 DONE，请改 ASK 或继续 EXTRACT。"
                continue
            payload = skill._build_payload(state, completed=True)  # noqa: SLF001
            skill._store.delete(sid)  # noqa: SLF001
            reply = (
                "已为你完成售后受理信息收集。\n"
                f"- 工单类型：{payload.get('intent', '-')}\n"
                f"- 型号：{payload.get('product_model', '-')}\n"
                f"- 问题现象：{payload.get('issue', '-')}\n"
                f"- 优先级：{payload.get('priority', '-')}\n"
                "建议下一步：转人工售后创建正式工单并安排处理。"
            )
            return state, CaseIntakeResult(
                completed=True,
                exited=False,
                reply_text=reply,
                missing_slots=[],
                ticket_payload=payload,
                context_block=skill._build_context_block(state, []),  # noqa: SLF001
                react_trace=tuple(trace),
            ), trace

        if action == CaseIntakeAgentAction.ASK:
            req = skill._required_slots_for_intent(state.intent)  # noqa: SLF001
            missing = [k for k in req if not state.slots.get(k)]
            ask = user_facing.strip() if user_facing else skill._build_followup_question(missing)  # noqa: SLF001
            if ask and "请补充以下信息" not in ask:
                ask = f"为尽快处理，请补充以下信息：\n- {ask}"
            return state, CaseIntakeResult(
                completed=False,
                exited=False,
                reply_text=ask,
                missing_slots=missing,
                ticket_payload=skill._build_payload(state, completed=False),  # noqa: SLF001
                context_block=skill._build_context_block(state, missing),  # noqa: SLF001
                react_trace=tuple(trace),
            ), trace

        trace.append(f"step={step} unknown_action_fallback")

    return state, None, trace
