"""Case Intake Skill: 售后受理/分诊信息收集。"""

from __future__ import annotations

import re

from app.core.config import settings
from app.services.skills.case_intake_types import CaseIntakeResult, CaseState


_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_ORDER_RE = re.compile(r"\b\d{8,20}\b")


_CANCEL_KWS: tuple[str, ...] = (
    "取消报修",
    "取消工单",
    "不报了",
    "不用了",
    "先不报",
    "算了",
    "不用处理",
)


class CaseIntakeSkill:
    """多轮收集关键字段，完成后输出可执行工单草案。

    ``CASE_INTAKE_REACT_ENABLED=true`` 时，每轮用户发言内用 ReAct（THOUGHT / ACTION /
    Observation）驱动槽位与追问，支持 ACTION=EXIT 中途退出；关闭时仅用规则抽取与模板追问。
    """

    def __init__(self, *, state_store=None) -> None:
        from app.services.skills.case_intake_redis_store import get_case_intake_state_store

        self._store = state_store if state_store is not None else get_case_intake_state_store()

    @staticmethod
    def _session_key(session_id: str) -> str:
        return (session_id or "").strip() or "__default__"

    def has_pending_intake(self, session_id: str) -> bool:
        """是否存在未收齐的工单草稿（用于粘性路由，避免补充信息句被路由误判）。"""
        sid = self._session_key(session_id)
        if sid == "__default__":
            return False
        state = self._store.load(sid)
        if state is None:
            return False
        required = self._required_slots_for_intent(state.intent)
        return any(not state.slots.get(k) for k in required)

    def try_cancel_intake(self, session_id: str, question: str) -> bool:
        """用户明确取消时清除草稿，返回 True 表示已取消。"""
        sid = self._session_key(session_id)
        if sid == "__default__":
            return False
        q = (question or "").strip()
        if not q or not any(k in q for k in _CANCEL_KWS):
            return False
        self._store.delete(sid)
        return True

    def run(
        self,
        *,
        question: str,
        session_id: str,
        conversation_history: str = "",
        enrichment: str = "",
    ) -> CaseIntakeResult:
        sid = self._session_key(session_id)
        q = (question or "").strip()
        if not q:
            return CaseIntakeResult(
                completed=False,
                reply_text="请先描述你的问题现象，例如“设备不启动/有异响/无法充电”。",
                missing_slots=["issue"],
            )

        if sid != "__default__" and self.try_cancel_intake(sid, q):
            return CaseIntakeResult(
                completed=False,
                exited=True,
                reply_text="好的，已取消当前工单收集。如需报修或退款，请随时再发起描述。",
                missing_slots=[],
                ticket_payload={},
                context_block="[工单收集状态]\nstatus: cancelled",
                react_trace=(),
            )

        # 合并多轮对话记忆，便于从上一轮里补全手机号/订单号/型号等槽位。
        blob_parts = [p.strip() for p in (conversation_history, enrichment, q) if (p or "").strip()]
        slot_text = "\n\n".join(blob_parts)[:6000]

        state = self._store.load(sid) or CaseState()
        state.intent = self._detect_intent(slot_text, state.intent)
        # 历史可补全旧槽位，但当前用户发言优先级最高；否则助手追问里的示例
        # “例如：DW-123”会被误抽为真实型号。
        history_text = "\n\n".join(p.strip() for p in (conversation_history, enrichment) if (p or "").strip())
        state.slots.update(self._extract_slots(history_text))
        state.slots.update(self._extract_slots(q))
        self._store.save(sid, state)

        react_trace: tuple[str, ...] = ()
        if settings.case_intake_react_enabled:
            from app.services.skills.case_intake_react import run_case_intake_react

            state, early, trace_list = run_case_intake_react(
                self, session_id=sid, slot_text=slot_text, state=state
            )
            react_trace = tuple(trace_list)
            if early is not None:
                return early
            state = self._store.load(sid) or state

        required = self._required_slots_for_intent(state.intent)
        missing = [k for k in required if not state.slots.get(k)]
        if missing:
            ask = self._build_followup_question(missing)
            return CaseIntakeResult(
                completed=False,
                exited=False,
                reply_text=ask,
                missing_slots=missing,
                ticket_payload=self._build_payload(state, completed=False),
                context_block=self._build_context_block(state, missing),
                react_trace=react_trace,
            )

        payload = self._build_payload(state, completed=True)
        reply = (
            "已为你完成售后受理信息收集。\n"
            f"- 工单类型：{payload.get('intent', '-')}\n"
            f"- 型号：{payload.get('product_model', '-')}\n"
            f"- 问题现象：{payload.get('issue', '-')}\n"
            f"- 优先级：{payload.get('priority', '-')}\n"
            "建议下一步：转人工售后创建正式工单并安排处理。"
        )
        # 本轮已闭环，清状态便于同 session 发起新工单（仍受 TTL 约束）。
        self._store.delete(sid)
        return CaseIntakeResult(
            completed=True,
            exited=False,
            reply_text=reply,
            missing_slots=[],
            ticket_payload=payload,
            context_block=self._build_context_block(state, []),
            react_trace=react_trace,
        )

    @staticmethod
    def _detect_intent(question: str, default_intent: str) -> str:
        q = question.lower()
        if any(k in q for k in ("退款", "退货", "换货")):
            return "refund"
        if any(k in q for k in ("报修", "故障", "坏了", "不转", "无法启动", "不能用")):
            return "repair"
        return default_intent

    @staticmethod
    def _extract_slots(question: str) -> dict[str, str]:
        q = question.strip()
        out: dict[str, str] = {}
        # 联系电话
        m_phone = _PHONE_RE.search(q)
        if m_phone:
            out["contact_phone"] = m_phone.group(0)
        # 订单号。手机号也是 11 位数字，不能被订单号正则误收进去，否则工单 payload
        # 会同时出现 order_id=手机号 和 contact_phone=手机号，后续工具系统容易误判。
        m_order = _ORDER_RE.search(q)
        if m_order and m_order.group(0) != out.get("contact_phone"):
            out["order_id"] = m_order.group(0)
        # 型号（简单规则：型号xxx / model xxx）
        model = ""
        for marker in ("型号", "model", "Model"):
            idx = q.find(marker)
            if idx >= 0:
                tail = q[idx + len(marker):].strip(" ：:，,。")
                if tail:
                    model = re.split(r"[\s，,。；;、]+", tail, maxsplit=1)[0][:32]
                    break
        if model:
            out["product_model"] = model
        # 问题现象
        if any(k in q for k in ("不转", "无法", "故障", "坏了", "异响", "漏电", "冒烟", "不能用")):
            out["issue"] = q[:120]
        # 已尝试操作
        if any(k in q for k in ("重启", "更换", "检查", "试过", "已经")):
            out["attempted_actions"] = q[:120]
        return out

    @staticmethod
    def _required_slots_for_intent(intent: str) -> list[str]:
        if intent == "refund":
            return ["order_id", "issue", "contact_phone"]
        return ["product_model", "issue", "contact_phone"]

    @staticmethod
    def _build_followup_question(missing_slots: list[str]) -> str:
        mapping = {
            "product_model": "请提供产品型号（例如：DW-123）。",
            "issue": "请补充具体故障现象（例如：是否有异响/指示灯状态）。",
            "contact_phone": "请提供联系电话（11位手机号），方便售后联系你。",
            "order_id": "请提供订单号（8-20位数字）。",
            "attempted_actions": "你已经尝试过哪些操作（如重启/更换电池）？",
        }
        prompts = [mapping[s] for s in missing_slots[:2] if s in mapping]
        return "为尽快处理，请补充以下信息：\n" + "\n".join(f"- {p}" for p in prompts)

    @staticmethod
    def _priority(issue: str) -> str:
        low = (issue or "").lower()
        if any(k in low for k in ("漏电", "冒烟", "起火", "烧焦")):
            return "high"
        if any(k in low for k in ("不转", "无法", "不能用", "故障")):
            return "medium"
        return "low"

    def _build_payload(self, state: CaseState, *, completed: bool) -> dict[str, str]:
        issue = state.slots.get("issue", "")
        return {
            "intent": state.intent,
            "product_model": state.slots.get("product_model", ""),
            "issue": issue,
            "attempted_actions": state.slots.get("attempted_actions", ""),
            "order_id": state.slots.get("order_id", ""),
            "contact_phone": state.slots.get("contact_phone", ""),
            "priority": self._priority(issue),
            "status": "ready" if completed else "collecting",
        }

    def _build_context_block(self, state: CaseState, missing: list[str]) -> str:
        payload = self._build_payload(state, completed=not missing)
        lines = ["[工单收集状态]", f"intent: {payload['intent']}"]
        lines.append(f"priority: {payload['priority']}")
        lines.append(f"status: {payload['status']}")
        if missing:
            lines.append("missing: " + ", ".join(missing))
        lines.append("slots:")
        for key in ("product_model", "issue", "attempted_actions", "order_id", "contact_phone"):
            lines.append(f"- {key}: {payload.get(key, '')}")
        return "\n".join(lines)
