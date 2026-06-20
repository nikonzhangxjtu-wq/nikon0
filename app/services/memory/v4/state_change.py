"""v4 状态变化检测。"""

from __future__ import annotations

import re

from app.services.memory.v3.types import TurnEvidencePacket
from app.services.memory.v4.types import IssueFactCandidate, SessionIssueMemory, StateChange


class StateChangeDetector:
    def detect(self, packet: TurnEvidencePacket, memory: SessionIssueMemory) -> StateChange:
        q = packet.question or ""
        if _is_pure_howto(packet):
            return StateChange(False, "no_change", [], "纯手册 how-to，不改变用户问题状态")

        candidates: list[IssueFactCandidate] = []
        source = "user"
        products = _extract_product_models(q)
        for value in products:
            candidates.append(_candidate("product_model", value, q, source=source))
        for value in _extract_fault_codes(q):
            candidates.append(_candidate("fault_code", value, q, source=source))
        for value in _extract_attempted_actions(q):
            candidates.append(_candidate("attempted_action", value, q, source=source))
        for value in _extract_symptoms(q):
            candidates.append(_candidate("symptom", value, q, source=source))
        for value in _extract_user_goals(q):
            candidates.append(_candidate("user_goal", value, q, source=source))

        if packet.branch_result:
            payload = packet.branch_result
            if payload.get("case_id"):
                candidates.append(_candidate("case_id", str(payload["case_id"]), str(payload), source="tool", priority=100))
            status = payload.get("status") or payload.get("case_status")
            if status:
                candidates.append(_candidate("case_status", str(status), str(payload), source="tool", priority=100))
            if payload.get("missing_slots"):
                for slot in payload["missing_slots"]:
                    candidates.append(_candidate("missing_slot", str(slot), str(payload), source="tool", priority=90))
            if payload.get("product_model"):
                candidates.append(_candidate("product_model", str(payload["product_model"]), str(payload), source="tool", priority=100))

        if _has_denial(q):
            for value in _extract_attempted_actions(q):
                candidates.append(
                    IssueFactCandidate(
                        kind="attempted_action",
                        value=value,
                        source="user",
                        confidence=0.92,
                        evidence_text=q,
                        status="rejected",
                    )
                )
            return StateChange(bool(candidates), "denial", candidates, "用户否定旧事实")

        if _has_correction(q):
            return StateChange(bool(candidates), "correction", candidates, "用户纠错")

        if _has_resolution(q):
            candidates.append(_candidate("resolution", "已解决", q, source=source))
            return StateChange(True, "resolution", candidates, "用户反馈问题已解决")

        if not candidates:
            return StateChange(False, "no_change", [], "未检测到问题状态变化")
        return StateChange(True, "new_fact", candidates, "检测到新的问题状态事实")


def _candidate(kind: str, value: str, evidence: str, *, source: str, priority: int = 80) -> IssueFactCandidate:
    return IssueFactCandidate(
        kind=kind,
        value=value,
        source=source,
        confidence=0.95 if source == "tool" else 0.88,
        evidence_text=evidence,
        source_priority=priority,
    )


def _is_pure_howto(packet: TurnEvidencePacket) -> bool:
    q = packet.question or ""
    if re.search(r"怎么|如何|步骤|清洗|安装|拆卸|使用", q) and not re.search(
        r"已经|试过|还是|不行|报修|退款|投诉|显示|故障|红灯|闪|无法|不能",
        q,
    ):
        return True
    return False


def _extract_product_models(text: str) -> list[str]:
    return _unique(re.findall(r"\b[A-Z]{1,6}[-_]?\d{2,6}[A-Z0-9-]*\b", text or ""))


def _extract_fault_codes(text: str) -> list[str]:
    product_models = set(_extract_product_models(text))
    values = re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", text or "")
    return _unique([
        v for v in values
        if not re.fullmatch(r"1[3-9]\d{9}", v) and v not in product_models
    ])


def _extract_attempted_actions(text: str) -> list[str]:
    actions = []
    for pattern in [
        r"(断电重启|重启|清洗滤网|清洗过滤网|拆下滤网|更换滤芯|重新安装)",
        r"(?:已经|已|试过|尝试过|按你说的)(?:把|进行)?([^，。；;!?？]{2,18}?)(?:了|过)",
    ]:
        for value in re.findall(pattern, text or ""):
            value = str(value).strip(" ，,。了过")
            if value:
                actions.append(_normalize_action(value))
    return _unique(actions)


def _extract_symptoms(text: str) -> list[str]:
    symptoms = []
    for pattern in [
        r"(还是不行)",
        r"((?:红灯|绿灯|黄灯|蓝灯).{0,4}(?:闪|闪烁|常亮))",
        r"(无法启动|不能启动|不制冷|不工作)",
        r"(显示\s*[A-Z]{1,3}\d{1,4})",
    ]:
        symptoms.extend([str(x).replace(" ", "") for x in re.findall(pattern, text or "")])
    return _unique(symptoms)


def _extract_user_goals(text: str) -> list[str]:
    goals = []
    if re.search(r"报修|维修|上门", text or ""):
        goals.append("报修")
    if re.search(r"退款|退货", text or ""):
        goals.append("退款")
    if re.search(r"投诉|升级处理", text or ""):
        goals.append("投诉")
    return goals


def _has_correction(text: str) -> bool:
    return bool(re.search(r"不是.+是|刚才说错|改成|更正", text or ""))


def _has_denial(text: str) -> bool:
    return bool(re.search(r"没有|没.*过|并没有", text or ""))


def _has_resolution(text: str) -> bool:
    return bool(re.search(r"好了|解决了|恢复了|正常了", text or ""))


def _normalize_action(value: str) -> str:
    if "断电" in value and "重启" in value:
        return "断电重启"
    return value.replace("过滤网", "滤网")


def _unique(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
