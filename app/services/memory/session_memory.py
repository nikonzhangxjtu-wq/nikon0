"""短期会话记忆存储。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from app.core.config import settings
from app.services.context.types import CriticalFacts
from app.services.memory.types import SessionFacts, SessionMemory


class SessionMemoryStore:
    """Redis STRING + JSON 的 session 记忆存储；测试可注入兼容客户端。"""

    def __init__(self, client: object | None = None) -> None:
        self._r = client
        self._ttl = settings.conversation_ttl_seconds
        self._prefix = (settings.redis_conversation_key_prefix or "kf").strip().rstrip(":")
        self._memory: dict[str, SessionMemory] = {}

    def key_for(self, session_id: str) -> str:
        digest = hashlib.sha256(session_id.strip().encode("utf-8")).hexdigest()
        return f"{self._prefix}:session_mem:{digest}"

    def get(self, session_id: str) -> SessionMemory:
        if not session_id:
            return SessionMemory(session_id="")
        if self._r is None:
            return self._memory.get(session_id, SessionMemory(session_id=session_id))
        raw = self._r.get(self.key_for(session_id))
        if not raw:
            return SessionMemory(session_id=session_id)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return SessionMemory(session_id=session_id)
        return SessionMemory.from_dict(obj if isinstance(obj, dict) else {}, session_id=session_id)

    def save(self, memory: SessionMemory) -> None:
        if not memory.session_id:
            return
        payload = json.dumps(memory.to_dict(), ensure_ascii=False)
        if self._r is None:
            self._memory[memory.session_id] = memory
            return
        self._r.set(self.key_for(memory.session_id), payload, ex=max(60, int(self._ttl)))

    def clear(self, session_id: str) -> None:
        if self._r is None:
            self._memory.pop(session_id, None)
            return
        self._r.delete(self.key_for(session_id))

    def merge_turn_facts(
        self,
        session_id: str,
        facts: CriticalFacts,
        *,
        question: str = "",
        answer: str = "",
    ) -> SessionMemory:
        current = self.get(session_id)
        incoming = SessionFacts.from_critical_facts(facts)
        attempted = _extract_attempted_actions(question, answer)
        incoming = SessionFacts(
            order_ids=incoming.order_ids,
            phones=incoming.phones,
            product_models=incoming.product_models,
            fault_codes=incoming.fault_codes,
            user_goals=incoming.user_goals,
            visual_entities=incoming.visual_entities,
            missing_slots=incoming.missing_slots,
            attempted_actions=attempted,
        )
        merged_facts = current.facts.merge(incoming)
        turn_count = current.turn_count + 1
        summary = current.summary
        if turn_count >= max(1, settings.memory_session_summary_trigger_turns):
            summary = _rollup_summary(summary, question, answer, merged_facts)
        updated = SessionMemory(
            session_id=session_id,
            facts=merged_facts,
            summary=summary,
            turn_count=turn_count,
            updated_at=time.time(),
        )
        self.save(updated)
        return updated


def build_session_memory_store() -> SessionMemoryStore:
    if not settings.memory_enabled:
        return SessionMemoryStore()
    try:
        import redis as redis_lib  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return SessionMemoryStore()
    try:
        client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
    except Exception:  # noqa: BLE001
        return SessionMemoryStore()
    return SessionMemoryStore(client)


def _extract_attempted_actions(question: str, answer: str) -> list[str]:
    text = f"{question}\n{answer}"
    patterns = (
        r"(?:已经|已|试过|尝试过)([^。！？\n]{2,40})",
        r"(?:建议|可以先|请先)([^。！？\n]{2,40})",
    )
    actions: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            item = str(match).strip(" ，,；;：:")
            if item:
                actions.append(item)
    return actions[:6]


def _rollup_summary(summary: str, question: str, answer: str, facts: SessionFacts) -> str:
    lines: list[str] = []
    if summary:
        lines.extend([ln for ln in summary.splitlines() if ln.strip()])
    core = []
    if facts.product_models:
        core.append(f"产品/型号 {', '.join(facts.product_models[:3])}")
    if facts.fault_codes:
        core.append(f"故障/状态 {', '.join(facts.fault_codes[:3])}")
    if facts.user_goals:
        core.append(f"诉求 {', '.join(facts.user_goals[:3])}")
    if facts.attempted_actions:
        core.append(f"已尝试 {', '.join(facts.attempted_actions[:3])}")
    if core:
        lines.append("；".join(core))
    elif question or answer:
        lines.append(f"用户问：{question[:80]}；助手答：{answer[:80]}")
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out[-8:])
