"""记忆蒸馏与 ADD/UPDATE/NOOP 规划。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.services.memory.types import MemoryNote, SessionMemory, UserProfile


@dataclass(frozen=True)
class ConsolidationResult:
    """一次蒸馏结果。"""

    profile_updates: dict[str, list[str]] = field(default_factory=dict)
    memory_notes: list[MemoryNote] = field(default_factory=list)
    session_summary: str = ""


class MemoryConsolidator:
    """LLM 优先、规则兜底的轻量记忆蒸馏器。"""

    def consolidate(
        self,
        *,
        session: SessionMemory,
        user_profile: UserProfile | None,
        recent_turns: list[dict[str, str]] | None = None,
    ) -> ConsolidationResult:
        result = self._llm_consolidate(session=session, user_profile=user_profile, recent_turns=recent_turns or [])
        if result is not None:
            return result
        return self._heuristic_consolidate(session=session, user_profile=user_profile)

    def _llm_consolidate(
        self,
        *,
        session: SessionMemory,
        user_profile: UserProfile | None,
        recent_turns: list[dict[str, str]],
    ) -> ConsolidationResult | None:
        if not settings.memory_enabled:
            return None
        try:
            from app.services.llm_clients import call_simple_llm_json
        except Exception:  # noqa: BLE001
            return None
        prompt = _build_consolidation_prompt(session, user_profile, recent_turns)
        try:
            obj = call_simple_llm_json(prompt, model=settings.simple_llm_model)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(obj, dict):
            return None
        return _parse_result(obj, session)

    def _heuristic_consolidate(
        self,
        *,
        session: SessionMemory,
        user_profile: UserProfile | None,
    ) -> ConsolidationResult:
        facts = session.facts
        summary = session.summary or session.render()
        notes: list[MemoryNote] = []
        user_key = user_profile.user_key if user_profile else ""
        fragments: list[str] = []
        if facts.product_models:
            fragments.append(f"用户相关产品/型号：{', '.join(facts.product_models[:4])}")
        if facts.fault_codes:
            fragments.append(f"用户遇到的故障/状态：{', '.join(facts.fault_codes[:4])}")
        if facts.user_goals:
            fragments.append(f"用户诉求：{', '.join(facts.user_goals[:4])}")
        if facts.attempted_actions:
            fragments.append(f"已尝试操作：{', '.join(facts.attempted_actions[:4])}")
        if user_key and fragments:
            memory_text = "；".join(fragments)
            notes.append(
                MemoryNote(
                    memory_id=_memory_id(user_key, memory_text),
                    user_key=user_key,
                    memory_text=memory_text,
                    source_session=session.session_id,
                    expire_ts=time.time() + settings.memory_user_profile_ttl_seconds,
                )
            )
        return ConsolidationResult(
            profile_updates={
                "products": facts.product_models,
                "contact_phones": facts.phones,
                "active_orders": facts.order_ids,
                "historical_issues": [", ".join([*facts.fault_codes, *facts.user_goals])],
            },
            memory_notes=notes,
            session_summary=summary,
        )


def _build_consolidation_prompt(
    session: SessionMemory,
    user_profile: UserProfile | None,
    recent_turns: list[dict[str, str]],
) -> str:
    return (
        "你是客服 Agent 的记忆蒸馏器。请只输出 JSON，字段为 "
        "profile_updates、memory_notes、session_summary。"
        "只保留后续客服有用的稳定事实，不保存无意义寒暄。\n\n"
        f"当前会话记忆:\n{session.render()}\n\n"
        f"已有用户画像:\n{user_profile.render() if user_profile else ''}\n\n"
        f"最近对话:\n{json.dumps(recent_turns, ensure_ascii=False)}"
    )


def _parse_result(obj: dict[str, Any], session: SessionMemory) -> ConsolidationResult:
    profile_updates = obj.get("profile_updates")
    if not isinstance(profile_updates, dict):
        profile_updates = {}
    notes_raw = obj.get("memory_notes")
    notes: list[MemoryNote] = []
    if isinstance(notes_raw, list):
        for item in notes_raw:
            if isinstance(item, dict):
                text = str(item.get("memory_text") or item.get("text") or "").strip()
                user_key = str(item.get("user_key") or "").strip()
            else:
                text = str(item).strip()
                user_key = ""
            if text:
                notes.append(
                    MemoryNote(
                        memory_id=_memory_id(user_key or session.session_id, text),
                        user_key=user_key,
                        memory_text=text,
                        source_session=session.session_id,
                        expire_ts=time.time() + settings.memory_user_profile_ttl_seconds,
                    )
                )
    return ConsolidationResult(
        profile_updates={
            str(k): [str(v) for v in values if str(v)]
            for k, values in profile_updates.items()
            if isinstance(values, list)
        },
        memory_notes=notes,
        session_summary=str(obj.get("session_summary") or ""),
    )


def dedupe_memory_notes(existing: list[MemoryNote], candidates: list[MemoryNote]) -> list[MemoryNote]:
    """ADD/UPDATE/NOOP 的轻量实现：文本近似重复时保留新分数更高者。"""

    by_key: dict[str, MemoryNote] = {_normalize(note.memory_text): note for note in existing}
    for note in candidates:
        key = _normalize(note.memory_text)
        if not key:
            continue
        old = by_key.get(key)
        if old is None or note.score >= old.score:
            by_key[key] = note
    existing_ids = {note.memory_id for note in existing}
    return [note for note in by_key.values() if note.memory_id not in existing_ids]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _memory_id(user_key: str, text: str) -> str:
    return hashlib.sha256(f"{user_key}\n{text}".encode("utf-8")).hexdigest()
