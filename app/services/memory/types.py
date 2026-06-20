"""记忆系统的数据结构。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.services.context.types import CriticalFacts


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = (value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out

"""
把每轮从对话里抽出来的 CriticalFacts，跨轮累积合并成一份结构化事实表
"""
@dataclass(frozen=True)
class SessionFacts:
    """session 级累积事实，来自每轮 CriticalFacts 的保真合并。"""

    order_ids: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    product_models: list[str] = field(default_factory=list)
    fault_codes: list[str] = field(default_factory=list)
    user_goals: list[str] = field(default_factory=list)
    visual_entities: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    attempted_actions: list[str] = field(default_factory=list)

    @classmethod
    def from_critical_facts(cls, facts: CriticalFacts) -> "SessionFacts":
        return cls(
            order_ids=list(facts.order_ids),
            phones=list(facts.phones),
            product_models=list(facts.product_models),
            fault_codes=list(facts.fault_codes),
            user_goals=list(facts.user_goals),
            visual_entities=list(facts.visual_entities),
            missing_slots=list(facts.missing_slots),
        )

    @classmethod
    def from_dict(cls, obj: dict[str, Any] | None) -> "SessionFacts":
        data = obj or {}
        return cls(**{field_name: [str(x) for x in data.get(field_name, []) if str(x)] for field_name in (
            "order_ids",
            "phones",
            "product_models",
            "fault_codes",
            "user_goals",
            "visual_entities",
            "missing_slots",
            "attempted_actions",
        )})

    def merge(self, other: "SessionFacts") -> "SessionFacts":
        return SessionFacts(
            order_ids=_unique([*self.order_ids, *other.order_ids]),
            phones=_unique([*self.phones, *other.phones]),
            product_models=_unique([*self.product_models, *other.product_models]),
            fault_codes=_unique([*self.fault_codes, *other.fault_codes]),
            user_goals=_unique([*self.user_goals, *other.user_goals]),
            visual_entities=_unique([*self.visual_entities, *other.visual_entities]),
            missing_slots=_unique([*self.missing_slots, *other.missing_slots]),
            attempted_actions=_unique([*self.attempted_actions, *other.attempted_actions]),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "order_ids": list(self.order_ids),
            "phones": list(self.phones),
            "product_models": list(self.product_models),
            "fault_codes": list(self.fault_codes),
            "user_goals": list(self.user_goals),
            "visual_entities": list(self.visual_entities),
            "missing_slots": list(self.missing_slots),
            "attempted_actions": list(self.attempted_actions),
        }

    def count(self) -> int:
        return sum(len(v) for v in self.to_dict().values())

    def render(self) -> str:
        rows: list[str] = []
        mapping = (
            ("订单号", self.order_ids),
            ("联系电话", self.phones),
            ("产品/型号", self.product_models),
            ("故障码/状态码", self.fault_codes),
            ("用户诉求", self.user_goals),
            ("视觉/OCR实体", self.visual_entities),
            ("缺失字段", self.missing_slots),
            ("已尝试操作", self.attempted_actions),
        )
        for label, values in mapping:
            if values:
                rows.append(f"{label}: {', '.join(values)}")
        return "\n".join(rows)


@dataclass(frozen=True)
class SessionMemory:
    """短期会话记忆：结构化槽位 + 滚动摘要。"""

    session_id: str
    facts: SessionFacts = field(default_factory=SessionFacts)
    summary: str = ""
    turn_count: int = 0
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, obj: dict[str, Any] | None, *, session_id: str) -> "SessionMemory":
        data = obj or {}
        return cls(
            session_id=str(data.get("session_id") or session_id),
            facts=SessionFacts.from_dict(data.get("facts") if isinstance(data.get("facts"), dict) else {}),
            summary=str(data.get("summary") or ""),
            turn_count=int(data.get("turn_count") or 0),
            updated_at=float(data.get("updated_at") or time.time()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "facts": self.facts.to_dict(),
            "summary": self.summary,
            "turn_count": self.turn_count,
            "updated_at": self.updated_at,
        }

    def render(self) -> str:
        parts: list[str] = []
        facts_text = self.facts.render()
        if facts_text:
            parts.append("[会话事实]\n" + facts_text)
        if self.summary:
            parts.append("[会话摘要]\n" + self.summary)
        return "\n\n".join(parts)


@dataclass(frozen=True)
class UserProfile:
    """长期用户画像，按 user_key 持久化。"""

    user_key: str
    products: list[str] = field(default_factory=list)
    contact_phones: list[str] = field(default_factory=list)
    active_orders: list[str] = field(default_factory=list)
    historical_issues: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, obj: dict[str, Any] | None, *, user_key: str) -> "UserProfile":
        data = obj or {}
        return cls(
            user_key=str(data.get("user_key") or user_key),
            products=[str(x) for x in data.get("products", []) if str(x)],
            contact_phones=[str(x) for x in data.get("contact_phones", []) if str(x)],
            active_orders=[str(x) for x in data.get("active_orders", []) if str(x)],
            historical_issues=[str(x) for x in data.get("historical_issues", []) if str(x)],
            preferences=[str(x) for x in data.get("preferences", []) if str(x)],
            updated_at=float(data.get("updated_at") or time.time()),
        )

    def merge_facts(self, facts: SessionFacts) -> "UserProfile":
        issues = [*self.historical_issues]
        if facts.fault_codes or facts.user_goals:
            issue = "；".join([*facts.fault_codes, *facts.user_goals])
            if issue:
                issues.append(issue)
        return UserProfile(
            user_key=self.user_key,
            products=_unique([*self.products, *facts.product_models]),
            contact_phones=_unique([*self.contact_phones, *facts.phones]),
            active_orders=_unique([*self.active_orders, *facts.order_ids]),
            historical_issues=_unique(issues),
            preferences=list(self.preferences),
            updated_at=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_key": self.user_key,
            "products": list(self.products),
            "contact_phones": list(self.contact_phones),
            "active_orders": list(self.active_orders),
            "historical_issues": list(self.historical_issues),
            "preferences": list(self.preferences),
            "updated_at": self.updated_at,
        }

    def render(self) -> str:
        rows: list[str] = []
        mapping = (
            ("用户产品", self.products),
            ("联系电话", self.contact_phones),
            ("相关订单", self.active_orders),
            ("历史问题", self.historical_issues),
            ("用户偏好", self.preferences),
        )
        for label, values in mapping:
            if values:
                rows.append(f"{label}: {', '.join(values[:6])}")
        return "\n".join(rows)


@dataclass(frozen=True)
class MemoryNote:
    """向量情景记忆条。"""

    memory_id: str
    user_key: str
    memory_text: str
    memory_type: str = "episodic"
    source_session: str = ""
    score: float = 0.0
    created_at: float = field(default_factory=time.time)
    expire_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_key": self.user_key,
            "memory_text": self.memory_text,
            "memory_type": self.memory_type,
            "source_session": self.source_session,
            "score": self.score,
            "created_at": self.created_at,
            "expire_ts": self.expire_ts,
        }


@dataclass(frozen=True)
class MemoryBundle:
    """一次读路径返回的全部可用记忆。"""

    session_memory: SessionMemory | None = None
    user_profile: UserProfile | None = None
    episodic_notes: list[MemoryNote] = field(default_factory=list)
    user_key: str = ""

    def is_empty(self) -> bool:
        return not self.render().strip()

    def render(self) -> str:
        parts: list[str] = []
        if self.session_memory:
            session_text = self.session_memory.render()
            if session_text:
                parts.append(session_text)
        if self.user_profile:
            profile_text = self.user_profile.render()
            if profile_text:
                parts.append("[长期用户画像]\n" + profile_text)
        notes = [n.memory_text for n in self.episodic_notes if n.memory_text.strip()]
        if notes:
            parts.append("[相关历史记忆]\n" + "\n".join(f"- {n}" for n in notes[:5]))
        if not parts:
            return ""
        return "[记忆]\n" + "\n\n".join(parts)
