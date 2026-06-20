"""Profile 级 v3 记忆存储。"""

from __future__ import annotations

import hashlib
import time
import uuid

from app.services.memory.v3.types import MemoryAtom, UserProfileV3, WriteDecision


class InMemoryUserProfileV3Store:
    def __init__(self) -> None:
        self._profiles: dict[str, UserProfileV3] = {}

    @staticmethod
    def hash_user_key(user_key: str) -> str:
        return "sha256:" + hashlib.sha256(user_key.encode("utf-8")).hexdigest()

    def get(self, user_key: str) -> UserProfileV3 | None:
        return self._profiles.get(self.hash_user_key(user_key))

    def apply_decisions(self, user_key: str, decisions: list[WriteDecision], *, turn_id: str) -> UserProfileV3:
        hashed = self.hash_user_key(user_key)
        profile = self._profiles.setdefault(hashed, UserProfileV3(user_key=hashed))
        for decision in decisions:
            if decision.action != "upsert_profile" or decision.candidate is None:
                continue
            candidate = decision.candidate
            now = time.time()
            atom = MemoryAtom(
                atom_id=f"atom_{uuid.uuid4().hex[:16]}",
                scope="profile",
                kind=candidate.kind,
                value=candidate.value,
                confidence=candidate.confidence,
                source=candidate.source,
                source_turn_id=turn_id,
                source_priority=candidate.source_priority,
                pii_level=candidate.pii_level,
                created_at=now,
                updated_at=now,
                evidence_text=candidate.evidence_text,
            )
            profile.stable_atoms[atom.atom_id] = atom
            if atom.kind == "phone":
                profile.preferred_contact_phone_atom_id = atom.atom_id
            elif atom.kind == "product_model":
                profile.default_product_model_atom_id = atom.atom_id
        profile.updated_at = time.time()
        return profile
