"""LLM 记忆判断器。

LLM 只输出结构化候选，不能直接写库；所有结果还要经过 schema 校验和 WriteGate。
"""

from __future__ import annotations

import json
import re
from typing import Callable, Any

from app.core.config import settings
from app.services.llm_clients import chat_text
from app.services.memory.v3.types import (
    LlmMemoryJudgement,
    ObservationCandidate,
    RawEvidence,
    TurnEvidencePacket,
)


class LlmMemoryJudge:
    def __init__(
        self,
        call_model: Callable[[str], str] | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.model = model or getattr(settings, "memory_v3_llm_judge_model", "") or settings.gen_model
        self._call_model = call_model

    def judge(
        self,
        *,
        packet: TurnEvidencePacket,
        raw_evidence: list[RawEvidence],
        current_session_memory: dict[str, Any],
    ) -> LlmMemoryJudgement | None:
        prompt = self._build_prompt(packet, raw_evidence, current_session_memory)
        try:
            raw = self._call(prompt)
        except Exception:
            return None
        return self.parse(raw)

    def parse(self, raw: str) -> LlmMemoryJudgement | None:
        try:
            data = json.loads(_extract_json(raw))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        observations_raw = data.get("observations") or []
        observations: list[ObservationCandidate] = []
        if isinstance(observations_raw, list):
            for item in observations_raw:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip()
                value = str(item.get("value") or "").strip()
                if not kind or not value:
                    continue
                confidence = _safe_float(item.get("confidence"), default=0.6)
                observations.append(
                    ObservationCandidate(
                        kind=kind,
                        value=value,
                        source="llm_judge",
                        confidence=confidence,
                        evidence_text=str(item.get("evidence_text") or ""),
                        scope_hint=str(data.get("target_scope") or "session"),
                        write_intent=str(data.get("write_intent") or "observe"),
                        product_model=item.get("product_model"),
                        pii_level=str(item.get("pii_level") or _pii_level_for(kind)),
                        source_priority=60,
                    )
                )
        return LlmMemoryJudgement(
            should_write=bool(data.get("should_write")),
            write_intent=str(data.get("write_intent") or "observe"),
            target_scope=str(data.get("target_scope") or "none"),
            confidence=_safe_float(data.get("confidence"), default=0.0),
            reason=str(data.get("reason") or ""),
            observations=observations,
            resolved_references={
                str(k): str(v)
                for k, v in (data.get("resolved_references") or {}).items()
            }
            if isinstance(data.get("resolved_references"), dict)
            else {},
        )

    def _call(self, prompt: str) -> str:
        if self._call_model is not None:
            return self._call_model(prompt)
        return chat_text(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=getattr(settings, "memory_v3_llm_judge_max_tokens", 512),
            timeout=getattr(settings, "memory_v3_llm_judge_timeout", 20),
        )

    @staticmethod
    def _build_prompt(
        packet: TurnEvidencePacket,
        raw_evidence: list[RawEvidence],
        current_session_memory: dict[str, Any],
    ) -> str:
        raw_items = [
            {
                "source": item.source,
                "evidence_type": item.evidence_type,
                "text": item.text,
                "structured": item.structured,
                "confidence": item.confidence,
            }
            for item in raw_evidence[:12]
        ]
        payload = {
            "question": packet.question,
            "answer": packet.answer,
            "recent_history": packet.recent_history,
            "branch_name": packet.branch_name,
            "raw_evidence": raw_items,
            "current_session_memory": current_session_memory,
        }
        return (
            "你是客服记忆系统的候选记忆判断器。只输出 JSON，不要解释。\n"
            "你不能直接写库；你只能提出候选 observations。\n"
            "禁止把 RAG 手册知识、产品说明、模型推测保存为用户记忆。\n"
            "输出字段: should_write, write_intent, target_scope, confidence, reason, "
            "resolved_references, observations。\n"
            f"输入:\n{json.dumps(payload, ensure_ascii=False)}"
        )


def _extract_json(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return match.group(0) if match else text


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pii_level_for(kind: str) -> str:
    if kind in {"phone", "address"}:
        return "high"
    if kind in {"order_id", "case_id"}:
        return "low"
    return "none"
