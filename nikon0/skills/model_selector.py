"""LLM-backed skill selector."""

from __future__ import annotations

import json
import asyncio
from typing import Protocol

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import SkillManifest, SkillSelection
from nikon0.skills.base import ManifestDrivenSkillSelector
from nikon0.skills.skill_router_prompt import ROUTING_EXCLUDED_SKILLS, SKILL_ROUTER_SYSTEM


class SkillSelectionModelClient(Protocol):
    async def complete(self, prompt: str) -> str:
        ...


class BailianOllamaSkillSelectionClient:
    """Uses the project's existing 百炼-first / Ollama-fallback LLM client."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 20,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete_sync, prompt)

    def _complete_sync(self, prompt: str) -> str:
        from app.services.llm_clients import chat_text

        return chat_text(
            model=self.model,
            messages=[
                {"role": "system", "content": SKILL_ROUTER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


class LlmSkillSelector(ManifestDrivenSkillSelector):
    """Model-driven selector with strict structured-output parsing."""

    def __init__(self, client: SkillSelectionModelClient, *, min_confidence: float = 0.55) -> None:
        self.client = client
        self.min_confidence = min(1.0, max(0.0, min_confidence))

    async def select(self, context: AgentContext, manifests: tuple[SkillManifest, ...]) -> SkillSelection:
        routable = self._routable_manifests(manifests)
        prompt = self._build_prompt(context, routable)
        try:
            raw = await self.client.complete(prompt)
            payload = self._parse_payload(raw)
        except Exception as exc:  # noqa: BLE001
            return SkillSelection(source="none", reason=f"model selector failed: {type(exc).__name__}: {exc}")

        selected = str(payload.get("selected_skill") or "").strip() or None
        confidence = self._coerce_confidence(payload.get("confidence"))
        reason = str(payload.get("reason") or "model selected skill")
        known = {manifest.name for manifest in routable}
        if selected not in known or confidence < self.min_confidence:
            return self.build_selection(
                selected_skill=None,
                reason=(
                    reason if selected in known else f"model selected unknown skill: {selected}"
                ),
                confidence=confidence,
                manifests=routable,
            )
        return self.build_selection(
            selected_skill=selected,
            reason=reason,
            confidence=confidence,
            manifests=routable,
        )

    @staticmethod
    def _routable_manifests(manifests: tuple[SkillManifest, ...]) -> tuple[SkillManifest, ...]:
        return tuple(item for item in manifests if item.name not in ROUTING_EXCLUDED_SKILLS)

    @staticmethod
    def _build_prompt(context: AgentContext, manifests: tuple[SkillManifest, ...]) -> str:
        skill_lines: list[str] = ["可用 Skill 清单（selected_skill 必须从中选取）："]
        for item in manifests:
            skill_lines.append(f"- {item.name}: {item.description}")
            if item.capabilities:
                skill_lines.append(f"  能力标签: {', '.join(item.capabilities)}")
            if item.required_tools:
                skill_lines.append(f"  依赖工具: {', '.join(item.required_tools)}")
            skill_lines.append(f"  风险等级: {item.risk_level}")

        sections = [
            "\n".join(skill_lines),
            f"用户当前问题：\n{context.request.message.strip()}",
        ]
        transcript = context.transcript_context[-2000:].strip()
        if transcript:
            sections.append(f"最近对话上下文：\n{transcript}")
        if context.session_state and context.session_state.flat_state:
            sections.append(
                "会话状态：\n"
                + json.dumps(context.session_state.flat_state, ensure_ascii=False)
            )
        sections.append("请根据分类规则输出一行 JSON。")
        return "\n\n".join(sections)

    @staticmethod
    def _parse_payload(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        if "{" in text and "}" in text:
            start, end = text.find("{"), text.rfind("}")
            text = text[start : end + 1]
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("selector output must be a JSON object")
        return data

    @staticmethod
    def _coerce_confidence(value: object) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
