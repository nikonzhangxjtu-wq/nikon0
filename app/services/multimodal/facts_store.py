"""手册图片结构化事实读取。

在线 attached-only 模式只按文本 chunk 中的 image_id 取图片事实，不做图片向量召回。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ManualImageFact:
    """一张图片可拼入 prompt 的结构化事实。"""

    image_id: str
    image_type: str = ""
    context_intent: str = ""
    parent_context_text: str = ""
    ocr_text: list[str] = field(default_factory=list)
    buttons: list[dict] = field(default_factory=list)
    indicators: list[dict] = field(default_factory=list)
    parts: list[dict] = field(default_factory=list)
    operation_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        parts: list[str] = []
        for label, value in (
            ("类型", self.image_type),
            ("图片意图", self.context_intent),
            ("OCR", " | ".join(self.ocr_text)),
            ("按键", _dict_list_to_text(self.buttons)),
            ("指示/提示", _dict_list_to_text(self.indicators)),
            ("部件", _dict_list_to_text(self.parts)),
            ("操作步骤", " | ".join(self.operation_steps)),
            ("警告/注意", " | ".join(self.warnings)),
            ("关系", " | ".join(self.relations)),
            ("父文本上下文", self.parent_context_text),
        ):
            text = str(value or "").strip()
            if text:
                parts.append(f"{label}: {text}")
        return "\n".join(parts)


class ManualImageFactStore:
    """从 VLM 缓存 JSON 中按 image_id 读取图片结构化事实。"""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self._facts: dict[str, ManualImageFact] | None = None

    def get_many(self, image_ids: list[str]) -> dict[str, ManualImageFact]:
        self._ensure_loaded()
        assert self._facts is not None
        out: dict[str, ManualImageFact] = {}
        for image_id in image_ids:
            fact = self._facts.get(image_id)
            if fact is not None:
                out[image_id] = fact
        return out

    def _ensure_loaded(self) -> None:
        if self._facts is not None:
            return
        self._facts = {}
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        for image_id, item in data.items():
            if not isinstance(item, dict):
                continue
            payload = item.get("understanding", item)
            if not isinstance(payload, dict):
                continue
            fact = _fact_from_payload(str(image_id), payload)
            if fact.image_id:
                self._facts[fact.image_id] = fact


def _fact_from_payload(image_id: str, payload: dict) -> ManualImageFact:
    return ManualImageFact(
        image_id=str(payload.get("image_id") or image_id),
        image_type=str(payload.get("image_type") or ""),
        context_intent=str(payload.get("context_intent") or ""),
        parent_context_text=str(payload.get("parent_context_text") or ""),
        ocr_text=_str_list(payload.get("ocr_text")),
        buttons=_dict_list(payload.get("buttons")),
        indicators=_dict_list(payload.get("indicators")),
        parts=_dict_list(payload.get("parts")),
        operation_steps=_str_list(payload.get("operation_steps")),
        warnings=_str_list(payload.get("warnings")),
        relations=_str_list(payload.get("relations")),
    )


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_list_to_text(items: list[dict]) -> str:
    chunks: list[str] = []
    for item in items:
        pairs = []
        for key, value in item.items():
            value_text = str(value).strip()
            if value_text:
                pairs.append(f"{key}={value_text}")
        if pairs:
            chunks.append("，".join(pairs))
    return " | ".join(chunks)
