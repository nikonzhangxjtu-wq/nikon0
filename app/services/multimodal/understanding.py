"""手册图片结构理解与缓存。"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.llm_clients import chat_with_image
from app.services.multimodal.catalog import file_sha256
from app.services.multimodal.types import ManualImageUnderstanding

# VLM 返回的 JSON 不一定严格，这两函数做容错归一化。
def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,，、;；\n]+", value) if v.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _coerce_dict_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    result: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            result.append({str(k): v for k, v in item.items()})
        elif item:
            result.append({"text": str(item)})
    return result

# 输入为image_id + vlm 原始字符串 raw，输出为manualimageunderstanding
def parse_understanding_json(image_id: str, raw: str) -> ManualImageUnderstanding:
    """从 VLM 输出中提取固定 JSON；失败时保留 raw_text 降级。"""
    text = (raw or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                text = part
                break
    start, end = text.find("{"), text.rfind("}")
    obj: dict[str, Any] = {}
    if start >= 0 and end > start:
        try:
            loaded = json.loads(text[start : end + 1])
            if isinstance(loaded, dict):
                obj = loaded
        except json.JSONDecodeError:
            obj = {}

    if not obj:
        return ManualImageUnderstanding(image_id=image_id, raw_text=raw)

    return ManualImageUnderstanding(
        image_id=image_id,
        image_type=str(obj.get("image_type") or "other").strip() or "other",
        parent_context_text=str(obj.get("parent_context_text") or "").strip(),
        context_intent=str(obj.get("context_intent") or "").strip(),
        ocr_text=_coerce_str_list(obj.get("ocr_text")),
        buttons=_coerce_dict_list(obj.get("buttons")),
        indicators=_coerce_dict_list(obj.get("indicators")),
        parts=_coerce_dict_list(obj.get("parts")),
        operation_steps=_coerce_str_list(obj.get("operation_steps")),
        warnings=_coerce_str_list(obj.get("warnings")),
        relations=_coerce_str_list(obj.get("relations")),
        raw_text=raw,
    )

# 缓存文件默认是 manual_image_understanding_cache.json，结构大致为
"""
{
  "Blower_02": {
    "image_path": ".../Blower_02.png",
    "file_hash": "f6e7b8bd...",
    "vision_model": "qwen3-vl-flash",
    "parent_context_hash": "e55f4352...",
    "parent_context_text": "…父 chunk 正文…",
    "understanding": { "image_id": "Blower_02", "ocr_text": [...], ... }
  }
}
"""
class ManualImageUnderstandingCache:
    """按图片 hash 缓存 VLM 结构理解结果。"""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self._items: dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._items = data
            except json.JSONDecodeError:
                self._items = {}

    def get(
        self,
        *,
        image_id: str,
        image_path: str | Path,
        parent_context_text: str = "",
    ) -> ManualImageUnderstanding | None:
        path = Path(image_path)
        cached = self._items.get(image_id)
        if not cached:
            return None
        if cached.get("file_hash") != file_sha256(path):
            return None
        if cached.get("vision_model") != (settings.manual_image_vlm_model or settings.vlm_model):
            return None
        if cached.get("parent_context_hash", "") != _context_hash(parent_context_text):
            return None
        payload = cached.get("understanding")
        if not isinstance(payload, dict):
            return None
        return ManualImageUnderstanding(**payload)
    # 把上述元数据 + asdict(understanding) 写入内存 dict，再整文件重写 JSON（indent=2）。简单直接，适合建索引这种批处理场景。
    def set(
        self,
        *,
        image_id: str,
        image_path: str | Path,
        parent_context_text: str = "",
        understanding: ManualImageUnderstanding,
    ) -> None:
        path = Path(image_path)
        self._items[image_id] = {
            "image_path": str(path),
            "file_hash": file_sha256(path),
            "vision_model": settings.manual_image_vlm_model or settings.vlm_model,
            "parent_context_hash": _context_hash(parent_context_text),
            "parent_context_text": parent_context_text,
            "understanding": asdict(understanding),
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

# 真正掉vlm的类
class ManualImageInterpreter:
    """调用 Ollama VLM，把手册插图抽取为结构化证据。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        session: object | None = None,
    ) -> None:
        self.model = (
            model
            or settings.manual_image_vlm_model
            or settings.vlm_model
            or settings.vision_model
        ).strip()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.session = session

    def understand_image(
        self,
        *,
        image_id: str,
        image_path: str | Path,
        parent_context_text: str = "",
    ) -> ManualImageUnderstanding:
        if not self.model:
            raise RuntimeError("缺少手册图片 VLM 模型配置：请设置 MANUAL_IMAGE_VLM_MODEL 或 VISION_MODEL")

        path = Path(image_path)
        prompt = self._prompt(image_id, parent_context_text=parent_context_text)
        try:
            raw = chat_with_image(
                model=self.model,
                prompt=prompt,
                image_path=path,
                temperature=0.0,
                max_tokens=1024,
                timeout=120,
            )
        except Exception as exc:
            raise RuntimeError(f"手册图片 VLM 理解失败 image_id={image_id}: {exc}") from exc
        return parse_understanding_json(image_id, raw)

    @staticmethod
    def _prompt(image_id: str, *, parent_context_text: str = "") -> str:
        # 这段 prompt 是结构抽取，不是 caption：caption 很难支撑按键/部件/步骤级检索。
        context_block = (
            f"\n图片在手册中的父文本上下文如下。请优先结合上下文判断图片表达的产品知识意图，"
            f"但不要编造上下文和图片都没有的信息：\n{parent_context_text[:1800]}\n"
            if parent_context_text.strip()
            else "\n未提供父文本上下文，请只根据图片可见内容抽取结构事实。\n"
        )
        return f"""你是多模态客服系统的手册图片结构理解模块。
图片 ID：{image_id}
{context_block}

请先判断图片类型，只能选择：
button_panel, product_prompt, operation_diagram, structure_diagram, error_indicator, other

不同类型重点抽取：
- button_panel：按键名称、图标、位置、功能、长按/短按/组合键。
- product_prompt/error_indicator：提示文字、故障码、指示灯颜色/闪烁状态、警告符号和含义。
- operation_diagram：操作对象、箭头方向、动作顺序、安装/拆卸/清洁步骤、注意事项。
- structure_diagram：部件名称、编号、相对位置、连接/拆装关系。

只输出一行 JSON，不要输出解释文字：
{{"context_intent":"结合父文本判断出的图片用途，例如说明滤网拆卸与清洁","parent_context_text":"可简短摘录与图片最相关的父文本","image_type":"button_panel","ocr_text":["图中可见文字"],"buttons":[{{"name":"按键名","position":"位置","function":"功能","operation":"短按/长按/组合键"}}],"indicators":[{{"name":"指示灯/故障码","state":"颜色/闪烁/常亮","meaning":"含义"}}],"parts":[{{"name":"部件名","number":"编号","position":"位置"}}],"operation_steps":["可见或由父文本明确说明的操作步骤"],"warnings":["警告或注意事项"],"relations":["部件/按键之间的位置或连接关系"]}}"""


def _context_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
