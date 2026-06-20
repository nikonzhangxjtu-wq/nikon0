"""用户上传图片的轻量视觉理解。

目标：把 Base64 图片转成 1~3 句图像摘要，供路由/检索/生成使用。
若模型或环境不支持视觉输入，会安全降级为返回空摘要。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.services.llm_clients import chat_with_image_inputs


@dataclass(frozen=True)
class VisualSummary:
    summary: str = ""
    ocr_text: str = ""
    key_entities: list[str] = field(default_factory=list)
    product_type: str = ""

    def to_context_text(self) -> str:
        parts: list[str] = []
        if self.ocr_text:
            parts.append(f"OCR文字：{self.ocr_text}")
        if self.key_entities:
            parts.append(f"关键实体：{', '.join(self.key_entities)}")
        if self.product_type:
            parts.append(f"产品类型：{self.product_type}")
        if self.summary:
            parts.append(f"图片摘要：{self.summary}")
        return "\n".join(parts)


def _ensure_data_uri(image_b64_or_uri: str) -> str:
    s = (image_b64_or_uri or "").strip()
    if not s:
        return ""
    if s.startswith("data:image/"):
        return s
    # 默认按 JPEG 处理；即使实际格式不同，模型通常也会给出错误并触发降级。
    return f"data:image/jpeg;base64,{s}"


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[,，、;；\s]+", value) if v.strip()]
    return []


def _extract_visual_summary(raw: str) -> VisualSummary:
    text = (raw or "").strip()
    if not text:
        return VisualSummary()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return VisualSummary(
                    summary=str(obj.get("summary") or "").strip(),
                    ocr_text=str(obj.get("ocr_text") or "").strip(),
                    key_entities=_coerce_str_list(obj.get("key_entities")),
                    product_type=str(obj.get("product_type") or "").strip(),
                )
        except json.JSONDecodeError:
            pass

    # 兜底：移除多余空白，截取首段文本
    text = re.sub(r"\s+", " ", text).strip()
    return VisualSummary(summary=text)


class VisionInterpreter:
    """调用本地视觉模型，把用户图片转成简短文本摘要。"""

    def summarize_images(self, question: str, images: list[str]) -> str:
        if not settings.vision_enabled:
            return ""
        if not images:
            return ""

        model = (settings.vision_model or "").strip() or settings.gen_model
        system = (
            "你是客服系统中的视觉理解模块。请只根据用户上传图片和问题，提取对客服问答有帮助的"
            "可观察信息（部件、指示灯状态、故障提示、型号/标签、界面文字等）。"
            "不要编造图片中看不到的细节。"
            "只输出一行 JSON："
            '{"summary":"1~3句中文摘要","ocr_text":"图片中可见文字，没有则空字符串",'
            '"key_entities":["型号/故障码/部件/品牌等关键词"],"product_type":"产品类型，没有则空字符串"}'
        )
        user_text = f"用户问题：{question.strip()}\n请输出图片摘要。"

        max_images = max(1, settings.vision_max_images)
        image_payloads: list[str] = []
        for img in images[:max_images]:
            uri = _ensure_data_uri(img)
            if not uri:
                continue
            image_payloads.append(uri)

        if not image_payloads:
            return ""

        try:
            raw = chat_with_image_inputs(
                model=model,
                prompt=f"{system}\n\n{user_text}",
                images=image_payloads,
                temperature=0.0,
                max_tokens=512,
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 视觉摘要失败，已降级为无图摘要: {exc}")
            return ""

        summary = _extract_visual_summary(raw)
        text = summary.to_context_text()
        if not text:
            return ""
        max_chars = max(80, settings.vision_summary_max_chars)
        return text[:max_chars]
