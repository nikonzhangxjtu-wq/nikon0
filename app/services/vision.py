"""用户上传图片的轻量视觉理解。

目标：把 Base64 图片转成 1~3 句图像摘要，供路由/检索/生成使用。
若模型或环境不支持视觉输入，会安全降级为返回空摘要。
"""

from __future__ import annotations

import json
import re

from app.core.config import settings


def _ensure_data_uri(image_b64_or_uri: str) -> str:
    s = (image_b64_or_uri or "").strip()
    if not s:
        return ""
    if s.startswith("data:image/"):
        return s
    # 默认按 JPEG 处理；即使实际格式不同，模型通常也会给出错误并触发降级。
    return f"data:image/jpeg;base64,{s}"


def _extract_summary(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
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
            summary = obj.get("summary")
            if isinstance(summary, str):
                return summary.strip()
        except json.JSONDecodeError:
            pass

    # 兜底：移除多余空白，截取首段文本
    text = re.sub(r"\s+", " ", text).strip()
    return text


class VisionInterpreter:
    """调用本地视觉模型，把用户图片转成简短文本摘要。"""

    def summarize_images(self, question: str, images: list[str]) -> str:
        if not settings.vision_enabled:
            return ""
        if not images:
            return ""

        try:
            from langchain_core.messages import HumanMessage
            from langchain_ollama import ChatOllama
        except ModuleNotFoundError as exc:
            print(f"[WARN] 视觉摘要跳过：缺少依赖 ({exc})")
            return ""

        model = (settings.vision_model or "").strip() or settings.gen_model
        client = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )

        system = (
            "你是客服系统中的视觉理解模块。请只根据用户上传图片和问题，提取对客服问答有帮助的"
            "可观察信息（部件、指示灯状态、故障提示、型号/标签、界面文字等）。"
            "不要编造图片中看不到的细节。"
            "只输出一行 JSON："
            '{"summary":"1~3句中文摘要，尽量包含可用于检索的关键词"}'
        )
        user_text = f"用户问题：{question.strip()}\n请输出图片摘要。"

        content: list[dict] = [{"type": "text", "text": user_text}]
        max_images = max(1, settings.vision_max_images)
        for img in images[:max_images]:
            uri = _ensure_data_uri(img)
            if not uri:
                continue
            # ChatOllama 兼容 OpenAI 风格的 image_url 内容块。
            content.append({"type": "image_url", "image_url": {"url": uri}})

        if len(content) <= 1:
            return ""

        try:
            msg = client.invoke([("system", system), HumanMessage(content=content)])
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 视觉摘要失败，已降级为无图摘要: {exc}")
            return ""

        raw = getattr(msg, "content", "") or ""
        summary = _extract_summary(raw)
        if not summary:
            return ""
        max_chars = max(80, settings.vision_summary_max_chars)
        return summary[:max_chars]

