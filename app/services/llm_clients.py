"""轻量 LLM 调用工具。

用于路由、手册名识别、图片结构理解这类“小模型/工具模型”场景。
有百炼 API key 时优先走 OpenAI-compatible API；否则回退本地 Ollama。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from app.core.config import settings


def chat_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 30,
) -> str:
    """文本 chat：百炼优先，Ollama 兜底。"""
    if settings.bailian_api_key:
        return _chat_bailian(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return _chat_ollama(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_with_image(
    *,
    model: str,
    prompt: str,
    image_path: str | Path,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    """单图 chat：百炼优先，Ollama 兜底。"""
    path = Path(image_path)
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    if settings.bailian_api_key:
        ext = path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        data_uri = f"data:image/{mime};base64,{image_b64}"
        return _chat_bailian(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return _chat_ollama(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_with_image_inputs(
    *,
    model: str,
    prompt: str,
    images: list[str],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    """多图 chat，images 支持 base64 或 data URI。"""
    clean_images = [_strip_data_uri(img) for img in images if (img or "").strip()]
    if not clean_images:
        return ""
    if settings.bailian_api_key:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for raw in clean_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{raw}"},
                }
            )
        return _chat_bailian(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return _chat_ollama(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": clean_images}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _strip_data_uri(value: str) -> str:
    raw = (value or "").strip()
    if raw.startswith("data:image/") and "," in raw:
        return raw.split(",", 1)[1]
    return raw


def _chat_bailian(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    resp = requests.post(
        f"{settings.bailian_base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.bailian_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _chat_ollama(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    resp = requests.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return str((resp.json() or {}).get("message", {}).get("content") or "")
