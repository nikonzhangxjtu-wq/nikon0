"""轻量答案后处理。

用于比赛提交前的稳定性修正：去模板、去重复、控长度、移除非法图片标记。
"""

from __future__ import annotations

import re


_TEMPLATE_TAILS = (
    "如有任何疑问",
    "如果您还有其他问题",
    "祝您生活愉快",
    "感谢您的理解",
    "很高兴为您服务",
)

_TEMPLATE_PREFIXES = (
    "很高兴为您解答，",
    "您好，",
    "亲，",
)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _deduplicate_sentences(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for sentence in _split_sentences(text):
        key = re.sub(r"\s+", "", sentence)
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    if not out:
        return text.strip()
    sep = "\n" if "\n" in text else ""
    return sep.join(out)


def _truncate_answer(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ("。", "！", "？", "\n", ";", ". "):
        pos = truncated.rfind(sep)
        if pos > max_chars * 0.65:
            return truncated[: pos + len(sep)].strip()
    return truncated.rstrip() + "…"


def postprocess_answer(answer: str) -> str:
    """清理生成答案中的低价值模板和格式噪声。"""
    text = (answer or "").strip()
    if not text:
        return text

    # 删除模型直接生成或残留的非法 IMG 标记；合法图片已在 finalize_answer_images 转为 <PIC>。
    text = re.sub(r"<IMG(?:_\d+)?(?::[^>]+)?>", "", text)

    for prefix in _TEMPLATE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip()

    for tail in _TEMPLATE_TAILS:
        idx = text.find(tail)
        if idx > 0:
            text = text[:idx].rstrip(" ，,。")
            break

    text = _deduplicate_sentences(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return _truncate_answer(text)
