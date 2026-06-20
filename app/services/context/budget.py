"""token 预算与文本裁剪工具。"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算。

    这里刻意不引入 tokenizer：压缩器需要稳定、低延迟、容易单测。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    ascii_chars = len(text) - cjk
    return max(1, int(cjk * 0.75 + ascii_chars / 4) + 1)


def token_budget_to_chars(max_tokens: int) -> int:
    # 中文客服场景偏多，1 token 约 1.5 字符是偏保守的换算。
    return max(120, int(max_tokens * 1.5))


def truncate_at_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    for sep in ("\n\n", "\n", "。", "！", "？", ". ", "; ", "，", ", "):
        pos = chunk.rfind(sep)
        if pos > max_chars * 0.6:
            return chunk[: pos + len(sep)].rstrip() + "\n…"
    return chunk.rstrip() + "…"


def keywords(question: str) -> set[str]:
    """从问题中抽轻量关键词，供证据块相关性和自检使用。"""
    q = question or ""
    kws = {w.lower() for w in _WORD_RE.findall(q) if len(w) >= 3}
    cjk = "".join(_CJK_RE.findall(q))
    for n in (4, 3, 2):
        for i in range(0, max(0, len(cjk) - n + 1)):
            kws.add(cjk[i : i + n])
    return {k for k in kws if k.strip()}


def split_sentences(text: str) -> list[str]:
    return [
        p.strip()
        for p in re.split(r"(?<=[。！？!?])\s*|\n+", text or "")
        if p.strip()
    ]
