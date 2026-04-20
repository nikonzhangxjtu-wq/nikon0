"""手册 stem → 建库/检索时用中文还是英文向量模型。

英文产品 stem 与白名单对齐（``scripts/english_manual_naming.py``），比
``str.isascii()`` 可靠：避免「纯 ASCII 的中文拼音文件名」误判为英文。
"""

from __future__ import annotations

_ENGLISH_PRODUCT_STEMS: frozenset[str] | None = None


def english_product_stems() -> frozenset[str]:
    """拆分后的英文手册文件名 stem 集合（如 ``Coffee_Machine``、``Boat``）。"""
    global _ENGLISH_PRODUCT_STEMS
    if _ENGLISH_PRODUCT_STEMS is None:
        from scripts.english_manual_naming import MANUAL_PREFIX_TO_PRODUCT_STEM

        _ENGLISH_PRODUCT_STEMS = frozenset(MANUAL_PREFIX_TO_PRODUCT_STEM.values())
    return _ENGLISH_PRODUCT_STEMS


def manual_stem_uses_english_embedding(stem: str) -> bool:
    """若该 stem 对应英文拆分手册，则用 ``EMBED_MODEL_EN``。"""
    return stem in english_product_stems()


def query_prefers_chinese_embedding(query: str) -> bool:
    """检索 query 向量：含至少一个 CJK 字则偏向中文嵌入，否则英文。"""
    return any("\u4e00" <= c <= "\u9fff" for c in query)


def generation_reply_language_rule(question: str) -> str:
    """供生成 prompt 使用：按用户问题是否含中文，约束回答语言与中英一致的语气。"""
    q = question.strip()
    if not q:
        return "语言：若用户使用中文提问请用中文作答，若用户使用英文提问请用英文作答。"
    if query_prefers_chinese_embedding(q):
        return "语言要求：用户问题主要为中文，请全程使用中文回答，语气温和、专业。"
    return (
        "Language: The user's question is in English; write your entire reply in English, "
        "clearly and professionally."
    )
