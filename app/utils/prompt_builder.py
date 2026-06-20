"""Prompt 拼装工具。

核心原则：生成模型应主要依据结构化上下文回答，减少幻觉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.retriever import RetrievedChunk
from app.utils.prompts import PromptContext, compose_generation_prompt

PIC_MARKER = "<PIC>"
_IMG_REF_PATTERN = re.compile(r"<(IMG_\d+)(?::([^>]+))?>")


@dataclass(frozen=True)
class PromptImageRef:
    """Prompt 中可供模型引用的一张图片。"""

    token: str
    image_id: str
    chunk_id: str
    local_index: int


@dataclass(frozen=True)
class MultimodalContextBlock:
    """带图片引用映射的上下文块。"""

    context_block: str
    image_ref_map: dict[str, str] = field(default_factory=dict)
    image_refs: list[PromptImageRef] = field(default_factory=list)


MAX_CONTEXT_CHARS = 3000  # 防止 RAG 上下文过长导致模型推理引擎 abort


def _truncate_text(text: str, max_chars: int) -> str:
    """保留开头 max_chars 字符，在完整句/词边界截断。"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ("。", "，", ". ", ", ", " ", "\n"):
        last = truncated.rfind(sep)
        if last > max_chars * 0.6:
            return truncated[: last + len(sep)] + "…"
    return truncated + "…"


def _enforce_context_limit(raw: str) -> str:
    """若上下文超过 MAX_CONTEXT_CHARS，在句边界处截断。"""
    if len(raw) <= MAX_CONTEXT_CHARS:
        return raw
    return _truncate_text(raw, MAX_CONTEXT_CHARS)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """将检索结果压成一段便于塞进 prompt 的上下文字符串。"""

    if not chunks:
        return "（无检索上下文）"

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[片段 {idx}]")
        lines.append(f"chunk_id: {chunk.chunk_id}")
        lines.append(f"手册: {chunk.manual_name}")
        lines.append(f"分数: {chunk.score}")
        lines.append(f"正文: {chunk.text}")
        if chunk.image_ids:
            lines.append(f"图片ID: {', '.join(chunk.image_ids)}")
        lines.append("")
    return _enforce_context_limit("\n".join(lines).strip())


def build_multimodal_context_block(chunks: list[RetrievedChunk]) -> MultimodalContextBlock:
    """将检索结果压成上下文。文本中已含 `<IMG:image_id>` 引用，无需再做映射。"""
    if not chunks:
        return MultimodalContextBlock(context_block="（无检索上下文）")

    image_refs: list[PromptImageRef] = []
    image_ref_map: dict[str, str] = {}
    next_image_idx = 1
    lines: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        # 从已嵌入 `<IMG:xxx>` 的文本中提取图片引用
        chunk_img_ids: list[str] = []
        for m in re.finditer(r"<IMG:([^>]+)>", chunk.text):
            chunk_img_ids.append(m.group(1))

        lines.append(f"[片段 {idx}]")
        lines.append(f"chunk_id: {chunk.chunk_id}")
        lines.append(f"手册: {chunk.manual_name}")
        lines.append(f"分数: {chunk.score}")
        lines.append(f"正文: {chunk.text}")

        # 构建可引用图片列表（用于 prompt 提示和 answer 校验）
        if chunk_img_ids:
            chunk_refs: list[str] = []
            for local_i, img_id in enumerate(chunk_img_ids, start=1):
                token = f"IMG_{next_image_idx}"
                next_image_idx += 1
                image_ref_map[token] = img_id
                image_refs.append(
                    PromptImageRef(
                        token=token,
                        image_id=img_id,
                        chunk_id=chunk.chunk_id,
                        local_index=local_i,
                    )
                )
                chunk_refs.append(f"<{token}:{img_id}>")
            lines.append(f"可引用图片: {', '.join(chunk_refs)}")
        if getattr(chunk, "image_evidence", None):
            lines.append("[图片结构证据]")
            for evidence in chunk.image_evidence:
                token = f"IMG_{next_image_idx}"
                next_image_idx += 1
                image_ref_map[token] = evidence.image_id
                image_refs.append(
                    PromptImageRef(
                        token=token,
                        image_id=evidence.image_id,
                        chunk_id=chunk.chunk_id,
                        local_index=len(image_refs) + 1,
                    )
                )
                lines.append(
                    f"<{token}:{evidence.image_id}> "
                    f"score={evidence.score:.4f} reason={evidence.match_reason}"
                )
                if evidence.prompt_text:
                    lines.append(evidence.prompt_text)
        lines.append("")

    return MultimodalContextBlock(
        context_block=_enforce_context_limit("\n".join(lines).strip()),
        image_ref_map=image_ref_map,
        image_refs=image_refs,
    )


def finalize_answer_images(answer: str, image_ref_map: dict[str, str]) -> tuple[str, list[str]]:
    """把答案中的图片引用转成 `<PIC>`，按出现顺序返回图片 ID。

    兼容两种格式：
    - ``<IMG_n:image_id>`` — 模型通过 prompt 中「可引用图片」行引用（经 image_ref_map 校验）
    - ``<IMG:image_id>`` — 模型直接从上下文段落的正文中复制引用

    单次扫描，按文本出现顺序收集。
    """
    images: list[str] = []
    seen: set[str] = set()
    pic_placeholder = "__PIC__"
    # 统一正则：先匹配 <IMG_n:image_id>，再匹配 <IMG:image_id>
    _UNIFIED_IMG = re.compile(r"<IMG_(\d+)(?::([^>]+))?>|<IMG:([^>]+)>")

    def replace_unified(match: re.Match[str]) -> str:
        nonlocal images, seen
        token = match.group(1)      # <IMG_n:...> 的数字部分
        declared_id = match.group(2)  # <IMG_n:image_id> 的 image_id
        direct_id = match.group(3)   # <IMG:image_id> 的 image_id

        if token is not None:
            # 映射引用
            actual_id = image_ref_map.get(f"IMG_{token}")
            if not actual_id:
                return ""
            if declared_id and declared_id != actual_id:
                return ""
            img_id = actual_id
        else:
            # 直接引用
            img_id = direct_id
            if not img_id:
                return ""

        if img_id not in seen:
            seen.add(img_id)
            images.append(img_id)
        return pic_placeholder

    normalized = _UNIFIED_IMG.sub(replace_unified, answer)
    # 删除模型可能直接输出的裸 <PIC>
    normalized = normalized.replace(PIC_MARKER, "")
    normalized = normalized.replace(pic_placeholder, PIC_MARKER)
    return normalized.strip(), images


def build_generation_prompt(
    question: str,
    context_block: str,
    domain_hint: str,
    need_rag: bool = True,
    route_reason: str = "",
) -> str:
    """为 qwen2 构造生成 prompt；实现委托给 `app.utils.prompts` 注册表。"""
    return compose_generation_prompt(
        PromptContext(
            question=question,
            need_rag=need_rag,
            domain_hint=domain_hint,
            context_block=context_block,
            route_reason=route_reason,
        )
    )
