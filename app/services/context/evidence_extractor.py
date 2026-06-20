"""结构感知 RAG 证据块抽取。

RAG 压缩的关键是不要切碎步骤、表格和问答。这里的最小单位是
EvidenceBlock，而不是单句。
"""

from __future__ import annotations

import re

from app.services.context.budget import estimate_tokens, keywords, split_sentences, token_budget_to_chars, truncate_at_boundary
from app.services.context.types import CriticalFacts, EvidenceBlock

_CHUNK_START_RE = re.compile(r"(?m)^\[片段\s+\d+\]")
_STEP_LINE_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十\d]+步|步骤\s*\d+|Step\s*\d+|STEP\s*\d+|\d+[.)、]|[（(]\d+[）)])"
)
_STEP_WORD_RE = re.compile(r"步骤|首先|然后|接着|随后|最后|安装|拆卸|清洁|清洗|更换|启动|关闭|设置|排查|操作")
_TABLE_LINE_RE = re.compile(r"^\s*[^：:\n]{1,28}[：:]\s*[^：:\n]{1,120}$")
_QA_RE = re.compile(r"(?:^|\n)\s*(?:问|Q|问题)[:：].+?(?:\n|$).*(?:答|A|解决|处理)[:：]", re.IGNORECASE | re.S)
_WARNING_RE = re.compile(r"注意|警告|危险|小心|提示|Note|Warning|Caution", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]+[、.．]|#+\s*|[A-Z][A-Za-z ]{2,40}:?$|.{2,32}[：:])\s*$")


def parse_rag_context(context_block: str, question: str, facts: CriticalFacts) -> list[EvidenceBlock]:
    """解析当前 prompt context 中的 [片段] 块。"""
    text = (context_block or "").strip()
    if not text:
        return []

    starts = list(_CHUNK_START_RE.finditer(text))
    if not starts:
        return [_block_from_fields("", "", text, question, facts, 0.0, [])]

    chunks: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        chunks.append(text[start.start() : end].strip())

    blocks: list[EvidenceBlock] = []
    for chunk in chunks:
        blocks.append(_parse_chunk(chunk, question, facts))
    return blocks


def compress_evidence_blocks(
    *,
    context_block: str,
    question: str,
    facts: CriticalFacts,
    max_tokens: int,
) -> tuple[str, list[EvidenceBlock], list[str]]:
    blocks = parse_rag_context(context_block, question, facts)
    if not blocks:
        return "", [], []

    notes: list[str] = []
    compressed: list[EvidenceBlock] = []
    per_block_budget = max(280, int(max_tokens / max(1, min(len(blocks), 4))))
    for block in blocks:
        new_block = _compress_block(block, question, facts, per_block_budget)
        compressed.append(new_block)
        if new_block.fallback_used:
            notes.append(f"rag_fallback:{new_block.chunk_id or new_block.block_type}")

    ranked = sorted(compressed, key=lambda b: _block_rank(b, question, facts), reverse=True)
    selected: list[EvidenceBlock] = []
    used_tokens = 0
    for block in ranked:
        block_tokens = estimate_tokens(_render_block(block))
        if selected and used_tokens + block_tokens > max_tokens:
            # 裁剪顺序：先丢低相关 evidence block，不从块中间切。
            continue
        selected.append(block)
        used_tokens += block_tokens

    if not selected:
        selected = ranked[:1]
        notes.append("rag_budget:keep_top_block")

    selected.sort(key=lambda b: compressed.index(b))
    return "\n\n".join(_render_block(b) for b in selected), selected, notes


def _parse_chunk(chunk: str, question: str, facts: CriticalFacts) -> EvidenceBlock:
    chunk_id = _field_value(chunk, "chunk_id")
    manual_name = _field_value(chunk, "手册")
    score = _float_or_zero(_field_value(chunk, "分数"))
    body = _extract_body(chunk)
    image_refs = re.findall(r"<IMG[^>]*>|<IMG:[^>]+>", chunk)
    return _block_from_fields(chunk_id, manual_name, body, question, facts, score, image_refs, raw_text=chunk)


def _block_from_fields(
    chunk_id: str,
    manual_name: str,
    text: str,
    question: str,
    facts: CriticalFacts,
    score: float,
    image_refs: list[str],
    *,
    raw_text: str = "",
) -> EvidenceBlock:
    block_type = _detect_block_type(text, question)
    title = _detect_title(text)
    reason = _preserved_reason(block_type, question, facts)
    return EvidenceBlock(
        chunk_id=chunk_id,
        manual_name=manual_name,
        block_type=block_type,
        title=title,
        text=text.strip(),
        score=score,
        preserved_reason=reason,
        image_refs=image_refs,
        raw_text=raw_text or text,
    )


def _compress_block(block: EvidenceBlock, question: str, facts: CriticalFacts, max_tokens: int) -> EvidenceBlock:
    if estimate_tokens(block.text) <= max_tokens:
        return block

    if block.block_type == "step_block":
        text = _extract_step_block(block.text)
        # 步骤块自检失败时回退完整 chunk，宁可多 token 也不丢步骤。
        if not _has_valid_step_sequence(text):
            return _fallback(block, "step_verifier_failed")
        if estimate_tokens(text) <= max_tokens:
            return _replace_text(block, text)
        shortened = _shorten_step_descriptions(text, max_tokens)
        if _has_valid_step_sequence(shortened):
            return _replace_text(block, shortened)
        return _fallback(block, "step_budget_fallback")

    if block.block_type == "table_like":
        text = _extract_table_rows(block.text, question, facts)
        if not _table_has_query_hit(text, question, facts):
            return _fallback(block, "table_verifier_failed")
        return _replace_text(block, text if estimate_tokens(text) <= max_tokens else truncate_at_boundary(text, token_budget_to_chars(max_tokens)))

    if block.block_type == "qa_block":
        text = _extract_qa_pair(block.text, question)
        return _replace_text(block, text if estimate_tokens(text) <= max_tokens else truncate_at_boundary(text, token_budget_to_chars(max_tokens)))

    text = _extract_plain_window(block.text, question, facts, max_tokens)
    return _replace_text(block, text)


def _field_value(chunk: str, field: str) -> str:
    m = re.search(rf"(?m)^{re.escape(field)}:\s*(.*)$", chunk)
    return (m.group(1).strip() if m else "")


def _extract_body(chunk: str) -> str:
    lines = chunk.splitlines()
    out: list[str] = []
    in_body = False
    for line in lines:
        if line.startswith("正文:"):
            in_body = True
            out.append(line.split(":", 1)[1].strip())
            continue
        if in_body and re.match(r"^(可引用图片|chunk_id|手册|分数):", line):
            break
        if in_body:
            out.append(line)
    return "\n".join(out).strip() or chunk


def _detect_block_type(text: str, question: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    step_count = sum(1 for ln in lines if _STEP_LINE_RE.search(ln))
    table_count = sum(1 for ln in lines if _TABLE_LINE_RE.search(ln))
    if step_count >= 2 or (_is_step_question(question) and _STEP_WORD_RE.search(text)):
        return "step_block"
    if table_count >= 2 or any(k in question for k in ("故障码", "指示灯", "状态", "含义")) and table_count >= 1:
        return "table_like"
    if _QA_RE.search(text):
        return "qa_block"
    return "plain_block"


def _detect_title(text: str) -> str:
    for line in text.splitlines()[:5]:
        s = line.strip()
        if s and (s.endswith(("：", ":")) or _HEADING_RE.search(s)):
            return s[:80]
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first[:60]


def _is_step_question(question: str) -> bool:
    return any(k in (question or "") for k in ("如何", "怎么", "步骤", "安装", "清洁", "清洗", "更换", "拆卸", "启动", "关闭", "设置", "排查", "操作"))


def _preserved_reason(block_type: str, question: str, facts: CriticalFacts) -> str:
    if block_type == "step_block":
        return "步骤型内容，保留连续步骤与注意事项"
    if block_type == "table_like":
        return "表格/状态类内容，保留表头、命中行和相邻行"
    if block_type == "qa_block":
        return "问答型内容，保留完整问答对"
    if facts.count() > 0:
        return "普通说明，按问题关键词和关键事实抽取窗口"
    return "普通说明，按问题关键词抽取窗口"


def _extract_step_block(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    step_indices = [i for i, ln in enumerate(lines) if _STEP_LINE_RE.search(ln)]
    if not step_indices:
        return text
    start = max(0, step_indices[0] - 1)
    end = step_indices[-1] + 1
    while end < len(lines) and (_WARNING_RE.search(lines[end]) or not _HEADING_RE.search(lines[end])):
        end += 1
        if end - start > 80:
            break
    selected = lines[start:end]
    warnings = [ln for ln in lines if _WARNING_RE.search(ln) and ln not in selected]
    return "\n".join([*selected, *warnings[:4]])


def _has_valid_step_sequence(text: str) -> bool:
    nums: list[int] = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)[.)、]", line)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return bool(_STEP_WORD_RE.search(text))
    if nums[0] != 1:
        return False
    return nums == list(range(nums[0], nums[0] + len(nums)))


def _shorten_step_descriptions(text: str, max_tokens: int) -> str:
    max_chars = token_budget_to_chars(max_tokens)
    out: list[str] = []
    used = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _STEP_LINE_RE.search(s) and len(s) > 80:
            s = truncate_at_boundary(s, 80)
        elif _WARNING_RE.search(s) and len(s) > 120:
            s = truncate_at_boundary(s, 120)
        if used + len(s) + 1 > max_chars and out:
            break
        out.append(s)
        used += len(s) + 1
    return "\n".join(out)


def _extract_table_rows(text: str, question: str, facts: CriticalFacts) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kws = _all_terms(question, facts)
    selected: set[int] = set()
    for idx, line in enumerate(lines):
        if idx == 0 or line.endswith(("：", ":")):
            selected.add(idx)
        if any(term and term.lower() in line.lower() for term in kws):
            selected.update({max(0, idx - 1), idx, min(len(lines) - 1, idx + 1)})
    if len(selected) <= 1:
        for idx, line in enumerate(lines):
            if _TABLE_LINE_RE.search(line):
                selected.update({max(0, idx - 1), idx, min(len(lines) - 1, idx + 1)})
                break
    return "\n".join(lines[i] for i in sorted(selected))


def _table_has_query_hit(text: str, question: str, facts: CriticalFacts) -> bool:
    terms = _all_terms(question, facts)
    return any(term and term.lower() in text.lower() for term in terms) or bool(_TABLE_LINE_RE.search(text))


def _extract_qa_pair(text: str, question: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kws = keywords(question)
    best_idx = 0
    best_score = -1
    for idx, line in enumerate(lines):
        score = sum(1 for kw in kws if kw.lower() in line.lower())
        if score > best_score:
            best_idx, best_score = idx, score
    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 4)
    return "\n".join(lines[start:end])


def _extract_plain_window(text: str, question: str, facts: CriticalFacts, max_tokens: int) -> str:
    units = split_sentences(text)
    if not units:
        return truncate_at_boundary(text, token_budget_to_chars(max_tokens))
    terms = _all_terms(question, facts)
    scored: list[tuple[int, int, str]] = []
    for idx, unit in enumerate(units):
        score = sum(1 for term in terms if term and term.lower() in unit.lower())
        scored.append((score, -idx, unit))
    selected: list[tuple[int, str]] = []
    max_chars = token_budget_to_chars(max_tokens)
    used = 0
    for score, neg_idx, unit in sorted(scored, reverse=True):
        if score <= 0 and selected:
            continue
        idx = -neg_idx
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(units) and all(existing_idx != j for existing_idx, _ in selected):
                candidate = units[j]
                if used + len(candidate) + 1 > max_chars and selected:
                    continue
                selected.append((j, candidate))
                used += len(candidate) + 1
        if used >= max_chars:
            break
    if not selected:
        return truncate_at_boundary(text, max_chars)
    selected.sort(key=lambda x: x[0])
    return "\n".join(unit for _, unit in selected)


def _all_terms(question: str, facts: CriticalFacts) -> set[str]:
    terms = set(keywords(question))
    for values in (
        facts.order_ids,
        facts.phones,
        facts.product_models,
        facts.fault_codes,
        facts.user_goals,
        facts.visual_entities,
    ):
        terms.update(v for v in values if v)
    return terms


def _block_rank(block: EvidenceBlock, question: str, facts: CriticalFacts) -> float:
    terms = _all_terms(question, facts)
    overlap = sum(1 for term in terms if term and term.lower() in block.text.lower())
    type_bonus = {"step_block": 3, "table_like": 2.5, "qa_block": 2, "plain_block": 1}.get(block.block_type, 1)
    return block.score * 2 + overlap + type_bonus


def _render_block(block: EvidenceBlock) -> str:
    rows = [
        f"[证据块 | {block.block_type}]",
        f"chunk_id: {block.chunk_id or '-'}",
        f"手册: {block.manual_name or '-'}",
        f"标题: {block.title or '-'}",
        f"保留原因: {block.preserved_reason}",
        "正文:",
        block.text.strip(),
    ]
    if block.image_refs:
        rows.append("图片引用: " + ", ".join(block.image_refs))
    return "\n".join(rows).strip()


def _replace_text(block: EvidenceBlock, text: str) -> EvidenceBlock:
    return EvidenceBlock(
        chunk_id=block.chunk_id,
        manual_name=block.manual_name,
        block_type=block.block_type,
        title=block.title,
        text=text.strip(),
        score=block.score,
        preserved_reason=block.preserved_reason,
        image_refs=block.image_refs,
        raw_text=block.raw_text,
        fallback_used=False,
    )


def _fallback(block: EvidenceBlock, reason: str) -> EvidenceBlock:
    return EvidenceBlock(
        chunk_id=block.chunk_id,
        manual_name=block.manual_name,
        block_type=block.block_type,
        title=block.title,
        text=(block.raw_text or block.text).strip(),
        score=block.score,
        preserved_reason=f"{block.preserved_reason}；自检回退: {reason}",
        image_refs=block.image_refs,
        raw_text=block.raw_text,
        fallback_used=True,
    )


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
