"""带检索上下文的说明书 / 通用 RAG prompt。"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.context import PromptContext


class RagManualPromptBuilder:
    """有检索片段时：要求模型严格依据上下文作答。"""

    def build(self, ctx: PromptContext) -> str:
        lang = generation_reply_language_rule(ctx.question)
        return f"""你是电商/设备客服助手。

规则：
0）{lang}
1）直接回答用户问题。
2）优先使用下方「上下文」中的信息；引用时可提及片段编号或 chunk_id。
3）若上下文不足以回答，明确说明缺少哪些信息，不要编造型号、步骤、故障码或政策细节。
4）条理清晰，可适当分点。
5）领域提示（仅供参考）：{ctx.domain_hint}

上下文：
{ctx.context_block}

用户问题：
{ctx.question}
""".strip()
