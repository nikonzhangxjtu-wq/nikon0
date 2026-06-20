"""未走 RAG、领域不明确或为 unknown 时的通用安全 prompt。

原则：不知道就说不知道，不编造，给一条最实际的建议。
"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.answer_structure import STRUCTURED_ANSWER_FRAMEWORK
from app.utils.prompts.context import PromptContext


class NoRagGenericPromptBuilder:
    def build(self, ctx: PromptContext) -> str:
        from app.utils.manual_lang import query_prefers_chinese_embedding

        lang = generation_reply_language_rule(ctx.question)
        is_cn = query_prefers_chinese_embedding(ctx.question)

        if ctx.evidence_status == "no_passing_chunks":
            evidence_note = (
                "⚠️ 已尝试检索手册知识库但无可用结果，不得假装引用手册内容。\n"
                "⚠️ Attempted to search the manual knowledge base but found no usable results. Do not pretend to cite the manual.\n"
            )
        elif ctx.evidence_status == "insufficient_chunks":
            evidence_note = (
                "⚠️ 检索返回片段偏少，证据不足，不要展开臆测性技术细节。\n"
                "⚠️ Too few search results; evidence is insufficient. Do not speculate on technical details.\n"
            )
        else:
            evidence_note = (
                "注意：本轮无可用的检索上下文，不得编造型号、参数、步骤、故障码等具体细节。\n"
                "Note: No retrieval context is available this round. Do not fabricate models, parameters, steps, or error codes.\n"
            )

        low_conf = ""
        if ctx.route_low_confidence:
            low_conf = (
                "另：路由置信度偏低，少做绝对断言，多引导用户澄清需求。\n"
                "Also: route confidence is low; avoid absolute claims, guide the user to clarify their needs.\n"
            )

        visual = ""
        if ctx.visual_context:
            visual = (
                "用户上传了图片，视觉摘要如下（仅辅助，不是已核实事实）：\n"
                "User uploaded images; visual summary below (auxiliary only, not verified facts):\n"
                f"{ctx.visual_context}\n"
            )

        if is_cn:
            rules = (
                "5. 如果无法给出确切答案，请直接说\"目前没有足够信息给出准确答复\"，然后给一条最实际的建议（如查说明书、联系官方客服、提供型号等）。\n"
                "6. 中文答案通常 50～200 字，覆盖所有子问题即可，不要为凑字数科普常识。\n"
                "7. 覆盖每个子问题，但不知道的就说不知道，不要猜。\n"
            )
            footer = "（提醒：逐一覆盖所有子问题，不知道就直说，不要写模板开头结尾）"
        else:
            rules = (
                "5. If you cannot give a definitive answer, say \"There is not enough information for an accurate answer\", "
                "then provide the most practical suggestion (check the manual, contact official support, provide the model number, etc.).\n"
                "6. English answers are typically 40–150 words. Cover all sub-questions without padding.\n"
                "7. Address every sub-question, but say you don't know when you don't — do not guess.\n"
            )
            footer = "(Reminder: cover every sub-question; if you don't know, say so directly; no template intros or closings)"

        conv_history = ""
        if ctx.conversation_history:
            conv_history = f"{ctx.conversation_history}\n"

        memory = ""
        if ctx.memory_context:
            memory = (
                "可用记忆 / Available Memory：\n"
                f"{ctx.memory_context}\n"
            )

        return f"""{lang}

你是电商/设备客服助手。{evidence_note}
You are an e-commerce/device customer support assistant. {evidence_note}

{STRUCTURED_ANSWER_FRAMEWORK}

【通用场景补充规则 / General Supplemental Rules】
{rules}
{low_conf}{visual}{memory}{conv_history}
用户问题 / Question：
{ctx.question}

{lang}

{footer}
""".strip()
