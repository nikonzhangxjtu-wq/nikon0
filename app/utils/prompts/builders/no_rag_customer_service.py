"""未走 RAG、且路由为客服域时的 prompt。

关键原则：不编造订单/政策细节，给通用合规引导即可。
"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.answer_structure import STRUCTURED_ANSWER_FRAMEWORK
from app.utils.prompts.context import PromptContext


class NoRagCustomerServicePromptBuilder:
    def build(self, ctx: PromptContext) -> str:
        from app.utils.manual_lang import query_prefers_chinese_embedding

        lang = generation_reply_language_rule(ctx.question)
        is_cn = query_prefers_chinese_embedding(ctx.question)

        retrieval_note = ""
        if ctx.evidence_status == "no_passing_chunks":
            retrieval_note = (
                "⚠️ 系统已尝试检索知识库但无可用证据，请勿编造任何政策、条款或流程细节。\n"
                "⚠️ Attempted knowledge base search found no usable evidence. Do not fabricate policies, terms, or procedures.\n"
            )
        elif ctx.evidence_status == "insufficient_chunks":
            retrieval_note = (
                "⚠️ 检索证据不足，回答时少做具体断言，多引导用户联系官方确认。\n"
                "⚠️ Insufficient evidence from retrieval; avoid specific claims, guide the user to contact official channels.\n"
            )

        low_conf = ""
        if ctx.route_low_confidence:
            low_conf = (
                "另：路由置信度偏低，请多澄清用户需求、少下定论。\n"
                "Also: route confidence is low; clarify the user's needs rather than making definitive statements.\n"
            )

        visual = ""
        if ctx.visual_context:
            visual = (
                "用户上传了图片，视觉摘要如下（仅辅助理解，不是已核实事实）：\n"
                "User uploaded images; visual summary below (auxiliary understanding only, not verified facts):\n"
                f"{ctx.visual_context}\n"
            )

        if is_cn:
            header = "你是电商/设备场景的客服助手。本轮未接入具体政策/订单知识库，请不要编造订单号、金额、物流状态、具体退换货结果或内部条款。"
            rules = (
                "5. 每个子问题至少给出 1-2 句实质性回应。知道就说知道，不确定就坦承不确定并建议联系哪里确认，不要跳过任何子问题。\n"
                "6. 给通用、合规的建议方向，不承诺个案结果。例如\"建议联系购买平台客服，提供订单号和商品照片，他们会按平台规则处理\"。\n"
                "7. 中文答案通常 80～300 字，覆盖所有子问题即可，不凑字数。\n"
            )
            q_label = "用户问题："
            footer = "（提醒：逐一覆盖所有子问题，不要漏答，不要写模板开头结尾）"
        else:
            header = "You are an e-commerce/device customer support assistant. This round has no access to specific policy or order knowledge bases. Do not fabricate order numbers, amounts, shipping statuses, refund results, or internal policies."
            rules = (
                "5. Give at least 1-2 substantive sentences per sub-question. Say what you know, openly admit uncertainty where needed, and suggest where to confirm.\n"
                "6. Provide general, compliant guidance. Do not promise case-specific outcomes. Example: \"Contact the platform's customer service with your order number and product photos; they will handle it per platform policy.\"\n"
                "7. English answers are typically 50–200 words. Cover all sub-questions without padding.\n"
            )
            q_label = "Question:"
            footer = "(Reminder: cover every sub-question fully, no template intros or closings)"

        conv_history = ""
        if ctx.conversation_history:
            conv_history = f"{ctx.conversation_history}\n"

        return f"""{lang}

{header}

{STRUCTURED_ANSWER_FRAMEWORK}

【客服场景补充规则 / CS Supplemental Rules】
{rules}
{retrieval_note}{low_conf}{visual}{conv_history}
{q_label}
{ctx.question}

{lang}

{footer}
""".strip()
