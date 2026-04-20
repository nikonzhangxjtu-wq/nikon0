"""未走 RAG、且路由为客服域时的 prompt（不编造订单/政策细节）。"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.context import PromptContext


class NoRagCustomerServicePromptBuilder:
    def build(self, ctx: PromptContext) -> str:
        hint = ctx.domain_hint or "customer_service"
        reason = ctx.route_reason or "（未提供）"
        lang = generation_reply_language_rule(ctx.question)
        return f"""你是电商/设备场景的客服助手。本轮未接入可检索的政策/订单知识库，请不要编造订单号、金额、物流状态、具体退换货结果或内部条款。

回答要求：
0）{lang}
1）先用 1～2 句话复述用户诉求并表示理解。
2）只给出通用、合规的处理路径（例如：如何联系官方渠道、需要准备哪些材料、建议用户保留哪些凭证），避免断言具体政策结论。
3）若需要个案信息才能处理，请列出需用户补充的信息清单（如订单号、购买时间、问题照片、序列号等）。
4）条理清晰，可分点；语气专业、克制。
5）领域提示（仅供参考，不得当作事实依据）：{hint}
6）路由说明（仅供你理解意图，勿复述给用户）：{reason}

用户问题：
{ctx.question}
""".strip()
