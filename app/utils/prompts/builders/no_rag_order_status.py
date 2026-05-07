"""订单状态 skill 的回答模板。"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.answer_structure import STRUCTURED_ANSWER_FRAMEWORK
from app.utils.prompts.context import PromptContext


class NoRagOrderStatusPromptBuilder:
    """基于订单进度证据块回答，不编造物流状态与时效。"""

    def build(self, ctx: PromptContext) -> str:
        lang = generation_reply_language_rule(ctx.question)
        evidence_note = ""
        if not ctx.context_block:
            evidence_note = (
                "⚠️ 当前未拿到可用订单状态证据，请明确说明无法直接查到并引导用户提供订单号。\n"
            )

        return f"""{lang}

你是电商客服助手。当前任务是基于订单状态证据回答用户进度查询。

{STRUCTURED_ANSWER_FRAMEWORK}

【订单状态规则】
1. 仅依据“订单进度信息”回答，不得编造物流节点、签收状态或退款结果。
2. 优先给出：当前状态、预计时间、下一步建议（催单/联系客服/补充信息）。
3. 若证据不足，请直接说明并提示用户提供订单号或收件手机号后四位。
4. 若上下文包含多笔订单，先确认用户要查询的具体订单号。
{evidence_note}
订单证据：
{ctx.context_block or "（无可用订单证据）"}

用户问题：
{ctx.question}

{lang}
""".strip()
