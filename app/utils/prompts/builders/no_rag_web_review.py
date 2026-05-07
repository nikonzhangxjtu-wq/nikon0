"""联网口碑 Skill 的回答模板。"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.answer_structure import STRUCTURED_ANSWER_FRAMEWORK
from app.utils.prompts.context import PromptContext


class NoRagWebReviewPromptBuilder:
    """基于联网证据块生成口碑总结，禁止编造来源与评分。"""

    def build(self, ctx: PromptContext) -> str:
        lang = generation_reply_language_rule(ctx.question)

        evidence_note = ""
        if not ctx.context_block:
            evidence_note = (
                "⚠️ 当前未拿到可用的口碑证据，请明确说明证据不足，"
                "不要编造平台评分、销量或用户反馈。\n"
            )

        return f"""{lang}

你是电商/设备客服助手。当前任务是根据上下文中的口碑评价证据回答用户问题（证据可能来自本地评价表或联网检索）。

{STRUCTURED_ANSWER_FRAMEWORK}

【口碑证据规则】
1. 仅基于上下文中的“口碑评价摘要/来源”回答，不得捏造评分、销量、平台政策。
2. 优先给出：总体倾向、高频优点、高频缺点、适用建议。
3. 若证据不足，请直接说明“当前证据不足”，并建议用户补充关注点（预算、噪音、续航等）。
4. 若上下文中有来源链接，回答末尾可简要提示“可参考来源列表”。
{evidence_note}
口碑证据：
{ctx.context_block or "（无可用口碑证据）"}

用户问题：
{ctx.question}

{lang}
""".strip()

