"""带检索上下文的说明书 / 通用 RAG prompt。

有检索片段时：必须严格依据上下文作答，不编造。
"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.answer_structure import STRUCTURED_ANSWER_FRAMEWORK
from app.utils.prompts.context import PromptContext


class RagManualPromptBuilder:
    """有检索片段时：要求模型严格依据上下文作答。"""

    def build(self, ctx: PromptContext) -> str:
        from app.utils.manual_lang import query_prefers_chinese_embedding

        lang = generation_reply_language_rule(ctx.question)
        is_cn = query_prefers_chinese_embedding(ctx.question)

        low_conf = ""
        if ctx.route_low_confidence:
            low_conf = "注意：路由置信度偏低，回答时多留余地，少做绝对断言。\n"

        visual = ""
        if ctx.visual_context:
            visual = (
                "用户上传了图片，视觉摘要如下（仅辅助参考，不可覆盖上下文事实）：\n"
                f"{ctx.visual_context}\n"
            )

        react_hint = ""
        if ctx.react_multi_evidence:
            react_hint = (
                "ℹ️ 以下上下文来自多轮定向检索结果的合并去重，覆盖了不同角度和关键词，"
                "信息更全面，请充分利用。\n"
            )

        conv_history = ""
        if ctx.conversation_history:
            conv_history = f"{ctx.conversation_history}\n"

        # 语言感知的元素
        if is_cn:
            header = "你是电商/设备客服助手，现在有产品手册的检索结果可供参考。"
            rule9 = "9. 步骤类答案按实际需要列出全部步骤，但要精简每条描述。中文答案通常 100～500 字，覆盖所有子问题即可，不必凑字数。"
            ctx_label = "上下文："
            q_label = "用户问题："
            conv_rule = ""
            if ctx.conversation_history:
                conv_rule = "10. 上方「对话历史」提供了此前轮次的上下文。当前问题可能包含指代（如「这个」「上述」「第二步」），请结合历史理解用户意图，但回答仍需基于「上下文」中的手册内容。\n"
            footer = "（提醒：逐一覆盖所有子问题，步骤列全但每条精简，不要写模板开头结尾）"
        else:
            header = "You are an e-commerce/device customer support assistant. Below are relevant excerpts from product manuals."
            rule9 = "9. Format step-by-step answers as a clean list, keeping each step concise. English answers are typically 80–300 words. Cover all sub-questions without fluff."
            ctx_label = "Context:"
            q_label = "Question:"
            conv_rule = ""
            if ctx.conversation_history:
                conv_rule = "10. The Conversation History above provides context from previous turns. The current question may contain references (e.g. \"this\", \"the above\", \"step two\"). Use the history to understand the user's intent, but base your answer on the Context below.\n"
            footer = "(Reminder: cover every sub-question, list all steps concisely, no template intros or closings)"

        return f"""{lang}

{header}

{STRUCTURED_ANSWER_FRAMEWORK}

【RAG 场景补充规则 / RAG Supplemental Rules】
5. 回答必须基于下方「上下文」。可引用片段编号（chunk_id），不得编造上下文中没有的型号、步骤、参数、政策。
   Base answers strictly on the context below. Cite chunk_id if helpful. Do not fabricate models, steps, specs, or policies.
6. 上下文中形如 `<IMG:xxx>` 的标记是图片引用。当图片能帮助理解操作步骤、部件位置、指示灯含义等信息时直接引用，不限制数量。仅当图片无信息增量（纯装饰、文字已充分说明的简单图示）时跳过。
   `<IMG:xxx>` tokens in the context are image references to figures/diagrams. Cite them whenever they help the user understand steps, part locations, indicators, etc. No arbitrary limit — cite as many as the answer needs. Skip only images that add zero information (purely decorative or when text alone is fully sufficient).
7. 充分利用上下文中已有的信息回答每个子问题。凡上下文已包含的内容，严禁说"手册未提及""信息不足"等推脱话术——这是严重错误。
   Fully use the information available in the context for each sub-question. If the context contains relevant info, you MUST NOT claim "not mentioned in manual" or "insufficient info" — that is a critical error.
8. 仅当某个子问题在上下文中确实完全找不到任何相关信息时，才可简短说明该点信息不足。即便如此，也必须先回答上下文能够覆盖的部分。
   Only when a sub-question truly has zero relevant info in the context may you briefly note it. Even then, answer the parts the context can cover first.
{rule9}
{conv_rule}
{conv_history}{react_hint}{visual}{low_conf}
{ctx_label}
{ctx.context_block}

{q_label}
{ctx.question}

{lang}

{footer}
""".strip()
