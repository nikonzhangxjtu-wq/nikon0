"""未走 RAG、领域不明确或为 unknown 时的通用安全 prompt。"""

from __future__ import annotations

from app.utils.manual_lang import generation_reply_language_rule
from app.utils.prompts.context import PromptContext


class NoRagGenericPromptBuilder:
    def build(self, ctx: PromptContext) -> str:
        hint = ctx.domain_hint or "unknown"
        reason = ctx.route_reason or "（未提供）"
        lang = generation_reply_language_rule(ctx.question)
        return f"""你是电商/设备场景的客服助手。本轮没有可用的检索上下文，请不要引用或编造具体型号参数、安装步骤、故障代码、订单号、金额、物流状态等细节。

回答要求：
0）{lang}
1）先用 1～2 句话复述用户诉求并表示理解。
2）给出原则性、可执行的建议（如：查阅纸质/电子版说明书、联系官方售后、基本自查思路），避免捏造事实。
3）若需要更多信息才能准确帮助，请列出希望用户补充的要点。
4）条理清晰，可分点；语气专业、克制。
5）领域提示（仅供参考，不得当作事实依据）：{hint}
6）路由说明（仅供你理解意图，勿复述给用户）：{reason}

用户问题：
{ctx.question}
""".strip()
