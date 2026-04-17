"""Prompt 构建器注册与解析。

新增一种 prompt：实现 PromptBuilder + 在此注册 +（可选）调整 resolve_prompt_key。
"""

from __future__ import annotations

from app.utils.prompts.builders.no_rag_customer_service import NoRagCustomerServicePromptBuilder
from app.utils.prompts.builders.no_rag_generic import NoRagGenericPromptBuilder
from app.utils.prompts.builders.rag_manual import RagManualPromptBuilder
from app.utils.prompts.builders.base import PromptBuilder
from app.utils.prompts.context import PromptContext

_REGISTRY: dict[str, PromptBuilder] = {
    "rag_manual": RagManualPromptBuilder(),
    "no_rag_customer_service": NoRagCustomerServicePromptBuilder(),
    "no_rag_generic": NoRagGenericPromptBuilder(),
}


def resolve_prompt_key(ctx: PromptContext) -> str:
    if ctx.need_rag:
        return "rag_manual"
    if ctx.domain_hint == "customer_service":
        return "no_rag_customer_service"
    return "no_rag_generic"


def compose_generation_prompt(ctx: PromptContext) -> str:
    key = resolve_prompt_key(ctx)
    builder = _REGISTRY.get(key)
    if builder is None:
        builder = _REGISTRY["no_rag_generic"]
    return builder.build(ctx)


def register_prompt_builder(key: str, builder: PromptBuilder) -> None:
    """运行时覆盖或扩展注册表（测试或插件用）。"""
    _REGISTRY[key] = builder
