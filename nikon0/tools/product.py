"""Product-support tools for resolver, retrieval, and grounding checks."""

from __future__ import annotations

import re
from typing import Any

from nikon0.app.schemas.capability import Evidence, ToolCallRequest, ToolCallResult, ToolSpec
from nikon0.app.schemas.knowledge import KnowledgeRequest
from nikon0.knowledge.product_resolver import ProductResolver
from nikon0.knowledge.runtime import KnowledgeRuntime


class ResolveProductTool:
    """Resolve product scope from user text and session state."""

    spec = ToolSpec(
        service_id="product-support",
        tool_name="resolve_product",
        description="Resolve product id, manuals, and ambiguity state from a product-support query.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "session_state": {"type": "object"},
            },
            "required": ["message"],
        },
    )

    def __init__(self, resolver: ProductResolver | None = None) -> None:
        self.resolver = resolver or ProductResolver()

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        message = str(request.arguments.get("message") or request.arguments.get("query") or "").strip()
        session_state = request.arguments.get("session_state")
        resolution = self.resolver.resolve(
            message,
            session_state=session_state if isinstance(session_state, dict) else None,
        )
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={"resolution": resolution.to_trace()},
        )


class SearchProductManualTool:
    """Search manuals through KnowledgeRuntime."""

    spec = ToolSpec(
        service_id="product-support",
        tool_name="search_product_manual",
        description="Retrieve manual evidence for a product-support query through KnowledgeRuntime.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "product_model": {"type": "string"},
                "allowed_manual_names": {"type": "array", "items": {"type": "string"}},
                "max_evidence": {"type": "integer"},
            },
            "required": ["query"],
        },
    )

    def __init__(self, knowledge_runtime: KnowledgeRuntime | None = None) -> None:
        self.knowledge_runtime = knowledge_runtime or KnowledgeRuntime()

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        arguments = request.arguments
        query = str(arguments.get("query") or arguments.get("message") or "").strip()
        if not query:
            return ToolCallResult(
                ok=False,
                service_id=request.service_id,
                tool_name=request.tool_name,
                error_code="invalid_arguments",
                error_message="query is required",
            )
        knowledge_request = KnowledgeRequest(
            query=query,
            product_model=_optional_string(arguments.get("product_model")),
            intent=str(arguments.get("intent") or "product_support"),
            need_images=bool(arguments.get("need_images") or arguments.get("images")),
            images=_string_list(arguments.get("images")),
            allowed_manual_names=_string_list(arguments.get("allowed_manual_names")),
            knowledge_version=_optional_string(arguments.get("knowledge_version")),
            tenant_id=_optional_string(arguments.get("tenant_id")),
            user_id=_optional_string(arguments.get("user_id")),
            max_evidence=int(arguments.get("max_evidence") or 3),
        )
        result = await self.knowledge_runtime.query(knowledge_request)
        manual_names = sorted(
            {
                str(item.payload.get("manual_name"))
                for item in result.evidence
                if item.payload.get("manual_name")
            }
        )
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={
                "answer_hints": list(result.answer_hints),
                "evidence": [item.model_dump() for item in result.evidence],
                "manual_names": manual_names,
                "backend_trace": result.backend_trace,
            },
        )


class ValidateAnswerGroundingTool:
    """Deterministic first-pass grounding validator.

    This tool is intentionally conservative and local in Phase 1. It gives the
    runtime a stable validation step before we add LLM-as-judge or policy gates.
    """

    spec = ToolSpec(
        service_id="product-support",
        tool_name="validate_answer_grounding",
        description="Check whether an answer is covered by provided evidence and required facts.",
        risk_level="low",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence": {"type": "array"},
                "required_terms": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "evidence"],
        },
    )

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        answer = str(request.arguments.get("answer") or "")
        evidence = _evidence_items(request.arguments.get("evidence"))
        required_terms = _string_list(request.arguments.get("required_terms"))
        evidence_text = "\n".join(item.text for item in evidence)
        missing_terms = [
            term
            for term in required_terms
            if _normalize(term) not in _normalize(answer)
        ]
        token_overlap = _meaningful_overlap(answer, evidence_text)
        grounded = bool(answer.strip()) and bool(evidence) and not missing_terms and token_overlap > 0
        return ToolCallResult(
            ok=True,
            service_id=request.service_id,
            tool_name=request.tool_name,
            data={
                "grounded": grounded,
                "missing_terms": missing_terms,
                "evidence_count": len(evidence),
                "token_overlap": token_overlap,
                "reason": "answer is covered by evidence" if grounded else "answer is not fully covered",
            },
        )


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _evidence_items(value: Any) -> list[Evidence]:
    if not isinstance(value, list):
        return []
    items: list[Evidence] = []
    for item in value:
        if isinstance(item, Evidence):
            items.append(item)
        elif isinstance(item, dict):
            try:
                items.append(Evidence(**item))
            except Exception:
                continue
    return items


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _meaningful_overlap(answer: str, evidence_text: str) -> int:
    answer_tokens = set(_tokens(answer))
    evidence_tokens = set(_tokens(evidence_text))
    return len(answer_tokens & evidence_tokens)


def _tokens(value: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", value.lower())
    return [token for token in raw_tokens if len(token) >= 2]
