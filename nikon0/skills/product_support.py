"""Product support skill backed by KnowledgeRuntime."""

from __future__ import annotations

from typing import Any

from nikon0.app.schemas.agent import AgentContext
from nikon0.app.schemas.capability import Evidence, FallbackPolicy, SkillManifest, SkillMatch, SkillResult, StateUpdate, ToolCallRequest
from nikon0.app.schemas.knowledge import KnowledgeResult
from nikon0.knowledge.product_resolver import (
    ProductResolution,
    ProductResolver,
    apply_product_disclosure,
    build_disambiguation_answer,
)
from nikon0.knowledge.runtime import KnowledgeRuntime
from nikon0.llm.generation import LlmAnswerGenerator
from nikon0.skills.routing_signals import looks_like_case_intake, looks_like_product_support
from nikon0.tools.product import ResolveProductTool, SearchProductManualTool, ValidateAnswerGroundingTool
from nikon0.tools.runtime import ToolRegistry, ToolRuntime


PRODUCT_SUPPORT_DESCRIPTION = (
    "商品说明书知识问答：基于产品手册 RAG 回答使用、安装、清洁保养、参数规格、"
    "功能模式、故障排查与安全限制等问题；不处理退款/物流/订单/投诉受理。"
)


class ProductSupportSkill:
    name = "product_support"
    description = PRODUCT_SUPPORT_DESCRIPTION
    risk_level = "low"
    manifest = SkillManifest(
        name=name,
        title="商品技术支持",
        description=PRODUCT_SUPPORT_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence": {"type": "array"},
            },
        },
        capabilities=[
            "manual_qa",
            "product_usage",
            "installation",
            "cleaning_maintenance",
            "replacement_schedule",
            "technical_specs",
            "feature_explanation",
            "troubleshooting",
            "safety_limits",
        ],
        required_tools=[
            "product-support.resolve_product",
            "product-support.search_product_manual",
            "product-support.validate_answer_grounding",
        ],
        risk_level="low",
        fallback_policy=FallbackPolicy(allow_general_fallback=True, allow_handoff=False),
    )

    def __init__(
        self,
        knowledge_runtime: KnowledgeRuntime | None = None,
        answer_generator: LlmAnswerGenerator | None = None,
        product_resolver: ProductResolver | None = None,
    ) -> None:
        self.knowledge_runtime = knowledge_runtime or KnowledgeRuntime()
        self.answer_generator = answer_generator
        self.product_resolver = product_resolver or ProductResolver()
        self.tool_runtime = ToolRuntime(
            registry=ToolRegistry(
                [
                    ResolveProductTool(self.product_resolver),
                    SearchProductManualTool(self.knowledge_runtime),
                    ValidateAnswerGroundingTool(),
                ]
            )
        )

    async def can_handle(self, context: AgentContext) -> SkillMatch:
        """Rule-fallback hint only; model/planner/sticky routes do not consult this guard."""
        message = context.request.message.strip()
        if looks_like_case_intake(message):
            return SkillMatch(
                matched=False,
                confidence=0.0,
                reason="message looks like case intake, not manual QA",
            )
        matched, hits = looks_like_product_support(message)
        if matched:
            preview = ", ".join(hits[:4])
            return SkillMatch(
                matched=True,
                confidence=0.82,
                reason=f"matched product support signals: {preview}",
            )
        return SkillMatch(
            matched=False,
            confidence=0.0,
            reason="no product manual QA signal for rule fallback",
        )

    async def run(self, context: AgentContext) -> SkillResult:
        session_state = (
            context.session_state.flat_state
            if context.session_state is not None
            else None
        )
        resolution_result = await self.tool_runtime.call_step(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="resolve_product",
                arguments={
                    "message": context.request.message,
                    "session_state": session_state or {},
                },
            ),
        )
        resolution = self._resolution_from_tool_result(resolution_result.data)
        context.trace.add_event(
            "product.resolve",
            f"product resolution status={resolution.status}",
            **resolution.to_trace(),
        )

        search_result = await self.tool_runtime.call_step(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="search_product_manual",
                arguments={
                    "query": context.request.message,
                    "intent": "product_support",
                    "need_images": bool(context.request.images),
                    "images": context.request.images,
                    "tenant_id": str(context.request.metadata.get("tenant_id") or "") or None,
                    "user_id": context.request.user_id,
                    "max_evidence": 3,
                    "product_model": resolution.product_id,
                    "allowed_manual_names": list(resolution.manual_names),
                },
            ),
        )
        result = self._knowledge_result_from_tool_result(search_result.data)
        if resolution.status == "disambiguation_required":
            resolution = self.product_resolver.resolve_from_retrieval(
                resolution,
                self._manual_names_from_evidence(result),
                self._manual_scores_from_evidence(result),
            )
            if resolution.status == "resolved":
                context.trace.add_event(
                    "product.resolve_from_retrieval",
                    "retrieval evidence resolved ambiguous product",
                    **resolution.to_trace(),
                )
                result = self._filter_result_to_manuals(result, resolution.manual_names)
        context.trace.knowledge_calls.append(
            {
                "query": context.request.message,
                "intent": "product_support",
                "evidence_count": len(result.evidence),
                "backend_trace": result.backend_trace,
                "product_resolution": resolution.to_trace(),
            }
        )
        if resolution.status == "disambiguation_required":
            return SkillResult(
                status="needs_more_info",
                answer_draft=build_disambiguation_answer(resolution.candidates),
                risk_level="low",
                state_updates=[
                    StateUpdate(
                        key="product_support",
                        value=self._disambiguation_state(resolution),
                        reason="retrieval could not disambiguate product",
                    )
                ],
            )
        if not result.evidence:
            return SkillResult(
                status="needs_more_info",
                answer_draft="我还没有找到足够的商品手册证据。请补充产品型号、故障码或具体操作场景。",
                risk_level="low",
                state_updates=[
                    StateUpdate(
                        key="product_support",
                        value=self._resolved_state(
                            context.request.message,
                            resolution,
                            evidence_count=0,
                            manual_names=[],
                        ),
                        reason="product support found no evidence",
                    )
                ],
            )

        context.evidence_context = list(result.evidence)
        governance = context.context_governance
        if governance is not None and hasattr(governance, "agovern"):
            await governance.agovern(context)
        fallback_answer = self._compose_answer(context.request.message, result.answer_hints)
        answer = fallback_answer
        product_context = self._product_context(resolution)
        if self.answer_generator is not None:
            answer = await self.answer_generator.product_support_answer(
                context=context,
                evidence=result.evidence,
                answer_hints=result.answer_hints,
                fallback_answer=fallback_answer,
                product_context=product_context,
            )
        answer = apply_product_disclosure(answer, resolution)
        grounding_result = await self.tool_runtime.call_step(
            context,
            ToolCallRequest(
                service_id="product-support",
                tool_name="validate_answer_grounding",
                arguments={
                    "answer": answer,
                    "evidence": [item.model_dump() for item in result.evidence],
                    "required_terms": [],
                },
            ),
        )
        context.trace.add_event(
            "evidence.usage",
            "recorded product-support evidence usage",
            retrieved_evidence_ids=[item.evidence_id for item in result.evidence],
            included_evidence_ids=[item.evidence_id for item in result.evidence],
            used_evidence_ids=[item.evidence_id for item in result.evidence],
            grounding_checked=bool(grounding_result.ok),
            grounding=grounding_result.data,
            product_resolution=resolution.to_trace(),
        )
        manual_names = sorted(
            {
                str(item.payload.get("manual_name"))
                for item in result.evidence
                if item.payload.get("manual_name")
            }
        )
        return SkillResult(
            status="success",
            answer_draft=answer,
            evidence=result.evidence,
            state_updates=[
                StateUpdate(
                    key="product_support",
                    value=self._resolved_state(
                        context.request.message,
                        resolution,
                        evidence_count=len(result.evidence),
                        manual_names=manual_names,
                    ),
                    reason="product support answered with knowledge evidence",
                    evidence_ids=[item.evidence_id for item in result.evidence],
                )
            ],
            risk_level="low",
        )

    @staticmethod
    def _product_context(resolution) -> dict[str, Any] | None:
        if resolution.status != "resolved" or not resolution.display_name:
            return None
        return {
            "product_id": resolution.product_id,
            "display_name": resolution.display_name,
            "manual_names": list(resolution.manual_names),
            "disclose_default_product": resolution.disclose_default_product,
            "matched_terms": list(resolution.matched_terms),
            "source": resolution.source,
        }

    @staticmethod
    def _disambiguation_state(resolution) -> dict[str, Any]:
        return {
            "disambiguation_pending": True,
            "disambiguation_candidates": [item.product_id for item in resolution.candidates],
            "last_query": "",
            "evidence_count": 0,
            "manual_names": [],
        }

    @staticmethod
    def _resolved_state(
        query: str,
        resolution,
        *,
        evidence_count: int,
        manual_names: list[str],
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "disambiguation_pending": False,
            "disambiguation_candidates": [],
            "last_query": query,
            "evidence_count": evidence_count,
            "manual_names": manual_names,
            "product_resolution": resolution.to_trace(),
        }
        if resolution.status == "resolved" and resolution.product_id:
            state.update(
                {
                    "selected_product_id": resolution.product_id,
                    "selected_display_name": resolution.display_name,
                    "allowed_manual_names": list(resolution.manual_names),
                }
            )
        return state

    def _resolution_from_tool_result(self, data: dict[str, Any]) -> ProductResolution:
        raw = data.get("resolution") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return ProductResolution(status="passthrough", source="passthrough", reason="invalid tool result")
        candidate_ids = [
            str(item)
            for item in raw.get("candidate_product_ids", [])
            if str(item).strip()
        ]
        candidates = tuple(
            product
            for product_id in candidate_ids
            if (product := self.product_resolver.catalog.product_by_id(product_id)) is not None
        )
        return ProductResolution(
            status=raw.get("status") or "passthrough",
            source=raw.get("source") or "passthrough",
            product_id=raw.get("product_id"),
            display_name=raw.get("display_name"),
            manual_names=tuple(str(item) for item in raw.get("manual_names", []) if str(item).strip()),
            matched_terms=tuple(str(item) for item in raw.get("matched_terms", []) if str(item).strip()),
            disclose_default_product=bool(raw.get("disclose_default_product")),
            candidates=candidates,
            reason=str(raw.get("reason") or ""),
        )

    @staticmethod
    def _knowledge_result_from_tool_result(data: dict[str, Any]) -> KnowledgeResult:
        if not isinstance(data, dict):
            return KnowledgeResult()
        evidence: list[Evidence] = []
        for item in data.get("evidence", []):
            if isinstance(item, Evidence):
                evidence.append(item)
            elif isinstance(item, dict):
                evidence.append(Evidence(**item))
        return KnowledgeResult(
            answer_hints=[str(item) for item in data.get("answer_hints", []) if str(item).strip()],
            evidence=evidence,
            backend_trace=list(data.get("backend_trace", [])),
        )

    @staticmethod
    def _compose_answer(question: str, hints: list[str]) -> str:
        top = hints[:3]
        lines = [
            "根据当前商品手册证据，建议如下：",
            f"- 你的问题：{question}",
        ]
        for idx, hint in enumerate(top, start=1):
            lines.append(f"- 证据 {idx}：{hint}")
        lines.append("如果你的产品型号或故障码与上述证据不一致，请补充型号后我再缩小范围。")
        return "\n".join(lines)

    @staticmethod
    def _manual_names_from_evidence(result: KnowledgeResult) -> list[str]:
        names: list[str] = []
        for item in result.evidence:
            manual_name = str(item.payload.get("manual_name") or "").strip()
            if manual_name:
                names.append(manual_name)
        return names

    @staticmethod
    def _manual_scores_from_evidence(result: KnowledgeResult) -> list[float]:
        return [float(item.confidence) for item in result.evidence]

    @staticmethod
    def _filter_result_to_manuals(result: KnowledgeResult, manual_names: tuple[str, ...]) -> KnowledgeResult:
        allowed = set(manual_names)
        if not allowed:
            return result
        evidence = [
            item
            for item in result.evidence
            if str(item.payload.get("manual_name") or "").strip() in allowed
        ]
        hints = [
            hint
            for hint in result.answer_hints
            if any(f"[{manual_name}]" in hint for manual_name in allowed)
        ]
        return result.model_copy(update={"evidence": evidence, "answer_hints": hints})
