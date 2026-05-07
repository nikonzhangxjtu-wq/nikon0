"""端到端对话流水线编排。

骨架步骤：
1）路由 / RAG 门控
2）需要时检索
3）检索后验证门（无可用证据则保守生成）
4）拼 prompt 上下文
5）生成答案
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.services.generator import Qwen2Generator
from app.services.online_review_skill import (
    NullReviewProvider,
    OnlineReviewSkill,
    ReviewSearchProvider,
)
from app.services.order_status_skill import (
    NullOrderStatusProvider,
    OrderStatusProvider,
    OrderStatusSkill,
)
from app.services.skills.case_intake_skill import CaseIntakeSkill
from app.services.skills.local_review_table import LocalReviewTableProvider
from app.services.skills.mcp_order_provider import MCPOrderProvider
from app.services.skills.mcp_review_provider import MCPReviewProvider
from app.services.rag_skill.query_construction import query_construction
from app.services.retriever import RetrievalTrace, VectorRetriever, retriever_context_filter
from app.services.router import QuestionRouter, RouteDecision
from app.services.vision import VisionInterpreter
from app.utils.prompt_builder import build_multimodal_context_block, finalize_answer_images
from app.utils.prompts import PromptContext, compose_generation_prompt

_llm_router = None
_react_agent = None
_rewriter = None
_online_review_skill = None
_order_status_skill = None
_case_intake_skill = None


def _get_rewriter():
    global _rewriter
    if _rewriter is None:
        from app.services.query_rewriter import QueryRewriter
        _rewriter = QueryRewriter()
    return _rewriter


def _get_llm_router():
    global _llm_router
    if _llm_router is None:
        from app.services.llm_router import LLMRouter
        _llm_router = LLMRouter()
    return _llm_router


def _get_react_agent():
    global _react_agent
    if _react_agent is None:
        from app.services.react_agent import ReActAgent
        _react_agent = ReActAgent()
    return _react_agent


def _review_search_mode() -> str:
    mode = (settings.review_search_mode or "local").strip().lower()
    if mode not in {"local", "mcp", "none"}:
        return "local"
    return mode


def _get_online_review_skill() -> OnlineReviewSkill:
    global _online_review_skill
    if _online_review_skill is None:
        provider: ReviewSearchProvider = NullReviewProvider()
        mode = _review_search_mode()
        if mode == "none":
            provider = NullReviewProvider()
        elif mode == "mcp":
            endpoint = (settings.mcp_review_endpoint or "").strip()
            if endpoint:
                try:
                    provider = MCPReviewProvider(
                        endpoint=endpoint,
                        api_key=settings.mcp_review_api_key,
                        timeout_sec=settings.mcp_review_timeout_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] MCPReviewProvider 初始化失败，回退 NullReviewProvider: {exc}")
                    provider = NullReviewProvider()
        else:
            provider = LocalReviewTableProvider()
        _online_review_skill = OnlineReviewSkill(provider=provider)
    return _online_review_skill


def reset_online_review_skill_singleton() -> None:
    """单测或切换配置后重置口碑 skill 单例。"""
    global _online_review_skill
    _online_review_skill = None


def _get_order_status_skill() -> OrderStatusSkill:
    global _order_status_skill
    if _order_status_skill is None:
        provider: OrderStatusProvider = NullOrderStatusProvider()
        endpoint = (settings.mcp_order_endpoint or "").strip()
        if endpoint:
            try:
                provider = MCPOrderProvider(
                    endpoint=endpoint,
                    api_key=settings.mcp_order_api_key,
                    timeout_sec=settings.mcp_order_timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] MCPOrderProvider 初始化失败，回退 NullOrderStatusProvider: {exc}")
                provider = NullOrderStatusProvider()
        _order_status_skill = OrderStatusSkill(provider=provider)
    return _order_status_skill


def _get_case_intake_skill() -> CaseIntakeSkill:
    global _case_intake_skill
    if _case_intake_skill is None:
        # 每次进程启动创建一次；状态读写由 skill 内 store（Redis 或内存）负责。
        _case_intake_skill = CaseIntakeSkill()
    return _case_intake_skill


@dataclass
class PipelineDebug:
    """评测用调试信息，不直接暴露给线上 API。"""

    route_needs_rag: bool
    route_domain_hint: str
    route_reason: str
    route_confidence: float = 0.0
    route_strategy: str = ""
    """路由置信度是否低于 ``pipeline_route_low_confidence_threshold``（与检索闸门分列）。"""
    route_low_confidence: bool = False
    """检索后闸门：未尝试检索为空串；否则 ok / no_passing_chunks / insufficient_chunks。"""
    post_retrieval_gate: str = ""
    # 是否使用了用户图片视觉摘要（与路由/检索闸门分列观察）
    used_visual_context: bool = False
    visual_context_chars: int = 0
    top_k: int = 0
    context_chars: int = 0
    context_chunk_count: int = 0
    retrieval: RetrievalTrace | None = None
    # ReAct 多轮检索统计
    react_iterations: int = 0
    react_queries: list[str] = field(default_factory=list)
    # Query 改写结果（空字符串表示未改写或没有历史）
    rewritten_query: str = ""


@dataclass
class PipelineResult:
    """流水线一次执行的输出。"""

    answer: str
    images: list[str] = field(default_factory=list)
    route_reason: str = ""
    debug: PipelineDebug = field(default_factory=lambda: PipelineDebug(False, "", ""))


class ChatPipeline:
    """应用主流程。"""

    def __init__(
        self,
        router: QuestionRouter | None = None,
        retriever: VectorRetriever | None = None,
        generator: Qwen2Generator | None = None,
        vision: VisionInterpreter | None = None,
        online_review_skill: OnlineReviewSkill | None = None,
        order_status_skill: OrderStatusSkill | None = None,
        case_intake_skill: CaseIntakeSkill | None = None,
    ) -> None:
        self.router = router or QuestionRouter()
        self.retriever = retriever or VectorRetriever()
        self.generator = generator or Qwen2Generator()
        self.vision = vision or VisionInterpreter()
        self.online_review_skill = online_review_skill or _get_online_review_skill()
        self.order_status_skill = order_status_skill or _get_order_status_skill()
        self.case_intake_skill = case_intake_skill or _get_case_intake_skill()

    def run(
        self, question: str, images: list[str], *, session_id: str | None = None
    ) -> PipelineResult:
        """执行流水线。

        说明：
        - 若有 `images`，会先做轻量视觉摘要，并作为路由/检索/生成的补充上下文。
        - 若提供 `session_id`，会读取历史对话以增强检索 query 和 Prompt，
          并在生成后保存本轮问答。
        """
        from app.services.conversation_store import get_conversation_store

        store = get_conversation_store() if session_id else None
        conversation_history, enrichment = self._load_conversation_context(store, session_id)
        visual_context = self.vision.summarize_images(question, images)
        rewritten_query = self._maybe_rewrite_query(question, enrichment)
        search_question = rewritten_query or question
        routing_query = self._compose_multimodal_query(search_question, visual_context)

        force_case_intake, cancelled_case_intake = self._resolve_case_intake_sticky_state(
            session_id=session_id,
            question=question,
        )
        if cancelled_case_intake:
            return self._build_case_intake_cancelled_result(
                session_id=session_id,
                question=question,
                images=images,
                store=store,
                visual_context=visual_context,
                rewritten_query=rewritten_query,
            )

        decision = self._route_decision(routing_query, force_case_intake=force_case_intake)
        route_low_confidence = (
            decision.confidence < settings.pipeline_route_low_confidence_threshold
        )
        used_visual_context = bool(visual_context)
        top_k = 4
        skill_case_intake_result = self._maybe_run_case_intake_branch(
            decision=decision,
            question=question,
            images=images,
            session_id=session_id,
            conversation_history=conversation_history,
            enrichment=enrichment,
            store=store,
            top_k=top_k,
            route_low_confidence=route_low_confidence,
            used_visual_context=used_visual_context,
            visual_context=visual_context,
            rewritten_query=rewritten_query,
        )
        if skill_case_intake_result is not None:
            return skill_case_intake_result

        order_status_result = self._maybe_run_order_status_branch(
            decision=decision,
            question=question,
            images=images,
            session_id=session_id,
            enrichment=enrichment,
            conversation_history=conversation_history,
            store=store,
            top_k=top_k,
            route_low_confidence=route_low_confidence,
            used_visual_context=used_visual_context,
            visual_context=visual_context,
            rewritten_query=rewritten_query,
        )
        if order_status_result is not None:
            return order_status_result

        web_review_result = self._maybe_run_web_review_branch(
            decision=decision,
            question=question,
            images=images,
            session_id=session_id,
            enrichment=enrichment,
            conversation_history=conversation_history,
            store=store,
            top_k=top_k,
            route_low_confidence=route_low_confidence,
            used_visual_context=used_visual_context,
            visual_context=visual_context,
            rewritten_query=rewritten_query,
        )
        if web_review_result is not None:
            return web_review_result

        if not decision.needs_rag:
            return self._run_no_rag_branch(
                decision=decision,
                question=question,
                images=images,
                session_id=session_id,
                conversation_history=conversation_history,
                store=store,
                top_k=top_k,
                route_low_confidence=route_low_confidence,
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                rewritten_query=rewritten_query,
            )

        chunks, filter_context, react_iterations, react_queries = self._retrieve_context(
            routing_query
        )

        if not chunks:
            post_retrieval_gate = "no_passing_chunks"
        elif not filter_context:
            post_retrieval_gate = "no_passing_chunks"
        elif len(filter_context) < settings.pipeline_min_filtered_chunks_for_rag:
            post_retrieval_gate = "insufficient_chunks"
        else:
            post_retrieval_gate = "ok"

        retrieval_trace = self.retriever.build_trace(
            query=routing_query,
            top_k=top_k,
            raw_chunks=chunks,
            filtered_chunks=filter_context,
        )

        prompt, context_block, context_chunk_count, image_ref_map = self._build_rag_prompt(
            question=question,
            decision=decision,
            post_retrieval_gate=post_retrieval_gate,
            route_low_confidence=route_low_confidence,
            visual_context=visual_context,
            conversation_history=conversation_history,
            filter_context=filter_context,
            react_iterations=react_iterations,
        )

        answer = self.generator.generate(prompt)
        answer, images = finalize_answer_images(answer, image_ref_map)
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=images,
            )
        return PipelineResult(
            answer=answer,
            images=images,
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block=context_block,
                context_chunk_count=context_chunk_count,
                retrieval_trace=retrieval_trace,
                route_low_confidence=route_low_confidence,
                post_retrieval_gate=post_retrieval_gate,
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                react_iterations=react_iterations,
                react_queries=react_queries,
                rewritten_query=rewritten_query,
            ),
        )

    @staticmethod
    def _load_conversation_context(store, session_id: str | None) -> tuple[str, str]:
        if not store:
            return "", ""
        return store.format_history(session_id), store.format_enrichment(session_id)

    @staticmethod
    def _maybe_rewrite_query(question: str, enrichment: str) -> str:
        if not (enrichment and settings.query_rewrite_enabled):
            return ""
        rewritten = _get_rewriter().rewrite(question, enrichment)
        return rewritten or ""

    def _resolve_case_intake_sticky_state(
        self,
        *,
        session_id: str | None,
        question: str,
    ) -> tuple[bool, bool]:
        force_case_intake = False
        cancelled_case_intake = False
        if session_id and settings.case_intake_skill_enabled:
            if self.case_intake_skill.try_cancel_intake(session_id, question):
                cancelled_case_intake = True
                force_case_intake = False
            elif self.case_intake_skill.has_pending_intake(session_id):
                force_case_intake = True
        return force_case_intake, cancelled_case_intake

    def _route_decision(self, routing_query: str, *, force_case_intake: bool) -> RouteDecision:
        if force_case_intake:
            return RouteDecision(
                needs_rag=False,
                domain_hint="case_intake",
                reason="工单信息收集中，继续补充",
                confidence=0.92,
                strategy="case_intake_sticky",
            )
        if settings.router_llm_enabled:
            return _get_llm_router().decide(routing_query)
        return self.router.decide(routing_query)

    def _build_case_intake_cancelled_result(
        self,
        *,
        session_id: str | None,
        question: str,
        images: list[str],
        store,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult:
        reply = "好的，已为你取消当前工单草稿。如需报修或退款，请随时再说明情况。"
        if store and session_id:
            store.add_turn(
                session_id,
                question=question,
                answer=reply,
                user_images=images,
                answer_images=[],
            )
        cancel_decision = RouteDecision(
            needs_rag=False,
            domain_hint="case_intake",
            reason="用户取消工单草稿",
            confidence=1.0,
            strategy="case_intake_cancelled",
        )
        return PipelineResult(
            answer=reply,
            images=[],
            route_reason=cancel_decision.reason,
            debug=self._build_debug(
                decision=cancel_decision,
                top_k=4,
                context_block="",
                context_chunk_count=0,
                retrieval_trace=None,
                route_low_confidence=False,
                post_retrieval_gate="ok",
                used_visual_context=bool(visual_context),
                visual_context=visual_context,
                react_iterations=0,
                react_queries=[],
                rewritten_query=rewritten_query,
            ),
        )

    def _maybe_run_case_intake_branch(
        self,
        *,
        decision: RouteDecision,
        question: str,
        images: list[str],
        session_id: str | None,
        conversation_history: str,
        enrichment: str,
        store,
        top_k: int,
        route_low_confidence: bool,
        used_visual_context: bool,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult | None:
        if not (
            decision.domain_hint == "case_intake"
            and settings.case_intake_skill_enabled
        ):
            return None
        skill_result = self.case_intake_skill.run(
            question=question,
            session_id=(session_id or ""),
            conversation_history=conversation_history,
            enrichment=enrichment,
        )
        if store:
            store.add_turn(
                session_id, question=question, answer=skill_result.reply_text,
                user_images=images, answer_images=[],
            )
        return PipelineResult(
            answer=skill_result.reply_text,
            images=[],
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block=skill_result.context_block,
                context_chunk_count=1 if skill_result.context_block else 0,
                retrieval_trace=None,
                route_low_confidence=route_low_confidence,
                post_retrieval_gate=("ok" if skill_result.completed else "insufficient_chunks"),
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                react_iterations=0,
                react_queries=[],
                rewritten_query=rewritten_query,
            ),
        )

    def _maybe_run_order_status_branch(
        self,
        *,
        decision: RouteDecision,
        question: str,
        images: list[str],
        session_id: str | None,
        enrichment: str,
        conversation_history: str,
        store,
        top_k: int,
        route_low_confidence: bool,
        used_visual_context: bool,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult | None:
        if not (
            decision.domain_hint == "order_status"
            and settings.order_status_skill_enabled
        ):
            return None
        skill_result = self.order_status_skill.run(
            question=question,
            enrichment=enrichment,
            top_k=settings.order_status_top_k,
        )
        context_block = skill_result.context_block if skill_result.ok else ""
        evidence_status = "ok" if skill_result.ok else "no_passing_chunks"
        prompt = compose_generation_prompt(
            PromptContext(
                question=question,
                need_rag=False,
                domain_hint="order_status",
                context_block=context_block,
                route_reason=decision.reason,
                evidence_status=evidence_status,
                route_low_confidence=route_low_confidence,
                visual_context=visual_context,
                conversation_history=conversation_history,
            )
        )
        answer = self.generator.generate(prompt)
        answer, result_images = finalize_answer_images(answer, {})
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=result_images,
            )
        return PipelineResult(
            answer=answer,
            images=result_images,
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block=context_block,
                context_chunk_count=1 if context_block else 0,
                retrieval_trace=None,
                route_low_confidence=route_low_confidence,
                post_retrieval_gate=evidence_status,
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                react_iterations=0,
                react_queries=[],
                rewritten_query=rewritten_query,
            ),
        )

    def _maybe_run_web_review_branch(
        self,
        *,
        decision: RouteDecision,
        question: str,
        images: list[str],
        session_id: str | None,
        enrichment: str,
        conversation_history: str,
        store,
        top_k: int,
        route_low_confidence: bool,
        used_visual_context: bool,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult | None:
        if not (
            decision.domain_hint == "web_review"
            and settings.online_review_skill_enabled
        ):
            return None
        skill_result = self.online_review_skill.run(
            question=question,
            enrichment=enrichment,
            top_k=settings.online_review_top_k,
        )
        context_block = skill_result.context_block if skill_result.ok else ""
        evidence_status = "ok" if skill_result.ok else "no_passing_chunks"
        prompt = compose_generation_prompt(
            PromptContext(
                question=question,
                need_rag=False,
                domain_hint="web_review",
                context_block=context_block,
                route_reason=decision.reason,
                evidence_status=evidence_status,
                route_low_confidence=route_low_confidence,
                visual_context=visual_context,
                conversation_history=conversation_history,
            )
        )
        answer = self.generator.generate(prompt)
        answer, result_images = finalize_answer_images(answer, {})
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=result_images,
            )
        return PipelineResult(
            answer=answer,
            images=result_images,
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block=context_block,
                context_chunk_count=1 if context_block else 0,
                retrieval_trace=None,
                route_low_confidence=route_low_confidence,
                post_retrieval_gate=evidence_status,
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                react_iterations=0,
                react_queries=[],
                rewritten_query=rewritten_query,
            ),
        )

    def _run_no_rag_branch(
        self,
        *,
        decision: RouteDecision,
        question: str,
        images: list[str],
        session_id: str | None,
        conversation_history: str,
        store,
        top_k: int,
        route_low_confidence: bool,
        used_visual_context: bool,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult:
        prompt = compose_generation_prompt(
            PromptContext(
                question=question,
                need_rag=False,
                domain_hint=decision.domain_hint,
                context_block="",
                route_reason=decision.reason,
                evidence_status="ok",
                route_low_confidence=route_low_confidence,
                visual_context=visual_context,
                conversation_history=conversation_history,
            )
        )
        answer = self.generator.generate(prompt)
        answer, result_images = finalize_answer_images(answer, {})
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=result_images,
            )
        return PipelineResult(
            answer=answer,
            images=result_images,
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block="",
                context_chunk_count=0,
                retrieval_trace=None,
                route_low_confidence=route_low_confidence,
                post_retrieval_gate="",
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                react_iterations=0,
                react_queries=[],
                rewritten_query=rewritten_query,
            ),
        )

    def _retrieve_context(
        self, routing_query: str
    ) -> tuple[list, list, int, list[str]]:
        if settings.react_enabled:
            ms_result = _get_react_agent().collect_evidence(routing_query)
            filter_context = ms_result.all_filtered_chunks
            return (
                filter_context,
                filter_context,
                ms_result.iterations,
                ms_result.search_queries,
            )
        manual_name = query_construction(routing_query)
        chunks = self.retriever.retrieve(
            routing_query, top_k=6, manual_name=manual_name or None
        )
        return chunks, retriever_context_filter(chunks), 1, []

    def _build_rag_prompt(
        self,
        *,
        question: str,
        decision: RouteDecision,
        post_retrieval_gate: str,
        route_low_confidence: bool,
        visual_context: str,
        conversation_history: str,
        filter_context: list,
        react_iterations: int,
    ) -> tuple[str, str, int, dict[str, str]]:
        if post_retrieval_gate != "ok":
            prompt = compose_generation_prompt(
                PromptContext(
                    question=question,
                    need_rag=False,
                    domain_hint=decision.domain_hint,
                    context_block="",
                    route_reason=decision.reason,
                    evidence_status=post_retrieval_gate,
                    route_low_confidence=route_low_confidence,
                    visual_context=visual_context,
                    conversation_history=conversation_history,
                )
            )
            return prompt, "", len(filter_context), {}

        multimodal_context = build_multimodal_context_block(filter_context)
        context_block = multimodal_context.context_block
        prompt = compose_generation_prompt(
            PromptContext(
                question=question,
                need_rag=True,
                domain_hint=decision.domain_hint,
                context_block=context_block,
                route_reason=decision.reason,
                evidence_status="ok",
                route_low_confidence=route_low_confidence,
                visual_context=visual_context,
                react_multi_evidence=react_iterations > 1,
                conversation_history=conversation_history,
            )
        )
        return (
            prompt,
            context_block,
            len(filter_context),
            multimodal_context.image_ref_map,
        )

    @staticmethod
    def _build_debug(
        *,
        decision: RouteDecision,
        top_k: int,
        context_block: str,
        context_chunk_count: int,
        retrieval_trace: RetrievalTrace | None,
        route_low_confidence: bool,
        post_retrieval_gate: str,
        used_visual_context: bool,
        visual_context: str,
        react_iterations: int = 0,
        react_queries: list[str] | None = None,
        rewritten_query: str = "",
    ) -> PipelineDebug:
        return PipelineDebug(
            route_needs_rag=decision.needs_rag,
            route_domain_hint=decision.domain_hint,
            route_reason=decision.reason,
            route_confidence=decision.confidence,
            route_strategy=decision.strategy,
            route_low_confidence=route_low_confidence,
            post_retrieval_gate=post_retrieval_gate,
            used_visual_context=used_visual_context,
            visual_context_chars=len(visual_context),
            top_k=top_k,
            context_chars=len(context_block),
            context_chunk_count=context_chunk_count,
            retrieval=retrieval_trace,
            react_iterations=react_iterations,
            react_queries=react_queries or [],
            rewritten_query=rewritten_query,
        )

    @staticmethod
    def _collect_image_ids(chunks: list) -> list[str]:
        """从检索结果中收集图片 ID，去重保序。"""
        seen: set[str] = set()
        result: list[str] = []
        for chunk in chunks:
            for img_id in chunk.image_ids:
                if img_id not in seen:
                    seen.add(img_id)
                    result.append(img_id)
        return result

    @staticmethod
    def _compose_multimodal_query(question: str, visual_context: str) -> str:
        """把视觉摘要补到查询文本，供路由/query_construction/检索使用。"""
        vc = (visual_context or "").strip()
        if not vc:
            return question
        return (
            f"{question.strip()}\n\n"
            "【用户上传图片的视觉摘要】\n"
            f"{vc}\n"
        )
