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
from app.services.answer_postprocess import postprocess_answer
from app.services.context_assembler import ContextAssembler, ContextAssemblyTrace
from app.services.memory import MemoryManager, get_memory_manager
from app.services.memory.v3 import get_memory_manager_v3
from app.services.memory.v3.evidence_packet import TurnEvidencePacketBuilder
from app.services.memory.v3.read_planner import MemoryReadPlanner
from app.services.memory.v4 import get_memory_manager_v4
from app.services.memory.v4.reader import IssueReadPlanner
from app.services.skills.case_intake_skill import CaseIntakeSkill
from app.services.rag_skill.query_construction import query_construction
from app.services.retriever import RetrievalTrace, VectorRetriever, retriever_context_filter
from app.services.router import QuestionRouter, RouteDecision
from app.services.vision import VisionInterpreter
from app.utils.prompt_builder import build_multimodal_context_block, finalize_answer_images
from app.utils.prompts import PromptContext, compose_generation_prompt

_llm_router = None
_react_agent = None
_rewriter = None
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


def _get_case_intake_skill() -> CaseIntakeSkill:
    global _case_intake_skill
    if _case_intake_skill is None:
        provider = (settings.case_intake_provider or "local").strip().lower()
        if provider == "gateway":
            try:
                from app.services.skills.gateway_case_intake import GatewayCaseIntakeSkill

                _case_intake_skill = GatewayCaseIntakeSkill()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] GatewayCaseIntakeSkill 初始化失败，回退本地 CaseIntakeSkill: {exc}")
                _case_intake_skill = CaseIntakeSkill()
        else:
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
    retrieval_source_queries: list[str] = field(default_factory=list)
    manual_name_decisions: list[str] = field(default_factory=list)
    context_original_tokens: int = 0
    context_final_tokens: int = 0
    context_compression_notes: list[str] = field(default_factory=list)
    evidence_block_count: int = 0
    critical_fact_count: int = 0
    compression_fallback_count: int = 0
    verifier_failed_reasons: list[str] = field(default_factory=list)
    preserved_block_types: list[str] = field(default_factory=list)
    memory_trace: dict = field(default_factory=dict)


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
        case_intake_skill: CaseIntakeSkill | None = None,
        context_assembler: ContextAssembler | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.router = router or QuestionRouter()
        self.retriever = retriever or VectorRetriever()
        self.generator = generator or Qwen2Generator()
        self.vision = vision or VisionInterpreter()
        self.case_intake_skill = case_intake_skill or _get_case_intake_skill()
        self.context_assembler = context_assembler or ContextAssembler()
        self.memory_manager = memory_manager or get_memory_manager()
        self.memory_read_planner_v3 = MemoryReadPlanner()
        self.memory_read_planner_v4 = IssueReadPlanner()
        self._last_context_trace = ContextAssemblyTrace(0, 0)

    def run(
        self,
        question: str,
        images: list[str],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
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
        memory_context = self._read_memory_context(
            question=question,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history,
        )
        visual_context = self.vision.summarize_images(question, images)
        rewritten_query = self._maybe_rewrite_query(question, enrichment, memory_context)
        search_question = rewritten_query or question
        routing_query = self._compose_multimodal_query(search_question, visual_context)

        force_case_intake, cancelled_case_intake = self._resolve_case_intake_sticky_state(
            session_id=session_id,
            question=question,
        )
        if cancelled_case_intake:
            return self._build_case_intake_cancelled_result(
                session_id=session_id,
                user_id=user_id,
                question=question,
                images=images,
                store=store,
                memory_context=memory_context,
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
            user_id=user_id,
            conversation_history=conversation_history,
            memory_context=memory_context,
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

        if not decision.needs_rag:
            return self._run_no_rag_branch(
                decision=decision,
                question=question,
                images=images,
                session_id=session_id,
                user_id=user_id,
                conversation_history=conversation_history,
                memory_context=memory_context,
                store=store,
                top_k=top_k,
                route_low_confidence=route_low_confidence,
                used_visual_context=used_visual_context,
                visual_context=visual_context,
                rewritten_query=rewritten_query,
            )

        chunks, filter_context, react_iterations, react_queries, manual_name_decisions = self._retrieve_context(
            routing_query,
            image_inputs=images,
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
            source_queries=react_queries or [routing_query],
            manual_name_decisions=manual_name_decisions,
        )

        prompt, context_block, context_chunk_count, image_ref_map = self._build_rag_prompt(
            question=question,
            decision=decision,
            post_retrieval_gate=post_retrieval_gate,
            route_low_confidence=route_low_confidence,
            visual_context=visual_context,
            conversation_history=conversation_history,
            memory_context=memory_context,
            filter_context=filter_context,
            react_iterations=react_iterations,
        )

        answer = self.generator.generate(prompt)
        answer, images = finalize_answer_images(answer, image_ref_map)
        answer = postprocess_answer(answer)
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=images,
            )
        self._record_memory_turn(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            visual_context=visual_context,
            context_block=context_block,
            conversation_history=conversation_history,
            memory_context_used=memory_context,
            branch_name="rag_manual",
            route_domain_hint=decision.domain_hint,
            route_needs_rag=decision.needs_rag,
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
                manual_name_decisions=manual_name_decisions,
                context_trace=self._last_context_trace,
            ),
        )

    @staticmethod
    def _load_conversation_context(store, session_id: str | None) -> tuple[str, str]:
        if not store:
            return "", ""
        return store.format_history(session_id), store.format_enrichment(session_id)

    def _read_memory_context(
        self,
        *,
        question: str,
        session_id: str | None,
        user_id: str | None,
        conversation_history: str,
    ) -> str:
        if not settings.memory_enabled:
            return ""
        if (settings.memory_version or "v1").strip().lower() == "v3":
            request = self.memory_read_planner_v3.plan(
                session_id=session_id,
                user_id=user_id,
                question=question,
                recent_history=conversation_history,
                route_domain_hint=None,
            )
            return get_memory_manager_v3().read(request).render()
        if (settings.memory_version or "v1").strip().lower() == "v4":
            request = self.memory_read_planner_v4.plan(
                session_id=session_id,
                query=question,
            )
            return get_memory_manager_v4().read(request).render()
        return self.memory_manager.read(
            session_id=session_id,
            user_id=user_id,
            query=question,
        ).render()

    @staticmethod
    def _maybe_rewrite_query(question: str, enrichment: str, memory_context: str = "") -> str:
        if not ((enrichment or memory_context) and settings.query_rewrite_enabled):
            return ""
        rewritten = _get_rewriter().rewrite(question, enrichment, memory_context=memory_context)
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
        user_id: str | None,
        question: str,
        images: list[str],
        store,
        memory_context: str,
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
        self._record_memory_turn(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=reply,
            visual_context=visual_context,
            context_block=memory_context,
            memory_context_used=memory_context,
            branch_name="case_intake",
            route_domain_hint="case_intake",
            route_needs_rag=False,
            branch_result={"case_status": "cancelled"},
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
        user_id: str | None,
        conversation_history: str,
        memory_context: str,
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
        self._record_memory_turn(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=skill_result.reply_text,
            visual_context=visual_context,
            context_block=skill_result.context_block,
            conversation_history=conversation_history,
            memory_context_used=memory_context,
            branch_name="case_intake",
            route_domain_hint=decision.domain_hint,
            route_needs_rag=decision.needs_rag,
            branch_result={
                **(skill_result.ticket_payload or {}),
                "completed": skill_result.completed,
                "missing_slots": skill_result.missing_slots,
                "context_block": skill_result.context_block,
            },
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

    def _run_no_rag_branch(
        self,
        *,
        decision: RouteDecision,
        question: str,
        images: list[str],
        session_id: str | None,
        user_id: str | None,
        conversation_history: str,
        memory_context: str,
        store,
        top_k: int,
        route_low_confidence: bool,
        used_visual_context: bool,
        visual_context: str,
        rewritten_query: str,
    ) -> PipelineResult:
        prompt = self._compose_generation_prompt(
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
                memory_context=memory_context,
            )
        )
        answer = self.generator.generate(prompt)
        answer, result_images = finalize_answer_images(answer, {})
        answer = postprocess_answer(answer)
        if store:
            store.add_turn(
                session_id, question=question, answer=answer,
                user_images=images, answer_images=result_images,
            )
        self._record_memory_turn(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            visual_context=visual_context,
            context_block=memory_context,
            conversation_history=conversation_history,
            memory_context_used=memory_context,
            branch_name="no_rag",
            route_domain_hint=decision.domain_hint,
            route_needs_rag=decision.needs_rag,
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
                context_trace=self._last_context_trace,
            ),
        )

    def _retrieve_context(
        self, routing_query: str, *, image_inputs: list[str] | None = None
    ) -> tuple[list, list, int, list[str], list[str]]:
        if settings.react_enabled:
            ms_result = _get_react_agent().collect_evidence(routing_query)
            filter_context = ms_result.all_filtered_chunks
            return (
                filter_context,
                filter_context,
                ms_result.iterations,
                ms_result.search_queries,
                [],
            )
        source_queries = self._build_retrieval_queries(routing_query)
        raw_chunks: list = []
        manual_decisions: list[str] = []
        for query in source_queries:
            manual_name = query_construction(routing_query) or None
            manual_decisions.append(
                f"{query[:60]} => {manual_name or '<all>'}"
            )
            raw_chunks.extend(
                self.retriever.retrieve(
                    query,
                    top_k=6,
                    manual_name=manual_name,
                    image_inputs=image_inputs or [],
                )
            )
        chunks = self._deduplicate_chunks(raw_chunks)
        return chunks, retriever_context_filter(chunks), len(source_queries), source_queries, manual_decisions

    @staticmethod
    def _build_retrieval_queries(routing_query: str) -> list[str]:
        """对复杂/多子问生成少量检索 query，简单题保持单 query。"""
        import re

        q = (routing_query or "").strip()
        if not q:
            return []

        signals = 0
        signals += q.count("\n")
        signals += len(re.findall(r"[？?]", q))
        signals += len(re.findall(r"同时|另外|还有|并且|而且|以及|分别|哪些|流程|步骤|前[一二三四五六七八九十\d]+", q))
        if signals < settings.retrieval_multi_query_min_signals:
            return [q]

        parts: list[str] = []
        for raw in re.split(r"[\n。；;？?]+", q):
            item = raw.strip(" ，,、：:\"'“”")
            if not item:
                continue
            if len(item) < 6 and not re.search(r"[A-Za-z]{3,}", item):
                continue
            parts.append(item)

        # 对长句中的并列诉求再粗拆一次，便于召回不同证据。
        expanded: list[str] = []
        for part in parts or [q]:
            subparts = re.split(r"(?:同时|另外|还有|并且|而且|以及)", part)
            if len(subparts) <= 1:
                expanded.append(part)
                continue
            for sub in subparts:
                sub = sub.strip(" ，,、：")
                if len(sub) >= 6:
                    expanded.append(sub)

        queries: list[str] = []
        seen: set[str] = set()
        for item in [q, *expanded]:
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            queries.append(item)
            if len(queries) >= max(1, settings.retrieval_multi_query_max_queries):
                break
        return queries or [q]

    def _build_rag_prompt(
        self,
        *,
        question: str,
        decision: RouteDecision,
        post_retrieval_gate: str,
        route_low_confidence: bool,
        visual_context: str,
        conversation_history: str,
        memory_context: str,
        filter_context: list,
        react_iterations: int,
    ) -> tuple[str, str, int, dict[str, str]]:
        if post_retrieval_gate != "ok":
            prompt = self._compose_generation_prompt(
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
                    memory_context=memory_context,
                )
            )
            return prompt, "", len(filter_context), {}

        multimodal_context = build_multimodal_context_block(filter_context)
        context_block = multimodal_context.context_block
        prompt = self._compose_generation_prompt(
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
                memory_context=memory_context,
            )
        )
        return (
            prompt,
            context_block,
            len(filter_context),
            multimodal_context.image_ref_map,
        )

    def _compose_generation_prompt(self, ctx: PromptContext) -> str:
        """统一入口：生成前先做上下文预算与压缩。

        这样 prompt builder 仍然只关心业务表达，ContextAssembler 专注决定
        哪些上下文应该进入模型窗口。
        """
        assembled = self.context_assembler.assemble(ctx)
        self._last_context_trace = assembled.trace
        return compose_generation_prompt(assembled.context)

    def _record_memory_turn(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        question: str,
        answer: str,
        visual_context: str,
        context_block: str,
        conversation_history: str = "",
        memory_context_used: str = "",
        branch_name: str = "",
        route_domain_hint: str = "",
        route_needs_rag: bool = False,
        branch_result: dict | None = None,
        tool_results: list[dict] | None = None,
    ) -> None:
        if not (settings.memory_enabled and session_id):
            return
        if (settings.memory_version or "v1").strip().lower() == "v3":
            packet = TurnEvidencePacketBuilder.build(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                route_domain_hint=route_domain_hint,
                route_needs_rag=route_needs_rag,
                branch_name=branch_name,
                recent_history=conversation_history,
                memory_context_used=memory_context_used,
                visual_context=visual_context,
                rag_context=context_block if branch_name == "rag_manual" else "",
                branch_result=branch_result,
                tool_results=tool_results or [],
            )
            get_memory_manager_v3().observe_and_write(packet)
            return
        if (settings.memory_version or "v1").strip().lower() == "v4":
            packet = TurnEvidencePacketBuilder.build(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                route_domain_hint=route_domain_hint,
                route_needs_rag=route_needs_rag,
                branch_name=branch_name,
                recent_history=conversation_history,
                memory_context_used=memory_context_used,
                visual_context=visual_context,
                rag_context=context_block if branch_name == "rag_manual" else "",
                branch_result=branch_result,
                tool_results=tool_results or [],
            )
            get_memory_manager_v4().observe_and_write(packet)
            return
        self.memory_manager.write_turn(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            visual_context=visual_context,
            context_block=context_block,
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
        manual_name_decisions: list[str] | None = None,
        context_trace: ContextAssemblyTrace | None = None,
    ) -> PipelineDebug:
        trace = context_trace or ContextAssemblyTrace(0, 0)
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
            retrieval_source_queries=react_queries or [],
            manual_name_decisions=manual_name_decisions or [],
            context_original_tokens=trace.original_tokens,
            context_final_tokens=trace.final_tokens,
            context_compression_notes=trace.notes,
            evidence_block_count=trace.evidence_block_count,
            critical_fact_count=trace.critical_fact_count,
            compression_fallback_count=trace.compression_fallback_count,
            verifier_failed_reasons=trace.verifier_failed_reasons,
            preserved_block_types=trace.preserved_block_types,
        )

    @staticmethod
    def _deduplicate_chunks(chunks: list) -> list:
        seen: dict[str, object] = {}
        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", "")
            if not chunk_id:
                continue
            prev = seen.get(chunk_id)
            if prev is None or getattr(chunk, "score", 0.0) > getattr(prev, "score", 0.0):
                seen[chunk_id] = chunk
        out = list(seen.values())
        out.sort(key=lambda c: getattr(c, "score", 0.0), reverse=True)
        return out

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
