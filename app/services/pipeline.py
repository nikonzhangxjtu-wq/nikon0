"""端到端对话流水线编排。

骨架步骤：
1）路由 / RAG 门控
2）需要时检索
3）拼 prompt 上下文
4）生成答案
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.generator import Qwen2Generator
from app.services.rag_skill.query_construction import query_construction
from app.services.retriever import RetrievalTrace, VectorRetriever, retriever_context_filter
from app.services.router import QuestionRouter, RouteDecision
from app.utils.prompt_builder import build_context_block
from app.utils.prompts import PromptContext, compose_generation_prompt


@dataclass
class PipelineDebug:
    """评测用调试信息，不直接暴露给线上 API。"""

    route_needs_rag: bool
    route_domain_hint: str
    route_reason: str
    route_confidence: float = 0.0
    route_strategy: str = ""
    top_k: int = 0
    context_chars: int = 0
    context_chunk_count: int = 0
    retrieval: RetrievalTrace | None = None

@dataclass
class PipelineResult:
    """流水线一次执行的输出。"""

    answer: str
    route_reason: str
    debug: PipelineDebug = field(default_factory=lambda: PipelineDebug(False, "", ""))


class ChatPipeline:
    """应用主流程。

    TODO（你来补）：
    - 检索置信度阈值与兜底策略
    - 按领域（说明书 / 客服）使用不同回答模板
    - 最终回复前的安全/政策后处理
    """

    def __init__(
        self,
        router: QuestionRouter | None = None,
        retriever: VectorRetriever | None = None,
        generator: Qwen2Generator | None = None,
    ) -> None:
        self.router = router or QuestionRouter()
        self.retriever = retriever or VectorRetriever()
        self.generator = generator or Qwen2Generator()

    def run(self, question: str, images: list[str]) -> PipelineResult:
        """执行 V1 流水线。

        说明：
        - `images` 为与赛题接口对齐而保留；V1 骨架尚未用于检索/生成。
        """

        decision = self.router.decide(question)
        top_k = 4

        if decision.needs_rag:
            manual_name = query_construction(question)
            chunks = self.retriever.retrieve(
                question, top_k=top_k, manual_name=manual_name or None
            )
            filter_context = retriever_context_filter(chunks)
            context_block = build_context_block(filter_context)
            retrieval_trace = self.retriever.build_trace(
                query=question,
                top_k=top_k,
                raw_chunks=chunks,
                filtered_chunks=filter_context,
            )
            prompt = compose_generation_prompt(
                PromptContext(
                    question=question,
                    need_rag=True,
                    domain_hint=decision.domain_hint,
                    context_block=context_block,
                    route_reason=decision.reason,
                )
            )
            answer = self.generator.generate(prompt)
            return PipelineResult(
                answer=answer,
                route_reason=decision.reason,
                debug=self._build_debug(
                    decision=decision,
                    top_k=top_k,
                    context_block=context_block,
                    context_chunk_count=len(filter_context),
                    retrieval_trace=retrieval_trace,
                ),
            )

        prompt = compose_generation_prompt(
            PromptContext(
                question=question,
                need_rag=False,
                domain_hint=decision.domain_hint,
                context_block="",
                route_reason=decision.reason,
            )
        )
        answer = self.generator.generate(prompt)
        return PipelineResult(
            answer=answer,
            route_reason=decision.reason,
            debug=self._build_debug(
                decision=decision,
                top_k=top_k,
                context_block="",
                context_chunk_count=0,
                retrieval_trace=None,
            ),
        )

    @staticmethod
    def _build_debug(
        *,
        decision: RouteDecision,
        top_k: int,
        context_block: str,
        context_chunk_count: int,
        retrieval_trace: RetrievalTrace | None,
    ) -> PipelineDebug:
        return PipelineDebug(
            route_needs_rag=decision.needs_rag,
            route_domain_hint=decision.domain_hint,
            route_reason=decision.reason,
            route_confidence=decision.confidence,
            route_strategy=decision.strategy,
            top_k=top_k,
            context_chars=len(context_block),
            context_chunk_count=context_chunk_count,
            retrieval=retrieval_trace,
        )
