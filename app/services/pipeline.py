"""End-to-end chat pipeline orchestration.

Pipeline skeleton:
1) route/gate
2) retrieve if needed
3) build prompt context
4) generate answer
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.generator import Qwen2Generator
from app.services.retriever import VectorRetriever
from app.services.router import QuestionRouter
from app.utils.prompt_builder import build_context_block, build_generation_prompt


@dataclass
class PipelineResult:
    """Result object from pipeline execution."""

    answer: str
    route_reason: str


class ChatPipeline:
    """Main application pipeline.

    TODO (you):
    - Add confidence threshold fallback.
    - Add domain-specific answer templates.
    - Add safety/policy post-check before final response.
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
        """Execute the V1 pipeline.

        Note:
        - `images` is accepted to keep API-compatible with the competition.
        - In V1 scaffold, image payload is not yet used for retrieval/generation.
        """

        decision = self.router.decide(question)

        if decision.needs_rag:
            chunks = self.retriever.retrieve(question, top_k=4)
            context_block = build_context_block(chunks)
            prompt = build_generation_prompt(question, context_block, decision.domain_hint)
            answer = self.generator.generate(prompt)
            return PipelineResult(answer=answer, route_reason=decision.reason)

        # Customer-service fallback path (non-RAG in V1)
        # Keep this minimal and safe; you will enrich with policy KB later.
        fallback_answer = (
            "您好，已收到您的问题。当前我会先基于已有客服规则为您处理。"
            "若涉及具体订单或售后单据，请提供相关信息以便进一步核实。"
        )
        return PipelineResult(answer=fallback_answer, route_reason=decision.reason)
