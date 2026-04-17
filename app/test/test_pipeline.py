from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.pipeline import ChatPipeline
from app.services.retriever import RetrievalTrace, RetrievedChunk
from app.services.router import RouteDecision


def test_run_rag_branch_uses_retrieval_and_generation():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=True,
        domain_hint="manual",
        reason="命中说明书类关键词",
        confidence=0.8,
        strategy="heuristic_keyword",
    )

    retriever = MagicMock()
    raw_chunks = [
        RetrievedChunk(chunk_id="c1", text="t1", score=0.9, manual_name="m", image_ids=[]),
        RetrievedChunk(chunk_id="c2", text="t2", score=0.1, manual_name="m", image_ids=[]),
    ]
    filtered_chunks = [raw_chunks[0]]
    retriever.retrieve.return_value = raw_chunks
    retriever.build_trace.return_value = RetrievalTrace(
        query="怎么安装",
        top_k=4,
        raw_count=2,
        filtered_count=1,
        score_threshold=0.3,
        top1_score=0.9,
        retrieved_chunk_ids=["c1", "c2"],
        filtered_chunk_ids=["c1"],
        retrieved_manual_names=["m", "m"],
        filtered_manual_names=["m"],
    )

    generator = MagicMock()
    generator.generate.return_value = "最终答案"

    pipeline = ChatPipeline(router=router, retriever=retriever, generator=generator)

    with (
        patch("app.services.pipeline.retriever_context_filter", return_value=filtered_chunks) as mock_filter,
        patch("app.services.pipeline.build_context_block", return_value="上下文块") as mock_context,
        patch("app.services.pipeline.compose_generation_prompt", return_value="PROMPT") as mock_compose,
    ):
        result = pipeline.run("怎么安装", images=[])

    router.decide.assert_called_once_with("怎么安装")
    retriever.retrieve.assert_called_once_with("怎么安装", top_k=4)
    mock_filter.assert_called_once_with(raw_chunks)
    mock_context.assert_called_once_with(filtered_chunks)
    mock_compose.assert_called_once()

    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.need_rag is True
    assert prompt_ctx.domain_hint == "manual"
    assert prompt_ctx.context_block == "上下文块"
    assert prompt_ctx.question == "怎么安装"

    generator.generate.assert_called_once_with("PROMPT")
    assert result.answer == "最终答案"
    assert result.route_reason == "命中说明书类关键词"
    assert result.debug.route_needs_rag is True
    assert result.debug.route_domain_hint == "manual"
    assert result.debug.route_confidence == 0.8
    assert result.debug.route_strategy == "heuristic_keyword"
    assert result.debug.context_chars == len("上下文块")
    assert result.debug.context_chunk_count == 1
    assert result.debug.retrieval is not None
    assert result.debug.retrieval.retrieved_chunk_ids == ["c1", "c2"]
    assert result.debug.retrieval.filtered_chunk_ids == ["c1"]
    assert result.debug.retrieval.top1_score == 0.9


def test_run_no_rag_branch_skips_retrieval():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="customer_service",
        reason="命中客服类关键词",
        confidence=0.75,
        strategy="heuristic_keyword",
    )

    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "客服答案"

    pipeline = ChatPipeline(router=router, retriever=retriever, generator=generator)

    with patch("app.services.pipeline.compose_generation_prompt", return_value="NO_RAG_PROMPT") as mock_compose:
        result = pipeline.run("我要退款", images=["img1"])

    retriever.retrieve.assert_not_called()
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.need_rag is False
    assert prompt_ctx.domain_hint == "customer_service"
    assert prompt_ctx.context_block == ""
    assert prompt_ctx.question == "我要退款"

    generator.generate.assert_called_once_with("NO_RAG_PROMPT")
    assert result.answer == "客服答案"
    assert result.route_reason == "命中客服类关键词"
    assert result.debug.route_needs_rag is False
    assert result.debug.route_domain_hint == "customer_service"
    assert result.debug.route_confidence == 0.75
    assert result.debug.context_chunk_count == 0
    assert result.debug.retrieval is None


if __name__ == "__main__":
    test_run_rag_branch_uses_retrieval_and_generation()
    test_run_no_rag_branch_skips_retrieval()
    print("[OK] test_pipeline passed")
