from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.pipeline import ChatPipeline
from app.services.retriever import RetrievalTrace, RetrievedChunk
from app.services.router import RouteDecision
from app.utils.prompt_builder import MultimodalContextBlock


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
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router, retriever=retriever, generator=generator, vision=vision
    )

    with (
        patch("app.services.pipeline.settings.react_enabled", False),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.query_construction", return_value=None),
        patch("app.services.pipeline.retriever_context_filter", return_value=filtered_chunks) as mock_filter,
        patch(
            "app.services.pipeline.build_multimodal_context_block",
            return_value=MultimodalContextBlock(context_block="上下文块", image_ref_map={}),
        ) as mock_context,
        patch("app.services.pipeline.compose_generation_prompt", return_value="PROMPT") as mock_compose,
    ):
        result = pipeline.run("怎么安装", images=[])

    router.decide.assert_called_once_with("怎么安装")
    vision.summarize_images.assert_called_once_with("怎么安装", [])
    retriever.retrieve.assert_called_once_with(
        "怎么安装", top_k=6, manual_name=None, image_inputs=[]
    )
    mock_filter.assert_called_once_with(raw_chunks)
    mock_context.assert_called_once_with(filtered_chunks)
    mock_compose.assert_called_once()

    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.need_rag is True
    assert prompt_ctx.domain_hint == "manual"
    assert "上下文块" in prompt_ctx.context_block
    assert "[关键事实]" in prompt_ctx.context_block
    assert prompt_ctx.question == "怎么安装"
    assert prompt_ctx.evidence_status == "ok"
    assert prompt_ctx.route_low_confidence is False
    assert result.debug.post_retrieval_gate == "ok"
    assert result.debug.route_low_confidence is False

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
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router, retriever=retriever, generator=generator, vision=vision
    )

    with (
        patch("app.services.pipeline.settings.react_enabled", False),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="NO_RAG_PROMPT") as mock_compose,
    ):
        result = pipeline.run("我要退款", images=["img1"])

    retriever.retrieve.assert_not_called()
    vision.summarize_images.assert_called_once_with("我要退款", ["img1"])
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.need_rag is False
    assert prompt_ctx.domain_hint == "customer_service"
    assert "退款" in prompt_ctx.context_block
    assert prompt_ctx.question == "我要退款"
    assert prompt_ctx.evidence_status == "ok"
    assert result.debug.post_retrieval_gate == ""

    generator.generate.assert_called_once_with("NO_RAG_PROMPT")
    assert result.answer == "客服答案"
    assert result.route_reason == "命中客服类关键词"
    assert result.debug.route_needs_rag is False
    assert result.debug.route_domain_hint == "customer_service"
    assert result.debug.route_confidence == 0.75
    assert result.debug.context_chunk_count == 0
    assert result.debug.retrieval is None


def test_pipeline_reads_and_writes_memory_context():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="customer_service",
        reason="客服问题",
        confidence=0.9,
        strategy="test",
    )
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "请继续提供订单号"
    vision = MagicMock()
    vision.summarize_images.return_value = ""
    memory_manager = MagicMock()
    memory_bundle = MagicMock()
    memory_bundle.render.return_value = "[记忆]\n产品/型号: AC900"
    memory_manager.read.return_value = memory_bundle

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
        memory_manager=memory_manager,
    )

    with (
        patch("app.services.pipeline.settings.memory_enabled", True),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="NO_RAG_PROMPT") as mock_compose,
    ):
        pipeline.run("我要退款", images=[], session_id="sid_mem", user_id="user-1")

    memory_manager.read.assert_called_once_with(
        session_id="sid_mem",
        user_id="user-1",
        query="我要退款",
    )
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.memory_context == "[记忆]\n产品/型号: AC900"
    memory_manager.write_turn.assert_called_once()
    assert memory_manager.write_turn.call_args.kwargs["session_id"] == "sid_mem"
    assert memory_manager.write_turn.call_args.kwargs["user_id"] == "user-1"


def test_legacy_web_review_domain_falls_back_to_no_rag_branch():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="web_review",
        reason="旧口碑分支标记",
        confidence=0.91,
        strategy="heuristic_web_review",
    )
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "这是口碑总结答案"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
    )

    with (
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="NO_RAG_PROMPT") as mock_compose,
    ):
        result = pipeline.run("这款电钻网上评价怎么样", images=[])

    retriever.retrieve.assert_not_called()
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.domain_hint == "web_review"
    assert prompt_ctx.need_rag is False
    assert "[口碑评价摘要]" not in prompt_ctx.context_block

    generator.generate.assert_called_once_with("NO_RAG_PROMPT")
    assert result.answer == "这是口碑总结答案"
    assert result.debug.route_domain_hint == "web_review"
    assert result.debug.post_retrieval_gate == ""
    assert result.debug.context_chunk_count == 0


def test_legacy_order_status_domain_falls_back_to_no_rag_branch():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="order_status",
        reason="旧订单分支标记",
        confidence=0.93,
        strategy="heuristic_order_status",
    )
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "订单状态答复"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
    )
    with (
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.compose_generation_prompt", return_value="NO_RAG_PROMPT") as mock_compose,
    ):
        result = pipeline.run("请帮我查订单 OD20260507001 到哪了", images=[])

    retriever.retrieve.assert_not_called()
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.domain_hint == "order_status"
    assert prompt_ctx.need_rag is False
    assert "[订单进度信息]" not in prompt_ctx.context_block

    generator.generate.assert_called_once_with("NO_RAG_PROMPT")
    assert result.answer == "订单状态答复"
    assert result.debug.route_domain_hint == "order_status"
    assert result.debug.post_retrieval_gate == ""
    assert result.debug.context_chunk_count == 0


def test_legacy_order_status_domain_with_needs_rag_uses_manual_rag_branch():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=True,
        domain_hint="order_status",
        reason="旧订单分支标记但需要检索",
        confidence=0.5,
        strategy="legacy_tool_domain",
    )

    retriever = MagicMock()
    raw_chunks = [
        RetrievedChunk(chunk_id="c1", text="订单相关手册说明", score=0.9, manual_name="m"),
    ]
    retriever.retrieve.return_value = raw_chunks
    retriever.build_trace.return_value = RetrievalTrace(
        query="订单状态是否需要查手册",
        top_k=4,
        raw_count=1,
        filtered_count=1,
    )
    generator = MagicMock()
    generator.generate.return_value = "检索后答案"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
    )
    with (
        patch("app.services.pipeline.settings.react_enabled", False),
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.query_construction", return_value=None),
        patch("app.services.pipeline.retriever_context_filter", return_value=raw_chunks),
        patch(
            "app.services.pipeline.build_multimodal_context_block",
            return_value=MultimodalContextBlock(context_block="上下文块", image_ref_map={}),
        ),
        patch("app.services.pipeline.compose_generation_prompt", return_value="RAG_PROMPT") as mock_compose,
    ):
        result = pipeline.run("订单状态是否需要查手册", images=[])

    retriever.retrieve.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.domain_hint == "order_status"
    assert prompt_ctx.need_rag is True
    assert result.debug.post_retrieval_gate == "ok"


def test_run_case_intake_skill_branch():
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="case_intake",
        reason="命中售后受理意图",
        confidence=0.9,
        strategy="heuristic_case_intake",
    )
    retriever = MagicMock()
    generator = MagicMock()
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
    )
    with (
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.settings.case_intake_skill_enabled", True),
    ):
        result = pipeline.run("电钻坏了，帮我报修", images=[], session_id="sid_case_1")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()
    assert "请补充以下信息" in result.answer or "已为你完成售后受理信息收集" in result.answer
    assert result.debug.route_domain_hint == "case_intake"
    assert result.debug.context_chunk_count in (0, 1)


if __name__ == "__main__":
    test_run_rag_branch_uses_retrieval_and_generation()
    test_run_no_rag_branch_skips_retrieval()
    test_legacy_web_review_domain_falls_back_to_no_rag_branch()
    test_legacy_order_status_domain_falls_back_to_no_rag_branch()
    test_legacy_order_status_domain_with_needs_rag_uses_manual_rag_branch()
    test_run_case_intake_skill_branch()
    print("[OK] test_pipeline passed")
