from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.online_review_skill import OnlineReviewSkill, ReviewHit
from app.services.order_status_skill import OrderStatusHit, OrderStatusSkill
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
        "怎么安装", top_k=6, manual_name=None
    )
    mock_filter.assert_called_once_with(raw_chunks)
    mock_context.assert_called_once_with(filtered_chunks)
    mock_compose.assert_called_once()

    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.need_rag is True
    assert prompt_ctx.domain_hint == "manual"
    assert prompt_ctx.context_block == "上下文块"
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
    assert prompt_ctx.context_block == ""
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


def test_run_web_review_skill_branch():
    class FakeReviewProvider:
        def search_reviews(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
            _ = (query, top_k)
            return [
                ReviewHit(
                    title="电钻用户反馈汇总",
                    url="https://example.com/review1",
                    snippet="动力足，续航不错，但噪音偏大。",
                    source="example",
                )
            ]

    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="web_review",
        reason="命中口碑查询",
        confidence=0.91,
        strategy="heuristic_web_review",
    )
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "这是口碑总结答案"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    skill = OnlineReviewSkill(provider=FakeReviewProvider())
    with patch.object(
        skill,
        "_call_llm",
        return_value='{"summary":"整体评价偏正向","pros":["动力足"],"cons":["噪音大"],"controversies":[],"advice":"家庭偶尔使用可考虑","confidence":"medium"}',
    ):
        pipeline = ChatPipeline(
            router=router,
            retriever=retriever,
            generator=generator,
            vision=vision,
            online_review_skill=skill,
        )

        with (
            patch("app.services.pipeline.settings.router_llm_enabled", False),
            patch("app.services.pipeline.settings.online_review_skill_enabled", True),
            patch("app.services.pipeline.compose_generation_prompt", return_value="WEB_REVIEW_PROMPT") as mock_compose,
        ):
            result = pipeline.run("这款电钻网上评价怎么样", images=[])

    retriever.retrieve.assert_not_called()
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.domain_hint == "web_review"
    assert prompt_ctx.need_rag is False
    assert "[口碑评价摘要]" in prompt_ctx.context_block
    assert "总体倾向" in prompt_ctx.context_block

    generator.generate.assert_called_once_with("WEB_REVIEW_PROMPT")
    assert result.answer == "这是口碑总结答案"
    assert result.debug.route_domain_hint == "web_review"
    assert result.debug.post_retrieval_gate == "ok"
    assert result.debug.context_chunk_count == 1


def test_run_order_status_skill_branch():
    class FakeOrderProvider:
        def search_order_status(self, query: str, *, top_k: int = 3) -> list[OrderStatusHit]:
            _ = (query, top_k)
            return [
                OrderStatusHit(
                    order_id="OD20260507001",
                    status="已发货",
                    logistics_status="运输中（杭州分拨中心）",
                    eta="2026-05-09",
                    updated_at="2026-05-07 19:20",
                    can_refund="可联系客服拦截退款",
                    note="建议先等待下一节点更新",
                )
            ]

    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=False,
        domain_hint="order_status",
        reason="命中订单进度查询",
        confidence=0.93,
        strategy="heuristic_order_status",
    )
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "订单状态答复"
    vision = MagicMock()
    vision.summarize_images.return_value = ""
    skill = OrderStatusSkill(provider=FakeOrderProvider())

    pipeline = ChatPipeline(
        router=router,
        retriever=retriever,
        generator=generator,
        vision=vision,
        order_status_skill=skill,
    )
    with (
        patch("app.services.pipeline.settings.router_llm_enabled", False),
        patch("app.services.pipeline.settings.order_status_skill_enabled", True),
        patch("app.services.pipeline.compose_generation_prompt", return_value="ORDER_STATUS_PROMPT") as mock_compose,
    ):
        result = pipeline.run("请帮我查订单 OD20260507001 到哪了", images=[])

    retriever.retrieve.assert_not_called()
    mock_compose.assert_called_once()
    prompt_ctx = mock_compose.call_args.args[0]
    assert prompt_ctx.domain_hint == "order_status"
    assert prompt_ctx.need_rag is False
    assert "[订单进度信息]" in prompt_ctx.context_block
    assert "OD20260507001" in prompt_ctx.context_block

    generator.generate.assert_called_once_with("ORDER_STATUS_PROMPT")
    assert result.answer == "订单状态答复"
    assert result.debug.route_domain_hint == "order_status"
    assert result.debug.post_retrieval_gate == "ok"
    assert result.debug.context_chunk_count == 1


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
    test_run_web_review_skill_branch()
    test_run_order_status_skill_branch()
    test_run_case_intake_skill_branch()
    print("[OK] test_pipeline passed")
