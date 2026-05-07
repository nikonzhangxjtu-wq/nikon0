"""测试多模态图片对齐 —— V3：<IMG:xxx> 直接嵌入文本。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.pipeline import ChatPipeline
from app.services.retriever import RetrievalTrace, RetrievedChunk
from app.services.router import RouteDecision
from app.utils.prompt_builder import (
    build_multimodal_context_block,
    finalize_answer_images,
)


def test_build_multimodal_context_block_from_embedded_img_tags():
    """chunk 文本已含 <IMG:xxx>，context_block 应保留原样并构建 image_ref_map。"""
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            manual_name="drill",
            score=0.9,
            text="充电中<IMG:drill10_04>已充满<IMG:drill10_05>过热延迟<IMG:drill10_06>",
            image_ids=["drill10_04", "drill10_05", "drill10_06"],
        )
    ]

    ctx = build_multimodal_context_block(chunks)

    # 正文保留 <IMG:xxx> 原始格式
    assert "<IMG:drill10_04>" in ctx.context_block
    assert "<IMG:drill10_05>" in ctx.context_block
    assert "<IMG:drill10_06>" in ctx.context_block
    # image_ref_map 按 token 映射
    assert ctx.image_ref_map["IMG_1"] == "drill10_04"
    assert ctx.image_ref_map["IMG_2"] == "drill10_05"
    assert ctx.image_ref_map["IMG_3"] == "drill10_06"
    assert [r.image_id for r in ctx.image_refs] == [
        "drill10_04", "drill10_05", "drill10_06",
    ]


def test_finalize_mapped_references():
    """模型回显 <IMG_n:image_id> 格式，经 image_ref_map 校验转换。"""
    answer, images = finalize_answer_images(
        "充电中<IMG_1:drill10_04>，过热延迟<IMG_3:drill10_06>",
        {"IMG_1": "drill10_04", "IMG_2": "drill10_05", "IMG_3": "drill10_06"},
    )
    assert answer == "充电中<PIC>，过热延迟<PIC>"
    assert images == ["drill10_04", "drill10_06"]


def test_finalize_direct_references():
    """模型从正文直接复制 <IMG:xxx> 格式。"""
    answer, images = finalize_answer_images(
        "步骤<IMG:drill10_04>完成后<IMG:drill10_05>",
        {"IMG_1": "drill10_04", "IMG_2": "drill10_05"},
    )
    assert answer == "步骤<PIC>完成后<PIC>"
    assert images == ["drill10_04", "drill10_05"]


def test_finalize_mixed_references_preserves_text_order():
    """混合格式时按文本出现顺序返回。"""
    answer, images = finalize_answer_images(
        "参见<IMG:drill10_04>和<IMG_2:drill10_05>了解",
        {"IMG_1": "drill10_04", "IMG_2": "drill10_05"},
    )
    assert images == ["drill10_04", "drill10_05"]


def test_finalize_deduplicates_repeated_refs():
    """同一图片多次引用只收集一次。"""
    answer, images = finalize_answer_images(
        "参见<IMG:drill10_04>和<IMG:drill10_04>",
        {},
    )
    assert answer == "参见<PIC>和<PIC>"
    assert images == ["drill10_04"]


def test_finalize_removes_unknown_bare_pic():
    """非法引用和裸 <PIC> 都应被清除。"""
    answer, images = finalize_answer_images(
        "未知图<IMG_999:bad>，裸图<PIC>，合法图<IMG_1:ok>",
        {"IMG_1": "ok"},
    )
    assert answer == "未知图，裸图，合法图<PIC>"
    assert images == ["ok"]


def test_pipeline_returns_only_images_referenced_by_answer():
    """端到端：含图 chunk → 模型引用部分图 → 只返回被引用的图。"""
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=True, domain_hint="manual", reason="命中关键词",
        confidence=0.9, strategy="heuristic_keyword",
    )

    # 新格式：<IMG:xxx> 已嵌入文本
    chunks = [
        RetrievedChunk(
            chunk_id="c1", manual_name="drill", score=0.9,
            text="充电中<IMG:drill10_04>已充满<IMG:drill10_05>过热延迟<IMG:drill10_06>",
            image_ids=["drill10_04", "drill10_05", "drill10_06"],
        )
    ]
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    retriever.build_trace.return_value = RetrievalTrace(
        query="充电灯怎么看", top_k=4, raw_count=1, filtered_count=1,
        score_threshold=0.3, top1_score=0.9,
        retrieved_chunk_ids=["c1"], filtered_chunk_ids=["c1"],
        retrieved_manual_names=["drill"], filtered_manual_names=["drill"],
    )

    generator = MagicMock()
    # 模型可能用 <IMG_1:xxx> 或 <IMG:xxx> 引用
    generator.generate.return_value = "充电中<IMG_1:drill10_04>；过热延迟<IMG:drill10_06>"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(router=router, retriever=retriever, generator=generator, vision=vision)

    with (
        patch("app.services.pipeline.settings.react_enabled", False),
        patch("app.services.pipeline.query_construction", return_value=None),
        patch("app.services.pipeline.retriever_context_filter", return_value=chunks),
    ):
        result = pipeline.run("充电灯怎么看", images=[])

    assert result.answer == "充电中<PIC>；过热延迟<PIC>"
    assert result.images == ["drill10_04", "drill10_06"]
    assert result.answer.count("<PIC>") == len(result.images)


def test_pipeline_returns_no_images_when_answer_has_no_image_refs():
    """无图回答返回空 images。"""
    router = MagicMock()
    router.decide.return_value = RouteDecision(
        needs_rag=True, domain_hint="manual", reason="命中关键词",
        confidence=0.9, strategy="heuristic_keyword",
    )
    chunks = [
        RetrievedChunk(
            chunk_id="c1", manual_name="drill", score=0.9,
            text="充电中<IMG:drill10_04>",
            image_ids=["drill10_04"],
        )
    ]
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    retriever.build_trace.return_value = RetrievalTrace(
        query="充电灯怎么看", top_k=4, raw_count=1, filtered_count=1,
        score_threshold=0.3, top1_score=0.9,
        retrieved_chunk_ids=["c1"], filtered_chunk_ids=["c1"],
        retrieved_manual_names=["drill"], filtered_manual_names=["drill"],
    )
    generator = MagicMock()
    generator.generate.return_value = "充电中表示电池正在充电。"
    vision = MagicMock()
    vision.summarize_images.return_value = ""

    pipeline = ChatPipeline(router=router, retriever=retriever, generator=generator, vision=vision)

    with (
        patch("app.services.pipeline.settings.react_enabled", False),
        patch("app.services.pipeline.query_construction", return_value=None),
        patch("app.services.pipeline.retriever_context_filter", return_value=chunks),
    ):
        result = pipeline.run("充电灯怎么看", images=[])

    assert result.answer == "充电中表示电池正在充电。"
    assert result.images == []


if __name__ == "__main__":
    test_build_multimodal_context_block_from_embedded_img_tags()
    test_finalize_mapped_references()
    test_finalize_direct_references()
    test_finalize_mixed_references_preserves_text_order()
    test_finalize_deduplicates_repeated_refs()
    test_finalize_removes_unknown_bare_pic()
    test_pipeline_returns_only_images_referenced_by_answer()
    test_pipeline_returns_no_images_when_answer_has_no_image_refs()
    print("[OK] test_multimodal_image_alignment passed")
