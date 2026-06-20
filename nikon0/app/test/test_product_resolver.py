"""Tests for product resolution and disclosure behavior."""

from __future__ import annotations

import asyncio

from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.memory import SessionIssueMemory
from nikon0.app.schemas.trace import ExecutionTrace
from nikon0.knowledge.product_resolver import (
    ProductCatalog,
    ProductResolver,
    apply_product_disclosure,
    build_disambiguation_answer,
    build_product_disclosure_prefix,
)
from nikon0.knowledge.runtime import KnowledgeRuntime, StructuredManualBackend
from nikon0.skills.product_support import ProductSupportSkill


def _camera_catalog() -> ProductCatalog:
    return ProductCatalog.load()


def test_strong_signal_selects_dslr_and_requires_disclosure() -> None:
    resolver = ProductResolver(_camera_catalog())
    resolution = resolver.resolve("相机怎么装 EF-S 镜头？")

    assert resolution.status == "resolved"
    assert resolution.source == "strong_signal"
    assert resolution.product_id == "canon_dslr"
    assert resolution.manual_names == ("DSLR_Camera",)
    assert resolution.disclose_default_product is True
    assert resolution.reason == "inferred product from technical signal"
    prefix = build_product_disclosure_prefix(resolution)
    assert "Canon EOS 单反相机" in prefix
    assert "EF-S" in prefix


def test_strong_signal_skips_disclosure_when_user_named_dslr() -> None:
    resolver = ProductResolver(_camera_catalog())
    resolution = resolver.resolve("单反相机电池多久能充满？")

    assert resolution.status == "resolved"
    assert resolution.product_id == "canon_dslr"
    assert resolution.disclose_default_product is False
    assert resolution.reason == "user named product identity"
    assert build_product_disclosure_prefix(resolution) == ""


def test_explicit_dishwasher_identity_outranks_shared_door_seal_term() -> None:
    resolver = ProductResolver(_camera_catalog())

    resolution = resolver.resolve("洗碗机门封条换了还是漏水")

    assert resolution.status == "resolved"
    assert resolution.product_id == "dishwasher"
    assert resolution.display_name == "洗碗机"
    assert resolution.reason == "user named product identity"


def test_weak_camera_query_requires_disambiguation() -> None:
    resolver = ProductResolver(_camera_catalog())
    resolution = resolver.resolve("相机电池大概多久能充满？")

    assert resolution.status == "disambiguation_required"
    assert len(resolution.candidates) == 2
    prompt = build_disambiguation_answer(resolution.candidates)
    assert "1." in prompt
    assert "2." in prompt


def test_user_choice_after_disambiguation() -> None:
    resolver = ProductResolver(_camera_catalog())
    session_state = {
        "product_support": {
            "disambiguation_pending": True,
            "disambiguation_candidates": ["canon_dslr", "instax_square"],
        }
    }
    resolution = resolver.resolve("2", session_state=session_state)

    assert resolution.status == "resolved"
    assert resolution.source == "user_choice"
    assert resolution.product_id == "instax_square"
    assert resolution.disclose_default_product is False


def test_apply_product_disclosure_prefixes_answer() -> None:
    resolver = ProductResolver(_camera_catalog())
    resolution = resolver.resolve("相机 CF 卡怎么格式化？")
    answer = apply_product_disclosure("请先进入设置菜单。", resolution)

    assert answer.startswith("根据您问题中的")
    assert "Canon EOS 单反相机" in answer
    assert answer.endswith("请先进入设置菜单。")


def _support_context(session_id: str, message: str, *, flat_state: dict | None = None) -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id=session_id, message=message),
        session_state=SessionIssueMemory(session_id=session_id, flat_state=flat_state or {}),
        trace=ExecutionTrace(trace_id="test-trace", session_id=session_id, user_message=message),
    )


def test_product_support_skill_resolves_ambiguous_camera_from_retrieved_evidence(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "DSLR_Camera.txt").write_text(
        "相机电池 BP-511A 和 BP-514 充电约 100 分钟；BP-511 和 BP-512 约 90 分钟。",
        encoding="utf-8",
    )
    (manual_dir / "相机手册.txt").write_text("拍立得相纸盒安装后才能打印照片。", encoding="utf-8")

    skill = ProductSupportSkill(
        knowledge_runtime=KnowledgeRuntime(StructuredManualBackend(manual_dir)),
    )
    context = _support_context("camera-disambig-s1", "相机电池大概多久能充满？")
    result = asyncio.run(skill.run(context))

    assert result.status == "success"
    assert result.answer_draft.startswith("根据召回的手册证据，")
    assert "当前默认按" in result.answer_draft
    assert "Canon EOS 单反相机" in result.answer_draft
    assert "100 分钟" in result.answer_draft
    assert context.trace.knowledge_calls[0]["product_resolution"]["source"] == "retrieval_evidence"
    assert context.trace.knowledge_calls[0]["product_resolution"]["product_id"] == "canon_dslr"
    assert [item["tool_name"] for item in context.trace.tool_calls] == [
        "resolve_product",
        "search_product_manual",
        "validate_answer_grounding",
    ]
    assert result.state_updates[0].value["selected_product_id"] == "canon_dslr"


def test_product_support_skill_asks_when_retrieval_cannot_disambiguate(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "DSLR_Camera.txt").write_text("相机电池充电说明。", encoding="utf-8")
    (manual_dir / "相机手册.txt").write_text("相机电池充电说明。", encoding="utf-8")

    skill = ProductSupportSkill(
        knowledge_runtime=KnowledgeRuntime(StructuredManualBackend(manual_dir)),
    )
    context = _support_context("camera-still-ambiguous-s1", "相机电池怎么充电？")
    result = asyncio.run(skill.run(context))

    assert result.status == "needs_more_info"
    assert "请先确认具体型号" in result.answer_draft
    assert [item["tool_name"] for item in context.trace.tool_calls] == [
        "resolve_product",
        "search_product_manual",
    ]
    assert context.trace.knowledge_calls[0]["product_resolution"]["status"] == "disambiguation_required"


def test_product_support_skill_auto_selects_dslr_with_disclosure(tmp_path) -> None:
    manual_dir = tmp_path / "manuals"
    manual_dir.mkdir()
    (manual_dir / "DSLR_Camera.txt").write_text(
        "安装 EF-S 镜头时，将白色安装标记对准机身白点，转动直到咔嗒锁定。",
        encoding="utf-8",
    )
    (manual_dir / "相机手册.txt").write_text("安装肩带时请将腕带套在手腕上。", encoding="utf-8")

    skill = ProductSupportSkill(
        knowledge_runtime=KnowledgeRuntime(StructuredManualBackend(manual_dir)),
    )
    context = _support_context("camera-strong-s1", "相机怎么装 EF-S 镜头？")
    result = asyncio.run(skill.run(context))

    assert result.status == "success"
    assert "当前默认按" in result.answer_draft
    assert "EF-S" in result.answer_draft or "咔嗒" in result.answer_draft
    assert context.trace.knowledge_calls[0]["product_resolution"]["product_id"] == "canon_dslr"
    assert context.trace.knowledge_calls[0]["evidence_count"] >= 1
    assert result.state_updates[0].value["selected_product_id"] == "canon_dslr"
