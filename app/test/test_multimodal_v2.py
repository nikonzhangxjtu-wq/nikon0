"""多模态 V2：图片资产、结构理解、Embedding 与 Prompt 融合测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.ingestion import ManualChunk
from app.services.multimodal.catalog import build_manual_image_catalog
from app.services.multimodal.embeddings import JinaMultimodalEmbeddingClient
from app.services.multimodal.facts_store import ManualImageFactStore
from app.services.multimodal.understanding import (
    ManualImageUnderstanding,
    ManualImageUnderstandingCache,
    parse_understanding_json,
)
from app.services.retriever import ManualImageEvidence, RetrievedChunk, VectorRetriever
from app.utils.prompt_builder import build_multimodal_context_block, finalize_answer_images


def test_build_manual_image_catalog_reports_missing_orphan_and_parent_links(tmp_path: Path):
    image_dir = tmp_path / "插图"
    image_dir.mkdir()
    (image_dir / "panel_01.jpg").write_bytes(b"fake")
    (image_dir / "orphan.png").write_bytes(b"fake")
    (image_dir / "DUP.jpg").write_bytes(b"fake")
    (image_dir / "dup.png").write_bytes(b"fake")

    chunks = [
        ManualChunk(
            chunk_id="manual_0001",
            manual_name="manual",
            text="按键见 <IMG:panel_01>，缺失图见 <IMG:missing_01>",
            image_ids=["panel_01", "missing_01"],
        ),
        ManualChunk(
            chunk_id="manual_0002",
            manual_name="manual",
            text="再次引用 <IMG:panel_01>",
            image_ids=["panel_01"],
        ),
    ]

    catalog, report = build_manual_image_catalog(chunks=chunks, image_dir=image_dir)

    assert catalog["panel_01"].parent_chunk_ids == ["manual_0001", "manual_0002"]
    assert report.missing_images == ["missing_01"]
    assert report.orphan_images == ["DUP", "dup", "orphan"]
    assert report.case_conflicts == [["DUP", "dup"]]


def test_parse_understanding_json_keeps_structured_fields():
    raw = """
    ```json
    {
      "image_type": "button_panel",
      "ocr_text": ["Start", "Water"],
      "buttons": [{"name": "Start", "position": "右下角"}],
      "indicators": [{"name": "Water", "state": "闪烁"}],
      "parts": [],
      "operation_steps": ["短按 Start 启动"],
      "warnings": ["不要湿手操作"],
      "relations": ["Start 位于 Water 指示灯下方"]
    }
    ```
    """

    item = parse_understanding_json("panel_01", raw)

    assert item.image_type == "button_panel"
    assert item.ocr_text == ["Start", "Water"]
    assert item.buttons[0]["name"] == "Start"
    assert "Start" in item.to_semantic_text()
    assert "不要湿手操作" in item.to_prompt_text()


def test_understanding_cache_uses_file_hash(tmp_path: Path):
    image = tmp_path / "panel_01.jpg"
    image.write_bytes(b"v1")
    cache_file = tmp_path / "cache.json"
    cache = ManualImageUnderstandingCache(cache_file)
    item = ManualImageUnderstanding(image_id="panel_01", image_type="button_panel")

    cache.set(image_id="panel_01", image_path=image, understanding=item)
    assert cache.get(image_id="panel_01", image_path=image) == item

    image.write_bytes(b"v2")
    assert cache.get(image_id="panel_01", image_path=image) is None


def test_jina_embedding_client_builds_text_and_image_requests(tmp_path: Path):
    image = tmp_path / "panel_01.jpg"
    image.write_bytes(b"abc")
    fake_session = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
        ]
    }
    fake_session.post.return_value = fake_response
    client = JinaMultimodalEmbeddingClient(
        provider="jina_api",
        api_key="secret",
        model="jina-clip-v2",
        session=fake_session,
    )

    assert client.embed_text("启动键在哪里") == [0.1, 0.2, 0.3]
    text_payload = fake_session.post.call_args.kwargs["json"]
    assert text_payload["model"] == "jina-clip-v2"
    assert text_payload["input"] == ["启动键在哪里"]

    assert client.embed_image(image) == [0.1, 0.2, 0.3]
    image_payload = fake_session.post.call_args.kwargs["json"]
    assert image_payload["input"][0] == {
        "image": base64.b64encode(b"abc").decode("ascii"),
    }


def test_jina_embedding_client_requires_api_key():
    client = JinaMultimodalEmbeddingClient(provider="jina_api", api_key="")
    with pytest.raises(RuntimeError, match="JINA_API_KEY"):
        client.embed_text("hello")


def test_dashscope_embedding_client_builds_multimodal_payload(tmp_path: Path):
    image = tmp_path / "panel_01.jpg"
    image.write_bytes(b"abc")
    fake_session = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "output": {
            "embeddings": [
                {"embedding": [0.9, 0.8, 0.7], "type": "text"},
            ]
        }
    }
    fake_session.post.return_value = fake_response
    client = JinaMultimodalEmbeddingClient(
        provider="dashscope_multimodal",
        api_key="secret",
        model="qwen3-vl-embedding",
        endpoint="https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        session=fake_session,
    )

    assert client.embed_text("启动键在哪里") == [0.9, 0.8, 0.7]
    text_payload = fake_session.post.call_args.kwargs["json"]
    assert text_payload["model"] == "qwen3-vl-embedding"
    assert text_payload["input"]["contents"] == [{"text": "启动键在哪里"}]
    assert text_payload["parameters"]["dimension"] == 1024

    assert client.embed_image(image) == [0.9, 0.8, 0.7]
    image_payload = fake_session.post.call_args.kwargs["json"]
    assert image_payload["input"]["contents"][0]["image"].startswith("data:image/jpeg;base64,")


def test_multimodal_context_renders_image_evidence_and_keeps_pic_output():
    chunk = RetrievedChunk(
        chunk_id="manual_0001",
        manual_name="manual",
        score=0.9,
        text="启动键见 <IMG:panel_01>",
        image_ids=["panel_01"],
        image_evidence=[
            ManualImageEvidence(
                image_id="panel_01",
                image_type="button_panel",
                match_reason="OCR/实体命中：Start",
                prompt_text="类型：button_panel\n按钮：Start 位于右下角",
                score=0.88,
            )
        ],
    )

    ctx = build_multimodal_context_block([chunk])

    assert "[图片结构证据]" in ctx.context_block
    assert "OCR/实体命中：Start" in ctx.context_block
    assert "<IMG_1:panel_01>" in ctx.context_block
    answer, images = finalize_answer_images("请看 <IMG_1:panel_01>", ctx.image_ref_map)
    assert answer == "请看 <PIC>"
    assert images == ["panel_01"]


def test_attached_image_facts_are_added_from_retrieved_chunk_images(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    image = tmp_path / "panel_01.jpg"
    image.write_bytes(b"fake")
    cache = ManualImageUnderstandingCache(cache_file)
    cache.set(
        image_id="panel_01",
        image_path=image,
        parent_context_text="按键见图示",
        understanding=ManualImageUnderstanding(
            image_id="panel_01",
            image_type="button_panel",
            context_intent="说明启动键位置",
            ocr_text=["Start"],
            buttons=[{"name": "Start", "position": "右下角"}],
        ),
    )
    fact_store = ManualImageFactStore(cache_file)
    retriever = object.__new__(VectorRetriever)
    retriever._image_fact_store = fact_store

    chunks = [
        RetrievedChunk(
            chunk_id="manual_0001",
            manual_name="manual",
            score=0.9,
            text="启动键见 <IMG:panel_01>",
            image_ids=["panel_01"],
        )
    ]

    result = retriever._attach_image_evidence_from_chunks(chunks)

    assert result[0].image_evidence[0].image_id == "panel_01"
    assert result[0].image_evidence[0].match_reason == "文本命中片段附带图片"
    assert "图片意图: 说明启动键位置" in result[0].image_evidence[0].prompt_text
    assert "Start" in result[0].image_evidence[0].prompt_text
