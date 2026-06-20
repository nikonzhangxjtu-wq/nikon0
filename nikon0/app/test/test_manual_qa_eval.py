from __future__ import annotations

from nikon0.app.schemas.knowledge import KnowledgeRequest
from nikon0.eval.judge_results import judge_item
from nikon0.knowledge.product_resolver import ProductResolver
from nikon0.knowledge.runtime import StructuredManualBackend


def test_judge_reads_correct_manual_from_tool_evidence_not_answer_filename() -> None:
    item = {
        "case_id": "qa-1",
        "category": "product_support",
        "expected": {"acceptable_skills": ["product_support"], "answer_must_contain": []},
        "metadata": {"source_manual": "手册/Airfryer.txt"},
    }
    result = {
        "case_id": "qa-1",
        "category": "product_support",
        "message": "question",
        "answer": "Here is a grounded answer without a filename.",
        "selected_skill": "product_support",
        "selection_source": "model",
        "actions": [
            {
                "kind": "tool",
                "name": "product-support.search_product_manual",
                "payload": {"evidence": [{"payload": {"manual_name": "Airfryer"}}]},
            }
        ],
    }

    judgement = judge_item(item, result)

    assert judgement["checks"]["evidence_from_correct_manual"] is True
    assert judgement["checks"]["has_evidence"] is True


def test_expanded_catalog_scopes_english_microwave_to_its_manual() -> None:
    resolution = ProductResolver().resolve("How do I set the child lock on the microwave?")

    assert resolution.product_id == "microwave_otr"
    assert resolution.manual_names == ("Microwave_OTR",)


def test_structured_manual_backend_reads_json_content_and_respects_scope(tmp_path) -> None:
    (tmp_path / "Airfryer.txt").write_text(
        '["# Cleaning\\nClean the basket with warm water after use.", []]',
        encoding="utf-8",
    )
    (tmp_path / "Microwave_OTR.txt").write_text(
        '["# Cleaning\\nClean the grease filter monthly.", []]',
        encoding="utf-8",
    )

    result = StructuredManualBackend(tmp_path).query(
        KnowledgeRequest(query="clean basket", allowed_manual_names=["Airfryer"], max_evidence=3)
    )

    assert result.evidence
    assert {item.payload["manual_name"] for item in result.evidence} == {"Airfryer"}
    assert "warm water" in result.evidence[0].text
