"""演示「本地评价表 → Provider 检索 → OnlineReviewSkill」完整流程。

运行（建议带 -s 看打印；用当前解释器调用，避免未安装 pytest 脚本）::

    python -m pytest app/test/test_local_review_skill_flow.py -s

若提示 No module named pytest，先安装::

    pip install pytest

或只跑演示用例（打桩 LLM，CI 友好）::

    python -m pytest app/test/test_local_review_skill_flow.py::test_local_review_skill_demo_flow_prints -s

真实 Ollama（走 ``OnlineReviewSkill._call_llm`` → ``/api/chat``，需本机已启动 Ollama，
且 ``SIMPLE_LLM_MODEL`` 已 ``ollama pull``）::

    python -m pytest app/test/test_local_review_skill_flow.py::test_local_review_skill_demo_flow_real_ollama -s -m ollama
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.services.online_review_skill import OnlineReviewSkill
from app.services.skills.local_review_table import LocalReviewTable, LocalReviewTableProvider


def _ollama_http_reachable() -> bool:
    try:
        import requests

        base = (settings.ollama_base_url or "").rstrip("/")
        if not base:
            return False
        r = requests.get(f"{base}/api/tags", timeout=4)
        return bool(r.ok)
    except Exception:
        return False


def _print_section(title: str) -> None:
    bar = "=" * 56
    print(f"\n{bar}\n  {title}\n{bar}")


def test_local_review_skill_demo_flow_prints() -> None:
    """逐步打印：表数据 → 检索 → Skill 摘要（LLM 已打桩）。"""
    _print_section("1) 构建本地评价表（内置几条商品评论）")
    table = LocalReviewTable.default()
    rows = table.all_rows()
    print(f"共 {len(rows)} 条评论，涉及商品：")
    seen: set[str] = set()
    for r in rows:
        if r.product_name not in seen:
            seen.add(r.product_name)
            print(f"  - {r.product_name} ({r.product_id})")

    _print_section("2) Provider.search_reviews（模拟 OnlineReviewSkill 拼好的检索 query）")
    provider = LocalReviewTableProvider(table=table)
    user_question = "追觅扫地机器人 X1 网上评价怎么样"
    search_query = f"{user_question.strip()} 真实评价 口碑 优缺点"
    hits = provider.search_reviews(search_query, top_k=4)
    print(f"query: {search_query!r}")
    print(f"命中 {len(hits)} 条：")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] score={h.score:.1f} | {h.title}")
        print(f"       {h.snippet[:80]}...")

    _print_section("3) OnlineReviewSkill.run（打桩 _call_llm，跳过真实 Ollama）")
    skill = OnlineReviewSkill(provider=provider)
    fake_llm_json = (
        '{"summary":"整体偏正向，避障与清洁力受认可","pros":["扫得干净","地图准"],'
        '"cons":["集尘噪音","缠头发"],"controversies":[],"advice":"长发家庭留意主刷维护",'
        '"confidence":"medium"}'
    )
    from unittest.mock import patch

    with patch.object(skill, "_call_llm", return_value=fake_llm_json):
        result = skill.run(user_question, enrichment="", top_k=4, triggered=True)

    print(f"ok={result.ok} triggered={result.triggered}")
    print(f"search_query: {result.search_query!r}")
    if result.fallback_reason:
        print(f"fallback_reason: {result.fallback_reason}")
    print("--- context_block（会进生成 prompt）---")
    print(result.context_block[:600] + ("..." if len(result.context_block) > 600 else ""))

    assert result.ok is True
    assert result.triggered is True
    assert len(result.hits) >= 1
    assert "追觅" in result.context_block or "扫地" in result.context_block
    assert "[口碑评价摘要]" in result.context_block


@pytest.mark.ollama
def test_local_review_skill_demo_flow_real_ollama() -> None:
    """与 demo 相同流程，但调用真实 Ollama（不 mock ``_call_llm``）。"""
    if not _ollama_http_reachable():
        pytest.skip(
            f"Ollama 不可达: {settings.ollama_base_url!r}（请先启动 ollama serve，并 pull 模型 "
            f"{settings.simple_llm_model!r}）"
        )

    _print_section("真实 Ollama：本地表检索 + LLM 摘要")
    print(f"ollama_base_url={settings.ollama_base_url!r}")
    print(f"simple_llm_model={settings.simple_llm_model!r}")

    table = LocalReviewTable.default()
    provider = LocalReviewTableProvider(table=table)
    skill = OnlineReviewSkill(provider=provider)
    user_question = "追觅扫地机器人 X1 网上评价怎么样"
    result = skill.run(user_question, enrichment="", top_k=4, triggered=True)

    print(f"ok={result.ok} triggered={result.triggered} fallback={result.fallback_reason!r}")
    print("--- summary ---")
    print(result.summary or "(空，可能走启发式摘要)")
    print("--- context_block（节选）---")
    print(result.context_block[:900] + ("..." if len(result.context_block) > 900 else ""))

    assert result.ok is True
    assert result.triggered is True
    assert len(result.hits) >= 1
    assert "[口碑评价摘要]" in result.context_block


def test_local_review_table_merge_json(tmp_path: Path) -> None:
    """可选 JSON 与内置表合并后能检索到新商品。"""
    p = tmp_path / "extra.json"
    p.write_text(
        '[{"product_id":"z1","product_name":"测试商品Z","rating":5,"title":"好","snippet":"非常好用"}]',
        encoding="utf-8",
    )
    table = LocalReviewTable.from_json_file(p)
    prov = LocalReviewTableProvider(table=table)
    hits = prov.search_reviews("测试商品Z 真实评价 口碑", top_k=3)
    assert len(hits) >= 1
    assert any("测试商品Z" in h.title for h in hits)


def test_local_review_table_no_match_returns_empty() -> None:
    prov = LocalReviewTableProvider(table=LocalReviewTable.default())
    hits = prov.search_reviews("完全不存在的商品型号 XYZ999", top_k=8)
    assert hits == []


def test_online_review_skill_no_hits_when_query_irrelevant() -> None:
    skill = OnlineReviewSkill(provider=LocalReviewTableProvider(table=LocalReviewTable.default()))
    result = skill.run("完全不存在的商品型号 XYZ999 评价如何", triggered=True)
    assert result.ok is False
    assert result.fallback_reason == "no_hits"
