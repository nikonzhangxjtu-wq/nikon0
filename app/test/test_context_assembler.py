from __future__ import annotations

from app.services.context_assembler import ContextAssembler, estimate_tokens
from app.services.context.evidence_extractor import compress_evidence_blocks, parse_rag_context
from app.services.context.fact_extractor import extract_critical_facts
from app.services.context.verifier import verify_evidence
from app.utils.prompts.context import PromptContext


def test_estimate_tokens_handles_mixed_text():
    assert estimate_tokens("你好，world") > 0
    assert estimate_tokens("") == 0


def test_context_assembler_compresses_long_sections(monkeypatch):
    monkeypatch.setattr("app.services.context_assembler.settings.context_rag_token_budget", 80)
    monkeypatch.setattr("app.services.context_assembler.settings.context_history_token_budget", 40)
    monkeypatch.setattr("app.services.context_assembler.settings.context_visual_token_budget", 30)
    monkeypatch.setattr("app.services.context_assembler.settings.context_total_token_budget", 180)

    rag = "\n".join(
        [
            "[片段 1]",
            "chunk_id: c1",
            "手册: 空调手册",
            "正文: 如何清洁空调滤网。先关闭电源，取下滤网，用清水冲洗并晾干。" * 20,
            "[片段 2]",
            "正文: 与问题无关的普通说明。" * 20,
        ]
    )
    history = "\n".join([f"用户: 历史问题{i}\n助手: 很长的历史回答{i}" * 10 for i in range(8)])
    visual = "OCR文字：E2 故障码\n关键实体：空调, E2\n图片摘要：" + "屏幕显示故障。" * 30

    ctx = PromptContext(
        question="如何清洁空调滤网？图片里还有 E2",
        need_rag=True,
        domain_hint="manual",
        context_block=rag,
        conversation_history=history,
        visual_context=visual,
    )

    assembled = ContextAssembler().assemble(ctx)

    assert assembled.trace.final_tokens < assembled.trace.original_tokens
    assert assembled.trace.notes
    assert "空调" in assembled.context.context_block
    assert "E2" in assembled.context.visual_context


def test_context_assembler_compresses_memory_and_extracts_memory_facts(monkeypatch):
    monkeypatch.setattr("app.services.context_assembler.settings.context_rag_token_budget", 80)
    monkeypatch.setattr("app.services.context_assembler.settings.context_memory_token_budget", 45)
    monkeypatch.setattr("app.services.context_assembler.settings.context_total_token_budget", 220)

    memory_context = (
        "[记忆]\n"
        "[会话事实]\n"
        "订单号: 202605300001\n"
        "联系电话: 13800138000\n"
        "产品/型号: AC900\n"
        "故障码/状态码: E2\n"
        "[相关历史记忆]\n"
        + "用户此前反复反馈空调 E2 故障，已经断电重启但仍复现。" * 30
    )
    ctx = PromptContext(
        question="这个型号现在怎么处理？",
        need_rag=False,
        domain_hint="customer_service",
        memory_context=memory_context,
    )

    assembled = ContextAssembler().assemble(ctx)

    assert "202605300001" in assembled.context.context_block
    assert "13800138000" in assembled.context.context_block
    assert "AC900" in assembled.context.context_block
    assert "E2" in assembled.context.context_block
    assert len(assembled.context.memory_context) < len(memory_context)
    assert "memory:compressed" in assembled.trace.notes


def test_extracts_critical_facts():
    facts = extract_critical_facts(
        question="订单 202605300001 的 DCB101 电钻 E2 故障码闪烁，我要报修，手机号 13800138000",
        visual_context="OCR文字：DCB101 E2\n关键实体：电钻, 红灯闪烁",
    )
    assert "202605300001" in facts.order_ids
    assert "13800138000" in facts.phones
    assert "DCB101" in facts.fault_codes or "DCB101" in facts.product_models
    assert "报修" in facts.user_goals
    assert "红灯闪烁" in facts.visual_entities


def test_step_block_keeps_complete_steps_and_warning():
    context = """[片段 1]
chunk_id: drill_1
手册: 电钻手册
分数: 0.91
正文: 充电步骤：
1. 关闭电钻并取下电池。
2. 将电池插入 DCB101 充电器。
3. 观察指示灯状态。
4. 充满后取下电池。
注意：如果红灯持续闪烁，请停止充电并检查电池温度。
"""
    facts = extract_critical_facts(question="DCB101 电钻如何充电？", context_block=context)
    out, blocks, notes = compress_evidence_blocks(
        context_block=context,
        question="DCB101 电钻如何充电？",
        facts=facts,
        max_tokens=120,
    )
    assert notes == []
    assert blocks[0].block_type == "step_block"
    assert "1. 关闭电钻" in out
    assert "4. 充满后" in out
    assert "注意" in out


def test_table_block_keeps_header_hit_and_neighbor_rows():
    context = """[片段 1]
chunk_id: drill_light
手册: 电钻手册
分数: 0.88
正文: DCB101 指示灯状态：
红灯闪烁：电池温度异常。
黄灯闪烁：电量不足。
绿灯常亮：充电完成。
"""
    facts = extract_critical_facts(question="DCB101 指示灯红灯闪烁是什么意思？", context_block=context)
    out, blocks, _ = compress_evidence_blocks(
        context_block=context,
        question="DCB101 指示灯红灯闪烁是什么意思？",
        facts=facts,
        max_tokens=80,
    )
    assert blocks[0].block_type == "table_like"
    assert "DCB101 指示灯状态" in out
    assert "红灯闪烁" in out
    assert "黄灯闪烁" in out


def test_verifier_reports_broken_step_sequence():
    context = """[片段 1]
chunk_id: bad_step
手册: 测试手册
正文: 3. 第三步。
4. 第四步。
"""
    facts = extract_critical_facts(question="如何安装？", context_block=context)
    blocks = parse_rag_context(context, "如何安装？", facts)
    reasons = verify_evidence("如何安装？", facts, blocks)
    assert "broken_step_sequence" in reasons or "missing_step_block" in reasons
