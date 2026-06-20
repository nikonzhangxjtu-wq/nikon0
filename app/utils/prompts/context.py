"""Prompt 组装所需的统一上下文。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    """一次生成所用的输入；新增字段时保持向后兼容默认值。"""

    question: str
    need_rag: bool
    domain_hint: str
    context_block: str = ""
    route_reason: str = ""
    # 检索后闸门：ok | no_passing_chunks | insufficient_chunks（与 PipelineDebug.post_retrieval_gate 一致）
    evidence_status: str = "ok"
    # 路由置信度是否低于 pipeline 阈值（与「无检索证据」分列，便于调参）
    route_low_confidence: bool = False
    # 用户上传图片的视觉摘要（由 VisionInterpreter 生成）。
    visual_context: str = ""
    # ReAct 多轮检索标记：True 表示上下文来自多轮合并去重，证据更全面。
    react_multi_evidence: bool = False
    # 多轮对话历史文本（已格式化，由 ConversationStore 生成）。
    conversation_history: str = ""
    # 分层记忆系统产出的短期会话事实、长期用户画像与情景记忆。
    memory_context: str = ""
