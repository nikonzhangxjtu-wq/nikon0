"""多模态图片资产与理解结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ManualImageAsset:
    """一张手册插图及其与文本 chunk 的关系。"""

    image_id: str
    image_path: Path
    parent_chunk_ids: list[str] = field(default_factory=list)
    parent_manual_names: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    file_hash: str = ""


@dataclass(frozen=True)
class ManualImageAssetReport:
    """图片资产质量报告。"""

    missing_images: list[str] = field(default_factory=list)
    orphan_images: list[str] = field(default_factory=list)
    duplicate_ids: list[list[str]] = field(default_factory=list)
    case_conflicts: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class ManualImageUnderstanding:
    """VLM 对手册图片的结构化理解结果。

    这里刻意不把图片压成一段 caption：客服场景需要能检索和引用的结构事实，
    例如按键位置、指示灯状态、部件关系和操作箭头，而不是泛泛描述“这是一张设备图”。
    """

    image_id: str   # Blower_02
    image_type: str = "other"   
    # 图片所在父 chunk 的上下文。说明书插图常是箭头/编号/线稿，脱离上下文很难判断意图。
    parent_context_text: str = ""
    # 结合父文本后判断出的图片知识意图，例如“说明滤网拆卸与清洁”。
    context_intent: str = ""
    ocr_text: list[str] = field(default_factory=list)  # 图上可见文字
    buttons: list[dict] = field(default_factory=list) # 按键列表
    indicators: list[dict] = field(default_factory=list) # 指示灯/图标/提示
    parts: list[dict] = field(default_factory=list) # 部件/零件
    operation_steps: list[str] = field(default_factory=list) # 操作步骤
    warnings: list[str] = field(default_factory=list) # 安全警告
    relations: list[str] = field(default_factory=list) # 部位/箭头/空间关系
    raw_text: str = ""  # VLM 原始输出；JSON 解析失败时至少保留全文，便于排查

    def visual_entities(self) -> list[str]:
        """抽取适合精确匹配的实体词。"""
        entities: list[str] = []
        if self.context_intent:
            entities.append(self.context_intent)
        entities.extend(self.ocr_text)
        for group in (self.buttons, self.indicators, self.parts):
            for item in group:
                if not isinstance(item, dict):
                    continue
                for key in ("name", "label", "code", "state", "position", "function"):
                    value = str(item.get(key) or "").strip()
                    if value:
                        entities.append(value)
        for text in [*self.operation_steps, *self.warnings, *self.relations]:
            if text:
                entities.append(str(text))
        seen: set[str] = set()
        out: list[str] = []
        for entity in entities:
            entity = entity.strip()
            if entity and entity not in seen:
                seen.add(entity)
                out.append(entity)
        return out
    # 把结构化字段拼成一段带标签的中文短文本，这段文本会送去 Jina 等模型做 semantic_vector，用于「意思相近」的向量检索
    def to_semantic_text(self) -> str:
        """拼成用于 semantic_vector 的短文本。"""
        parts: list[str] = []
        if self.context_intent:
            parts.append(f"图片意图: {self.context_intent}")
        parts.append(f"图片类型: {self.image_type}")
        if self.parent_context_text:
            parts.append(f"父文本上下文: {self.parent_context_text}")
        if self.ocr_text:
            parts.append("OCR: " + " | ".join(self.ocr_text))
        if self.buttons:
            parts.append("按键: " + _dict_list_to_text(self.buttons))
        if self.indicators:
            parts.append("指示/提示: " + _dict_list_to_text(self.indicators))
        if self.parts:
            parts.append("部件: " + _dict_list_to_text(self.parts))
        if self.operation_steps:
            parts.append("操作步骤: " + " | ".join(self.operation_steps))
        if self.warnings:
            parts.append("警告/注意: " + " | ".join(self.warnings))
        if self.relations:
            parts.append("关系: " + " | ".join(self.relations))
        return "\n".join(parts)

    def to_prompt_text(self) -> str:
        """渲染给生成模型的图片结构证据。"""
        return self.to_semantic_text()

# 格式化工具
def _dict_list_to_text(items: list[dict]) -> str:
    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pairs = []
        for key, value in item.items():
            value_text = str(value).strip()
            if value_text:
                pairs.append(f"{key}={value_text}")
        if pairs:
            chunks.append("，".join(pairs))
    return " | ".join(chunks)
