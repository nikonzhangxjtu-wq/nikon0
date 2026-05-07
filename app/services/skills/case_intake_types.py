"""Case Intake 共享类型（避免 redis_store 与 skill 循环导入）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CaseState:
    intent: str = "repair"
    slots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseIntakeResult:
    completed: bool
    reply_text: str
    missing_slots: list[str] = field(default_factory=list)
    ticket_payload: dict[str, str] = field(default_factory=dict)
    context_block: str = ""
    # ReAct：用户或模型主动中止；react_trace 便于调试/评测
    exited: bool = False
    react_trace: tuple[str, ...] = field(default_factory=tuple)
