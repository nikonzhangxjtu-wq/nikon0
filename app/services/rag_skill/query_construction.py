"""从用户问题推断 ``manual_name``（与 ``手册/*.txt`` 的 stem 一致）。

使用轻量 LLM（百炼 API 优先，Ollama 兜底）在**当前目录下已有手册名**中做
单选分类；解析失败或模型输出不在列表中时返回 ``""``，由检索端走全库。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许 ``python app/services/rag_skill/query_construction.py``：先把仓库根加入 path
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[3]
    _root_s = str(_repo_root)
    if _root_s not in sys.path:
        sys.path.insert(0, _root_s)

import json
import re
from dataclasses import dataclass

from app.core.config import settings
from app.services.llm_clients import chat_text


@dataclass(frozen=True)
class ManualNameDecision:
    """手册名识别结果。

    ``manual_name`` 只有在置信度达到阈值时才应用为检索过滤，避免错选手册。
    """

    manual_name: str
    confidence: float
    source: str
    reason: str = ""

    @property
    def should_filter(self) -> bool:
        return bool(self.manual_name) and self.confidence >= settings.manual_name_filter_min_confidence

_SYSTEM = """你是手册路由助手。根据用户问题中提到的**主体设备/产品**，判断最可能对应哪一本「操作手册」。

规则：
1. 你必须且只能从给定的「允许的手册名」列表中**原样**选出一个作为 manual_name。
2. 以问题中的**主体设备名词**（如"空调""水泵""冰箱"）为依据匹配手册，不要被功能描述词带偏。例如：
   - "如何使用空调的等离子净化功能？" → 主体是"空调"，选"空调手册"，不要因为"等离子净化"选"空气净化器手册"
   - "水泵安全排放燃油" → 主体是"水泵"，选"水泵手册"
3. 中文问题必须输出中文手册名（如"水泵手册"），严禁输出英文翻译（如"pump"）。纯英文问题才可输出英文手册名。
4. 若问题与任何一本都不相关、或无法判断、或属于通用客服而非具体某本手册，则 manual_name 置为空字符串。
5. 只输出一行合法 JSON，不要 Markdown、不要解释。格式严格为：{"manual_name":"..."} 其中值为列表中的某一字符串，或为空字符串。

"""


def _list_manual_stems(manual_dir: Path) -> list[str]:
    if not manual_dir.is_dir():
        return []
    return sorted({p.stem for p in manual_dir.glob("*.txt") if p.is_file()})


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_llm_manual_name(raw: str, stems: set[str]) -> str:
    try:
        data = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    name = data.get("manual_name")
    if name is None:
        return ""
    name = str(name).strip()
    if not name:
        return ""
    return name if name in stems else ""


def _keyword_fallback(question: str, stems: set[str]) -> ManualNameDecision:
    """Fallback: 用问题中的关键词匹配手册名（LLM 输出不在列表中时使用）。"""
    q_lower = question.lower()
    best = ""
    best_len = 0
    for stem in stems:
        candidates = [stem]
        # 剥离常见后缀作为搜索关键词
        for sfx in ("手册", "Manual", "manual"):
            if stem.endswith(sfx) and len(stem) > len(sfx):
                candidates.append(stem[: -len(sfx)])
        for kw in candidates:
            if kw.lower() in q_lower and len(kw) > best_len:
                best_len = len(kw)
                best = stem
    if not best:
        return ManualNameDecision("", 0.0, "keyword", "未命中手册名关键词")
    confidence = 0.95 if best_len >= 2 else 0.72
    return ManualNameDecision(best, confidence, "keyword", "问题文本直接命中手册名")


def query_construction_decision(question: str) -> ManualNameDecision:
    """返回带置信度的 ``manual_name`` 决策。"""
    q = question.strip()
    if not q:
        return ManualNameDecision("", 0.0, "empty", "空问题")

    manual_dir = Path(settings.manual_dir).expanduser().resolve()
    stems_list = _list_manual_stems(manual_dir)
    if not stems_list:
        return ManualNameDecision("", 0.0, "manual_dir", "手册目录为空")

    stems_set = set(stems_list)
    if len(stems_list) == 1:
        return ManualNameDecision(stems_list[0], 1.0, "single_manual", "仅有一本手册")

    keyword_decision = _keyword_fallback(q, stems_set)
    if keyword_decision.should_filter:
        return keyword_decision

    human = (
        "允许的手册名（JSON 数组，必须原样匹配其一）：\n"
        f"{json.dumps(stems_list, ensure_ascii=False)}\n\n"
        f"用户问题：\n{q}\n"
    )

    try:
        raw = chat_text(
            model=settings.simple_llm_model,
            messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": human},
            ],
            temperature=0.0,
            max_tokens=128,
            timeout=15,
        )
    except Exception:
        return keyword_decision

    result = _parse_llm_manual_name(raw, stems_set)
    if not result:
        return keyword_decision

    # LLM 的单选有帮助，但错选代价很高；只有当题面也弱命中该手册名时才给高置信。
    stem_core = result
    for sfx in ("手册", "Manual", "manual"):
        if stem_core.endswith(sfx) and len(stem_core) > len(sfx):
            stem_core = stem_core[: -len(sfx)]
            break
    if stem_core and stem_core.lower() in q.lower():
        confidence = 0.86
        reason = "LLM 选择且问题文本命中手册主体"
    else:
        confidence = 0.58
        reason = "LLM 选择但题面未直接命中手册主体，低置信全库检索"
    return ManualNameDecision(result, confidence, "llm", reason)


def query_construction(question: str) -> str:
    """返回高置信 ``manual_name``；低置信或无法判断则 ``""``。"""
    decision = query_construction_decision(question)
    return decision.manual_name if decision.should_filter else ""


if __name__ == "__main__":
    print(query_construction("无遥控器时如何操作空调"))
