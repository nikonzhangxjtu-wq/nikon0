"""联网评价 Skill：从外部来源收集口碑并生成结构化摘要。

设计目标：
1) 与 pipeline 解耦：本模块只负责「查询 -> 收集 -> 摘要 -> 返回证据块」。
2) Provider 可插拔：默认不联网；重新接入外部服务时仅需实现 `ReviewSearchProvider`。
3) 安全降级：无结果或 LLM 失败时返回可解释的 fallback_reason。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.core.config import settings
from app.services.llm_clients import chat_text


@dataclass(frozen=True)
class ReviewHit:
    """一条外部评价检索命中。"""

    title: str
    url: str
    snippet: str
    source: str = ""
    published_at: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class OnlineReviewResult:
    """OnlineReviewSkill 执行结果。"""

    ok: bool
    triggered: bool
    question: str
    search_query: str = ""
    summary: str = ""
    context_block: str = ""
    hits: list[ReviewHit] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    fallback_reason: str = ""


class ReviewSearchProvider(Protocol):
    """外部评价检索 Provider 协议。"""

    def search_reviews(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
        ...


class NullReviewProvider:
    """默认 provider：不做任何联网检索，用于本地占位和单元测试。"""

    def search_reviews(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
        _ = (query, top_k)
        return []


_REVIEW_SYSTEM = (
    "你是评价分析助手。根据多条公开评论摘要，输出结构化中文总结。\n"
    "要求：\n"
    "1) 只基于给定评论片段，不编造评分、销量、参数。\n"
    "2) 总结需包含：总体倾向、高频优点、高频缺点、争议点、适用人群建议。\n"
    "3) 若证据不足，明确写“证据不足”。\n"
    "4) 仅输出一行 JSON，不要 markdown：\n"
    '{"summary":"...","pros":["..."],"cons":["..."],"controversies":["..."],"advice":"...","confidence":"high|medium|low"}'
)

def _clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t


def _dedupe_hits(hits: list[ReviewHit], *, max_items: int) -> list[ReviewHit]:
    seen: set[str] = set()
    result: list[ReviewHit] = []
    for h in hits:
        key = (h.url or "").strip() or f"{h.source}|{h.title}|{h.snippet[:32]}"
        if not key or key in seen:
            continue
        seen.add(key)
        title = _clean_text(h.title)
        snippet = _clean_text(h.snippet)
        if not title and not snippet:
            continue
        result.append(
            ReviewHit(
                title=title[:120],
                url=(h.url or "").strip(),
                snippet=snippet[:220],
                source=_clean_text(h.source)[:40],
                published_at=_clean_text(h.published_at)[:32],
                score=float(h.score or 0.0),
            )
        )
        if len(result) >= max_items:
            break
    return result


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


class OnlineReviewSkill:
    """口碑查询 Skill：适用于“网上评价/值不值得买/优缺点”类问题。"""

    def __init__(
        self,
        provider: ReviewSearchProvider | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self._provider = provider or NullReviewProvider()
        self._model = (model or settings.simple_llm_model).strip()

    def run(
        self,
        question: str,
        *,
        enrichment: str = "",
        top_k: int = 8,
        triggered: bool = True,
    ) -> OnlineReviewResult:
        """执行 skill。

        说明：
        - 本 skill 默认由上游路由/编排层决定是否触发；
        - 这里不再做关键词意图判断，避免与 Router 职责重叠。
        - 若需显式标记未触发，可传入 ``triggered=False``。
        """
        # 判断问题是否为空以及开关是否开启
        q = (question or "").strip()
        if not q:
            return OnlineReviewResult(
                ok=False,
                triggered=False,
                question="",
                fallback_reason="empty_question",
            )
        if not triggered:
            return OnlineReviewResult(
                ok=False,
                triggered=False,
                question=q,
                fallback_reason="not_triggered_by_router",
            )
        # 构建搜索query
        search_query = self._build_search_query(q, enrichment=enrichment)
        try:
            # 搜索评论
            raw_hits = self._provider.search_reviews(search_query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return OnlineReviewResult(
                ok=False,
                triggered=True,
                question=q,
                search_query=search_query,
                fallback_reason=f"provider_error:{exc}",
            )
        # 去重
        hits = _dedupe_hits(raw_hits, max_items=max(4, top_k))
        # 如果没有命中，则返回空
        if not hits:
            return OnlineReviewResult(
                ok=False,
                triggered=True,
                question=q,
                search_query=search_query,
                fallback_reason="no_hits",
            )
        # 摘要
        summary = self._summarize(question=q, query=search_query, hits=hits)
        # 如果没有摘要，则返回兜底摘要
        if not summary:
            summary = self._heuristic_summary(q, hits)
        
        context = self._build_context_block(summary=summary, hits=hits)
        return OnlineReviewResult(
            ok=True,
            triggered=True,
            question=q,
            search_query=search_query,
            summary=summary,
            context_block=context,
            hits=hits,
            sources=[h.url for h in hits if h.url],
            fallback_reason="",
        )

    @staticmethod
    def _build_search_query(question: str, *, enrichment: str) -> str:
        """把问题改写成更适合网页评价检索的 query。"""
        q = _clean_text(question)
        e = _clean_text(enrichment)
        # 对话上文只取短片段，防止污染检索关键词。
        if e:
            e = e[:120]
            return f"{q} 真实评价 口碑 优缺点 {e}"
        return f"{q} 真实评价 口碑 优缺点"

    def _summarize(self, *, question: str, query: str, hits: list[ReviewHit]) -> str:
        prompt = self._build_summary_prompt(question=question, query=query, hits=hits)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"[OnlineReviewSkill] LLM 摘要失败: {exc}")
            return ""

        obj = _extract_json(raw)
        if not obj:
            return ""

        summary = _clean_text(str(obj.get("summary", "")))
        pros = obj.get("pros")
        cons = obj.get("cons")
        controversies = obj.get("controversies")
        advice = _clean_text(str(obj.get("advice", "")))
        confidence = _clean_text(str(obj.get("confidence", ""))).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        parts: list[str] = []
        if summary:
            parts.append(f"总体倾向：{summary}")
        if isinstance(pros, list) and pros:
            parts.append("高频优点：" + "；".join(_clean_text(str(x)) for x in pros[:4] if str(x).strip()))
        if isinstance(cons, list) and cons:
            parts.append("高频缺点：" + "；".join(_clean_text(str(x)) for x in cons[:4] if str(x).strip()))
        if isinstance(controversies, list) and controversies:
            parts.append("争议点：" + "；".join(_clean_text(str(x)) for x in controversies[:3] if str(x).strip()))
        if advice:
            parts.append(f"适用建议：{advice}")
        parts.append(f"结论置信：{confidence}")
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def _heuristic_summary(question: str, hits: list[ReviewHit]) -> str:
        """无 LLM 可用时的兜底摘要。"""
        q = _clean_text(question)
        pros_signals = ("续航", "稳定", "性价比", "好用", "轻便", "噪音小", "动力足")
        cons_signals = ("发热", "噪音大", "做工", "偏贵", "震动", "故障", "售后")

        text = " ".join(f"{h.title} {h.snippet}" for h in hits).lower()
        pros = [s for s in pros_signals if s in text][:3]
        cons = [s for s in cons_signals if s in text][:3]

        p = "、".join(pros) if pros else "证据不足"
        c = "、".join(cons) if cons else "证据不足"
        return (
            f"总体倾向：基于公开评论片段的快速汇总，关于“{q}”的信息有限。\n"
            f"高频优点：{p}\n"
            f"高频缺点：{c}\n"
            "适用建议：建议结合你的预算、使用频率和噪音容忍度进一步筛选。"
        )

    @staticmethod
    def _build_summary_prompt(*, question: str, query: str, hits: list[ReviewHit]) -> str:
        lines = [_REVIEW_SYSTEM, "", f"用户问题：{question}", f"检索 query：{query}", "", "评论片段："]
        for i, h in enumerate(hits[:8], start=1):
            source = h.source or "unknown"
            url = h.url or "-"
            ts = h.published_at or "-"
            lines.append(f"[{i}] source={source} time={ts}")
            lines.append(f"title={h.title}")
            lines.append(f"snippet={h.snippet}")
            lines.append(f"url={url}")
            lines.append("")
        return "\n".join(lines).strip()

    def _call_llm(self, prompt: str) -> str:
        return chat_text(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            timeout=20,
        )

    @staticmethod
    def _build_context_block(*, summary: str, hits: list[ReviewHit]) -> str:
        lines = ["[口碑评价摘要]", summary, "", "[来源]"]
        for idx, h in enumerate(hits[:8], start=1):
            src = h.source or "unknown"
            title = h.title or "(无标题)"
            url = h.url or "-"
            lines.append(f"{idx}. {title} | {src} | {url}")
        return "\n".join(lines).strip()
