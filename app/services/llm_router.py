"""LLM 驱动的路由分类器。

用本地 qwen2 模型判断用户问题是否需要 RAG 检索，替代关键词启发式。
失败时自动回退到关键词路由器。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.config import settings
from app.services.router import QuestionRouter, RouteDecision


@dataclass
class _LLMRouteOutput:
    needs_rag: bool
    domain: str
    reason: str
    confidence: float


_ROUTER_SYSTEM = (
    "你是客服对话系统的路由分类器。判断用户问题是否需要从产品说明书知识库检索（RAG）。\n"
    "\n"
    "分类规则：\n"
    "1. 涉及以下 → needs_rag=true, domain=\"manual\"：\n"
    "   产品使用、操作步骤、安装拆卸、故障排除、指示灯、错误代码、\n"
    "   技术规格参数、功能说明、清洁维护保养、配件更换\n"
    "2. 涉及以下 → needs_rag=false, domain=\"customer_service\"：\n"
    "   订单、退款、退货换货、物流快递、发票、保修政策、售后投诉、\n"
    "   价格优惠券、配送运费、购买渠道\n"
    "3. 涉及以下 → needs_rag=false, domain=\"web_review\"：\n"
    "   真实口碑、用户评价、网上测评、值得买吗、优缺点对比\n"
    "4. 涉及以下 → needs_rag=false, domain=\"order_status\"：\n"
    "   查订单状态、物流到哪、配送进度、预计送达、催发货\n"
    "5. 涉及以下 → needs_rag=false, domain=\"case_intake\"：\n"
    "   报修、故障受理、退换货申请、需要人工售后跟进的信息收集\n"
    "6. 其他 → needs_rag=false, domain=\"unknown\"：\n"
    "   纯寒暄、无关闲聊、无法判断的问题\n"
    "\n"
    "严格只输出一行 JSON，不要 markdown 代码块，不要额外文字：\n"
    '{"needs_rag": true/false, "domain": "manual"|"customer_service"|"web_review"|"order_status"|"case_intake"|"unknown", "reason": "中文 ≤30 字", "confidence": 0.0~1.0}\n'
    "\n"
    "示例：\n"
    "Q: 净水器怎么更换滤芯？\n"
    'A: {"needs_rag": true, "domain": "manual", "reason": "询问更换滤芯的操作方法", "confidence": 0.95}\n'
    "Q: 我买的扫地机坏了能退款吗？\n"
    'A: {"needs_rag": false, "domain": "customer_service", "reason": "售后退款问题非产品使用", "confidence": 0.9}\n'
    "Q: 今天天气真好\n"
    'A: {"needs_rag": false, "domain": "unknown", "reason": "与产品无关的闲聊", "confidence": 0.95}\n'
    "Q: how to clean the air filter\n"
    'A: {"needs_rag": true, "domain": "manual", "reason": "询问清洁空气滤网方法", "confidence": 0.95}\n'
    "Q: 指示灯一直闪红灯什么意思\n"
    'A: {"needs_rag": true, "domain": "manual", "reason": "询问指示灯故障含义", "confidence": 0.9}\n'
    "Q: 物流到哪了帮我查一下\n"
    'A: {"needs_rag": false, "domain": "customer_service", "reason": "物流查询属于客服范畴", "confidence": 0.9}\n'
    "Q: 这款电钻网上评价怎么样\n"
    'A: {"needs_rag": false, "domain": "web_review", "reason": "用户在询问真实口碑评价", "confidence": 0.9}\n'
    "Q: 订单 OD20260507001 到哪了\n"
    'A: {"needs_rag": false, "domain": "order_status", "reason": "用户在查询订单物流状态", "confidence": 0.93}\n'
    "Q: 电钻坏了，帮我报修\n"
    'A: {"needs_rag": false, "domain": "case_intake", "reason": "需要收集售后受理信息", "confidence": 0.92}\n'
)


class LLMRouter:
    """用本地 LLM 做 RAG 需求分类，失败时回退关键词路由器。"""

    def __init__(self, model: str | None = None) -> None:
        self._model = (model or settings.router_llm_model).strip()
        self._keyword_router = QuestionRouter()

    def decide(self, question: str) -> RouteDecision:
        q = (question or "").strip()
        if not q:
            return RouteDecision(
                needs_rag=True,
                domain_hint="unknown",
                reason="空输入，默认尝试检索",
                confidence=0.5,
                strategy="empty_input_default",
            )

        try:
            prompt = self._build_prompt(q)
            raw = self._call_llm(prompt)
            parsed = self._parse_output(raw)
            if parsed is not None:
                return RouteDecision(
                    needs_rag=parsed.needs_rag,
                    domain_hint=parsed.domain,
                    reason=parsed.reason,
                    confidence=parsed.confidence,
                    strategy="llm_classifier",
                )
        except Exception as exc:
            print(f"[LLMRouter] 分类失败，回退关键词路由: {exc}")

        return self._fallback_decision(q)

    def _build_prompt(self, question: str) -> str:
        return f"{_ROUTER_SYSTEM}\n用户问题：\n{question}\n"

    def _call_llm(self, prompt: str) -> str:
        import requests as _req

        resp = _req.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("message", {}).get("content", "")

    def _parse_output(self, raw: str) -> _LLMRouteOutput | None:
        text = (raw or "").strip()
        if not text:
            return None

        # 去掉可能的 markdown fence
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") and part.endswith("}"):
                    text = part
                    break

        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None

        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        nr = obj.get("needs_rag")
        if isinstance(nr, str):
            nr = nr.strip().lower() in ("true", "1", "yes", "y")
        if not isinstance(nr, bool):
            return None

        domain = str(obj.get("domain", "unknown")).strip().lower()
        if domain not in ("manual", "customer_service", "web_review", "order_status", "case_intake", "unknown"):
            domain = "unknown"

        reason = str(obj.get("reason", "LLM 分类")).strip()
        if not reason:
            reason = "LLM 分类未提供理由"

        confidence = float(obj.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        return _LLMRouteOutput(
            needs_rag=nr,
            domain=domain,
            reason=reason,
            confidence=confidence,
        )

    def _fallback_decision(self, question: str) -> RouteDecision:
        decision = self._keyword_router.decide(question)
        # 标记为 fallback 策略以便调试
        return RouteDecision(
            needs_rag=decision.needs_rag,
            domain_hint=decision.domain_hint,
            reason=f"[关键词回退] {decision.reason}",
            confidence=decision.confidence,
            strategy="heuristic_fallback",
            manual_score=decision.manual_score,
            cs_score=decision.cs_score,
            manual_signals=decision.manual_signals,
            cs_signals=decision.cs_signals,
        )
