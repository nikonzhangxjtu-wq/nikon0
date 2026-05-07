"""路由与 RAG 门控逻辑。

V2：短语 + 强弱关键词加权打分，输出 manual / customer_service / unknown / conflict。
unknown 时可选用 LLM 仲裁是否检索（仅采用 needs_rag 与 reason）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.core.config import settings

# LLM 仲裁 unknown 后的固定字段（除 needs_rag、reason 外）
_LLM_UNKNOWN_DOMAIN = "unknown"
_LLM_UNKNOWN_CONFIDENCE = 0.5
_LLM_UNKNOWN_STRATEGY = "llm_arbitrated_unknown"

# 权重：短语 > 强词 > 弱词
_WEIGHT_PHRASE = 3.0
_WEIGHT_STRONG = 2.0
_WEIGHT_WEAK = 1.0

# 手册：短语优先匹配（长在前，避免子串重复计分由 _is_redundant 处理）
_MANUAL_PHRASES: tuple[str, ...] = (
    # 英文短语
    "troubleshooting guide",
    "user manual",
    "how to install",
    "how to set up",
    "how to clean",
    "how to replace",
    "how to start",
    "how to stop",
    "how to charge",
    "how to adjust",
    # 中文短语（长优先 → 短靠后，避免子串匹配先到）
    "如何清洁空气滤网",
    "如何更换滤网",
    "如何清洁滤网",
    "如何安装电池",
    "技术规格",
    "规格参数",
    "维护保养",
    "故障排除",
    "常见问题",
    "按键功能",
    "指示灯闪烁",
    "指示灯不亮",
    "指示灯一直亮",
    "指示灯显示",
    "故障代码",
    "安装步骤",
    "清洗步骤",
    "怎么清洗",
    "如何清洁",
    "如何清洗",
    "怎么安装",
    "如何安装",
    "怎么拆卸",
    "如何拆卸",
    "怎么更换",
    "如何更换",
    "怎么使用",
    "怎么用",
    "如何使用",
    "怎么开机",
    "如何开机",
    "怎么关机",
    "如何关机",
    "怎么启动",
    "如何启动",
    "怎么关闭",
    "如何关闭",
    "怎么连接",
    "如何连接",
    "怎么拆",
    "如何拆",
    "怎么充电",
    "如何充电",
    "怎么调节",
    "如何调节",
    "怎么设置",
    "如何设置",
    "怎么操作",
    "如何操作",
)

_MANUAL_STRONG: frozenset[str] = frozenset(
    {
        # 中文强关键词
        "手册",
        "说明书",
        "安装",
        "拆卸",
        "更换",
        "故障",
        "指示灯",
        "清洁",
        "清洗",
        "维护",
        "保养",
        "操作",
        "充电",
        "启动",
        "关闭",
        "滤网",
        "配件",
        "电池",
        "组装",
        "部件",
        "组件",
        # 英文强关键词
        "clean",
        "install",
        "manual",
        "troubleshoot",
        "maintenance",
        "replace",
        "assemble",
        "component",
        "specification",
        "feature",
    }
)

_MANUAL_WEAK: frozenset[str] = frozenset({"步骤", "如何", "怎么", "功能", "按键", "按钮"})

_CS_PHRASES: tuple[str, ...] = (
    "退款多久到账",
    "多久退款",
    "退货流程",
    "申请退款",
    "发票怎么开",
    "怎么开发票",
    "保修政策",
    "保修多久",
    "物流查询",
    "查物流",
    "售后电话",
    "快递丢了",
    "订单取消",
    "以旧换新",
    "上门安装",
    "包装破损",
    "商品损坏",
    "商品少发",
    "7天无理由",
    "假货",
    "翻新机",
    "临期商品",
    "保质期",
    "生产日期",
    "寄到国外",
    "国际配送",
)

_CS_STRONG: frozenset[str] = frozenset(
    {
        "退货",
        "退款",
        "换货",
        "发票",
        "物流",
        "投诉",
        "保修",
        "运费",
        "订单",
        "快递",
        "配送",
        "发货",
        "签收",
        "包裹",
        "赔偿",
        "售后",
        "维修",
    }
)

_CS_WEAK: frozenset[str] = frozenset({"优惠券", "差价"})

# 联网口碑：由外部来源提供证据，不走手册 RAG
_WEB_REVIEW_PHRASES: tuple[str, ...] = (
    "网上评价",
    "网络评价",
    "用户评价",
    "用户口碑",
    "真实反馈",
    "值不值得买",
    "值得买吗",
    "优缺点",
    "测评",
    "评测",
    "review",
    "rating",
    "worth buying",
)

_ORDER_STATUS_PHRASES: tuple[str, ...] = (
    "查订单",
    "订单状态",
    "订单进度",
    "物流到哪",
    "快递到哪",
    "配送进度",
    "什么时候到",
    "预计送达",
    "催单",
    "催发货",
    "order status",
    "track order",
)

_CASE_INTAKE_PHRASES: tuple[str, ...] = (
    "报修",
    "坏了",
    "不转",
    "无法启动",
    "不能用",
    "故障",
    "售后处理",
    "我要退货",
    "我要退款",
    "申请换货",
)

# 领域判定阈值（与伪代码一致，可按评测调参）
_CONFLICT_MIN_SIDE = 2.0
_CONFLICT_MAX_GAP = 1.5
_WIN_MARGIN = 1.5

# 客服侧关闭 RAG 的保守条件
_CS_NO_RAG_MIN_SCORE = 2.0
_CS_NO_RAG_LEAD = 0.5


def normalize(question: str) -> str:
    """统一小写与首尾空白（中文不受影响）。"""
    return (question or "").strip().lower()


def _is_redundant(kw_lower: str, covered_lower: list[str]) -> bool:
    """若关键词已完全落在此前命中的更长片段内，则不再重复加分。"""
    for chunk in covered_lower:
        if kw_lower != chunk and kw_lower in chunk:
            return True
    return False


def _score_side(
    q: str,
    phrases: tuple[str, ...],
    strong: frozenset[str],
    weak: frozenset[str],
    label: str,
) -> tuple[float, list[str]]:
    score = 0.0
    signals: list[str] = []
    covered: list[str] = []

    for phrase in sorted(phrases, key=len, reverse=True):
        pl = phrase.lower()
        if pl not in q:
            continue
        score += _WEIGHT_PHRASE
        signals.append(f"{label}phrase:{phrase}")
        covered.append(pl)

    for kw in sorted(strong, key=len, reverse=True):
        kl = kw.lower()
        if kl not in q or _is_redundant(kl, covered):
            continue
        score += _WEIGHT_STRONG
        signals.append(f"{label}strong:{kw}")
        covered.append(kl)

    for kw in sorted(weak, key=len, reverse=True):
        kl = kw.lower()
        if kl not in q or _is_redundant(kl, covered):
            continue
        score += _WEIGHT_WEAK
        signals.append(f"{label}weak:{kw}")
        covered.append(kl)

    return score, signals


def build_confidence(domain_hint: str, manual_score: float, cs_score: float) -> float:
    gap = abs(manual_score - cs_score)
    total = manual_score + cs_score
    top = max(manual_score, cs_score)

    if domain_hint == "unknown":
        if total <= 0:
            return 0.28
        return min(0.42, 0.22 + 0.06 * min(total, 3.0))

    if domain_hint == "conflict":
        return min(0.58, 0.42 + 0.02 * min(total, 8.0))

    # manual 或 customer_service：单边占优
    base = 0.66 + 0.06 * min(gap, 5.0) + 0.025 * min(top, 10.0)
    return min(0.95, base)


def build_reason(
    domain_hint: str,
    manual_signals: list[str],
    cs_signals: list[str],
    manual_score: float,
    cs_score: float,
) -> str:
    def _preview(sigs: list[str], n: int = 4) -> str:
        if not sigs:
            return "无"
        return "、".join(sigs[:n]) + ("…" if len(sigs) > n else "")

    if domain_hint == "unknown":
        return "未命中明确手册或客服信号，按未知领域处理"

    if domain_hint == "conflict":
        return (
            f"手册与客服信号并存(M{manual_score:.1f}/C{cs_score:.1f})；"
            f"手册:{_preview(manual_signals)}；客服:{_preview(cs_signals)}"
        )

    if domain_hint == "manual":
        return (
            f"说明书信号更强(M{manual_score:.1f}>C{cs_score:.1f})：{_preview(manual_signals)}"
        )

    # customer_service
    return f"客服信号更强(C{cs_score:.1f}>M{manual_score:.1f})：{_preview(cs_signals)}"


def _parse_llm_arbiter_output(raw: str) -> tuple[bool | None, str | None]:
    """从模型输出中解析 needs_rag 与 reason；失败返回 (None, None)。"""
    text = (raw or "").strip()
    if not text:
        return None, None
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                text = p
                break

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, None

    nr = obj.get("needs_rag")
    reason = obj.get("reason")
    if isinstance(nr, str):
        nr = nr.strip().lower() in ("true", "1", "yes", "y")
    if not isinstance(nr, bool):
        return None, None
    if not isinstance(reason, str) or not reason.strip():
        return None, None
    return nr, reason.strip()


def _llm_arbitrate_unknown(question: str) -> tuple[bool, str]:
    """调用本地 LLM，仅判断 unknown 场景下是否需要检索手册。"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        print(f"[WARN] LLM 路由仲裁跳过：无法导入 langchain_ollama ({exc})")
        return True, "LLM 仲裁不可用，默认尝试检索"

    model = (settings.router_arbiter_model or "").strip() or settings.gen_model
    client = ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
    system = (
        "你是对话系统的路由仲裁模块。用户问题无法由关键词明确归类为"
        "「产品说明书/安装故障」或「订单/售后/发票」等客服政策。"
        "请判断：要给出可靠回答，是否需要从产品说明手册知识库中检索（RAG）。\n"
        "规则：涉及设备用法、安装、故障、参数、操作步骤等，needs_rag 应为 true；"
        "纯寒暄、与产品无关、或仅需通用客服话术且明显不需要查手册时，可为 false。\n"
        "只输出一行合法 JSON，不要其它文字、不要 markdown："
        '{"needs_rag": true 或 false, "reason": "不超过 80 字的中文简短理由"}'
    )
    user = f"用户问题：\n{question.strip()}\n"
    try:
        msg = client.invoke([("system", system), ("human", user)])
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] LLM 路由仲裁失败，默认尝试检索: {exc}")
        return True, f"LLM 仲裁失败，默认尝试检索（{exc}）"

    raw = getattr(msg, "content", "") or ""
    parsed = _parse_llm_arbiter_output(raw)
    if parsed[0] is None:
        print(f"[WARN] LLM 路由仲裁无法解析输出，默认尝试检索。原始输出: {raw[:200]!r}")
        return True, "LLM 输出无法解析，默认尝试检索"

    return parsed[0], parsed[1]


@dataclass
class RouteDecision:
    """传给下游流水线的路由决策。"""

    needs_rag: bool
    domain_hint: str
    reason: str
    confidence: float = 0.0
    strategy: str = "heuristic_scoring"
    manual_score: float = 0.0
    cs_score: float = 0.0
    manual_signals: list[str] = field(default_factory=list)
    cs_signals: list[str] = field(default_factory=list)


class QuestionRouter:
    """启发式打分路由器。"""

    def decide(self, question: str) -> RouteDecision:
        q = normalize(question)

        # case_intake: 工单收集/分诊优先最高，避免被普通客服分支吞掉。
        if q and any(phrase in q for phrase in _CASE_INTAKE_PHRASES):
            return RouteDecision(
                needs_rag=False,
                domain_hint="case_intake",
                reason="命中售后受理意图，转工单收集 skill",
                confidence=0.9,
                strategy="heuristic_case_intake",
            )

        # web_review 走独立 skill：优先于 manual/cs 评分，避免误进 RAG。
        if q and any(phrase in q for phrase in _WEB_REVIEW_PHRASES):
            return RouteDecision(
                needs_rag=False,
                domain_hint="web_review",
                reason="命中口碑/测评意图，转联网评价 skill",
                confidence=0.88,
                strategy="heuristic_web_review",
            )

        # order_status 走 MCP 订单查询 skill：优先于通用客服 no-rag。
        if q and any(phrase in q for phrase in _ORDER_STATUS_PHRASES):
            return RouteDecision(
                needs_rag=False,
                domain_hint="order_status",
                reason="命中订单进度查询意图，转订单状态 skill",
                confidence=0.9,
                strategy="heuristic_order_status",
            )

        manual_score = 0.0
        cs_score = 0.0
        manual_signals: list[str] = []
        cs_signals: list[str] = []

        if q:
            manual_score, manual_signals = _score_side(
                q, _MANUAL_PHRASES, _MANUAL_STRONG, _MANUAL_WEAK, "M"
            )
            cs_score, cs_signals = _score_side(
                q, _CS_PHRASES, _CS_STRONG, _CS_WEAK, "C"
            )

        if manual_score == 0 and cs_score == 0:
            domain_hint = "unknown"
        elif (
            manual_score >= _CONFLICT_MIN_SIDE
            and cs_score >= _CONFLICT_MIN_SIDE
            and abs(manual_score - cs_score) <= _CONFLICT_MAX_GAP
        ):
            domain_hint = "conflict"
        elif manual_score >= cs_score + _WIN_MARGIN:
            domain_hint = "manual"
        elif cs_score >= manual_score + _WIN_MARGIN:
            domain_hint = "customer_service"
        else:
            domain_hint = "unknown"

        if domain_hint == "customer_service":
            if cs_score >= _CS_NO_RAG_MIN_SCORE and cs_score >= manual_score + _CS_NO_RAG_LEAD:
                needs_rag = False
            else:
                needs_rag = True
        else:
            needs_rag = True

        confidence = build_confidence(domain_hint, manual_score, cs_score)
        reason = build_reason(
            domain_hint, manual_signals, cs_signals, manual_score, cs_score
        )

        if (
            domain_hint == "unknown"
            and settings.router_unknown_llm_arbitrate
            and q
        ):
            needs_rag, reason = _llm_arbitrate_unknown(question)
            return RouteDecision(
                needs_rag=needs_rag,
                domain_hint=_LLM_UNKNOWN_DOMAIN,
                reason=reason,
                confidence=_LLM_UNKNOWN_CONFIDENCE,
                strategy=_LLM_UNKNOWN_STRATEGY,
                manual_score=0.0,
                cs_score=0.0,
                manual_signals=[],
                cs_signals=[],
            )

        return RouteDecision(
            needs_rag=needs_rag,
            domain_hint=domain_hint,
            reason=reason,
            confidence=confidence,
            strategy="heuristic_scoring",
            manual_score=manual_score,
            cs_score=cs_score,
            manual_signals=list(manual_signals),
            cs_signals=list(cs_signals),
        )
