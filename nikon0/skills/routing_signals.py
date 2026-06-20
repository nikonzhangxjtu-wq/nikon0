"""Shared keyword signals for product_support rule-fallback routing."""

from __future__ import annotations

# Strong manual / product-support signals (aligned with question_public.csv patterns).
PRODUCT_SUPPORT_KEYWORDS: tuple[str, ...] = (
    "怎么",
    "如何",
    "怎样",
    "多久",
    "多长时间",
    "多少",
    "几次",
    "清洁",
    "清洗",
    "保养",
    "维护",
    "更换",
    "安装",
    "拆卸",
    "拆解",
    "组装",
    "连接",
    "滤网",
    "过滤器",
    "喷淋臂",
    "传感器",
    "安全锁",
    "模式",
    "功能",
    "步骤",
    "参数",
    "规格",
    "技术规格",
    "电压",
    "伏",
    "指示灯",
    "故障码",
    "故障",
    "怎么办",
    "怎么处理",
    "能不能",
    "可以吗",
    "要不要",
    "有什么",
    "有哪些",
    "是什么",
    "什么意思",
    "表现",
    "预约",
    "半载",
    "洗涤块",
    "亮碟剂",
    "软化",
    "水硬度",
    "排插",
    "延长线",
    "熄火绳",
    "油门",
    "转向",
    "尾流",
    "载重",
    "充电",
    "格式化",
    "对焦",
    "白平衡",
    "曝光",
    "自拍",
    "镜头",
    "外接",
    "交流电",
    "iso",
    "ef-s",
    "ef ",
    "cf卡",
    "空气净化器",
    "洗碗机",
    "摩托艇",
    "相机",
    "单反",
    "拍立得",
    "空调",
    "空气炸锅",
    "烤箱",
    "洗衣机",
    "微波炉",
    "咖啡机",
    "蓝牙鼠标",
    "vr头显",
    "头显",
    "冰箱",
    "异味",
    "焦味",
    "漏光",
    "遮光罩",
    "接收器",
    "替换件",
    "吹风机",
    "电钻",
    "发电机",
    "健身",
    "蒸汽清洁",
    "冰箱",
    "水泵",
    "e1",
    "e2",
    "e3",
    "e4",
    "显示",
    "启动",
    "AC900",
    # English product names and manual-QA verbs. These are a deterministic
    # safety floor for regression evaluation; production routing may still use
    # the LLM selector before this fallback is considered.
    "airfryer",
    "air fryer",
    "microwave",
    "nespresso",
    "coffee machine",
    "washing machine",
    "dishwasher",
    "refrigerator",
    "fridge",
    "oven",
    "vr headset",
    "headset",
    "bluetooth mouse",
    "hair dryer",
    "leaf blower",
    "clean",
    "cleaning",
    "descale",
    "install",
    "setup",
    "use",
    "program",
    "troubleshoot",
    "troubleshooting",
    "error code",
    "won't start",
    "not working",
    "not heating",
    "sparking",
    "child lock",
    "sensor cook",
)

# Platform / transactional intents — exclude from product_support rule fallback.
CUSTOMER_SERVICE_KEYWORDS: tuple[str, ...] = (
    "退款",
    "退货",
    "换货",
    "投诉",
    "报修",
    "发票",
    "物流",
    "快递",
    "订单",
    "优惠券",
    "运费",
    "7天无理由",
    "七天无理由",
    "赔偿",
    "假货",
    "二手",
    "临期",
    "补发",
    "取消订单",
    "refund",
    "return",
    "replacement",
    "replace",
    "warranty",
    "repair",
    "complaint",
    "order",
    "shipping",
    "delivery",
    "invoice",
)

CASE_INTAKE_KEYWORDS: tuple[str, ...] = (
    "报修", "售后", "维修", "坏了", "无法启动", "不能用了", "用不了", "不转",
    "退款", "退货", "换货", "投诉", "没人管", "帮我处理", "要求赔偿",
    "保修", "换新", "改约", "预约上门",
    "repair", "broken", "not working", "won't start", "warranty", "claim",
    "refund", "return", "replacement", "replace", "complaint", "damaged",
    "leaking", "urgent repair", "service appointment",
)

HANDOFF_KEYWORDS: tuple[str, ...] = (
    "投诉升级", "转人工", "人工客服", "人工处理", "律师", "起诉", "漏电", "触电",
    "human agent", "live agent", "operator", "legal action", "lawsuit",
    "property damage", "electric shock", "electrocuted", "speak to a person",
    "parlare con un operatore",
)

APPROVAL_KEYWORDS: tuple[str, ...] = (
    "退款", "赔偿", "修改订单", "发送外部消息", "创建正式工单",
    "refund", "compensation", "modify order", "send external message", "create ticket",
)

# Manual prohibition / safety questions that contain「不能用」but are NOT case intake.
MANUAL_PROHIBITION_PATTERNS: tuple[str, ...] = (
    "为什么不能用",
    "为何不能用",
    "能不能用排插",
    "能不能用延长线",
    "为什么不能用排插",
)


def looks_like_product_support(message: str) -> tuple[bool, list[str]]:
    text = (message or "").strip()
    if not text:
        return False, []
    lowered = text.lower()
    if any(keyword.lower() in lowered for keyword in CUSTOMER_SERVICE_KEYWORDS):
        # Allow manual questions that also mention CS words only when strong manual signal exists.
        manual_hits = [kw for kw in PRODUCT_SUPPORT_KEYWORDS if kw in text or kw in lowered]
        if not manual_hits:
            return False, []
    hits = [kw for kw in PRODUCT_SUPPORT_KEYWORDS if kw in text or kw in lowered]
    return bool(hits), hits


def looks_like_case_intake(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if any(pattern in text for pattern in MANUAL_PROHIBITION_PATTERNS):
        return False
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in CASE_INTAKE_KEYWORDS)


def has_handoff_signal(message: str) -> bool:
    lowered = (message or "").lower()
    return any(keyword.lower() in lowered for keyword in HANDOFF_KEYWORDS)


def has_approval_signal(message: str) -> bool:
    lowered = (message or "").lower()
    return any(keyword.lower() in lowered for keyword in APPROVAL_KEYWORDS)
