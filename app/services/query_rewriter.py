"""Query 改写器：用本地轻量模型将省略指代的多轮问题改写为自包含检索查询。

仅在存在对话历史时触发，改写后的 query 只用于检索，不进入生成 prompt。
"""

from __future__ import annotations

from app.core.config import settings
from app.services.llm_clients import chat_text

_REWRITE_SYSTEM = (
    "你是查询改写助手。结合对话历史，将用户的当前问题改写为一个独立完整的检索查询。\n"
    "\n"
    "规则：\n"
    "1. 补全历史中已出现但当前省略的主语、宾语（如\"它\"\"这个\"\"那个\"\"第二步\"）\n"
    "2. 保留原始问题的全部子问题和关键细节，不要丢失任何信息\n"
    "3. 不要新增历史中没有的事实，不要猜测用户未提及的内容\n"
    "4. 只输出改写后的问题文本，不加引号、不解释、不评价\n"
    "\n"
    "示例：\n"
    "历史：用户问「得伟电钻不转了怎么办」→ 助手答「请检查电池和钻头」\n"
    "当前：指示灯闪红灯什么意思？\n"
    "输出：得伟电钻指示灯闪红灯什么意思？\n"
    "\n"
    "示例：\n"
    "历史：用户问「净水器怎么更换滤芯」→ 助手答「请打开面板取出旧滤芯」\n"
    "当前：第二步怎么清洗？\n"
    "输出：净水器滤芯更换后怎么清洗滤芯？\n"
)


class QueryRewriter:
    """用本地轻量 LLM 改写多轮对话中的省略问题。"""

    def __init__(self, model: str | None = None) -> None:
        self._model = (model or settings.simple_llm_model).strip()

    def rewrite(self, question: str, enrichment: str, memory_context: str = "") -> str:
        """返回改写后的 query；失败时返回空字符串让调用方回退。"""
        q = (question or "").strip()
        e = (enrichment or "").strip()
        m = (memory_context or "").strip()
        if not q or not (e or m):
            return ""

        prompt = self._build_prompt(q, e, memory_context=m)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            print(f"[QueryRewriter] 调用失败: {exc}")
            return ""

        rewritten = (raw or "").strip()
        if not rewritten or len(rewritten) < 3:
            return ""

        # 防止模型输出胡言乱语：改写后至少保留原问题 30% 的字面重叠
        if _char_overlap_ratio(q, rewritten) < 0.2:
            print(f"[QueryRewriter] 改写偏离过大，丢弃: {rewritten[:120]}")
            return ""

        return rewritten

    def _build_prompt(self, question: str, enrichment: str, memory_context: str = "") -> str:
        memory = ""
        if memory_context.strip():
            memory = (
                "\n[可用记忆]\n"
                f"{memory_context.strip()}\n"
                "[可用记忆结束]\n"
            )
        return f"{_REWRITE_SYSTEM}\n{enrichment}{memory}\n当前问题：{question}\n"

    def _call_llm(self, prompt: str) -> str:
        return chat_text(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
            timeout=15,
        )


def _char_overlap_ratio(a: str, b: str) -> float:
    """两个字符串的字符级 Jaccard 相似度，用于防止改写跑偏。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
