"""生成服务封装。

通过 LangChain 的 Ollama 集成调用本地 qwen2。
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import settings
import time

class Qwen2Generator:
    """大模型生成封装，用于合成最终回答。

    TODO（你来补）：
    - 按评测调 temperature / top_p
    - 增加重试、超时
    - 可选：强制结构化输出格式
    """

    def __init__(self) -> None:
        self.client = ChatOllama(
            model=settings.gen_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

    def generate(self, prompt: str) -> str:
        """调用生成模型，返回纯文本。"""
        time_start = time.time()
        try:
            result = self.client.invoke(prompt)
        except Exception as e:
            raise Exception(f"生成失败: {e}")
        time_end = time.time()
        print(result)
        return getattr(result, "content", "").strip()