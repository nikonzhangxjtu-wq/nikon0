"""生成服务封装。

支持双后端：
- 百炼 API（OpenAI 兼容，BAILIAN_API_KEY 设了就启用）
- 本地 Ollama（回退方案）

比赛模式：低温、空输出时自动重试（次数见配置）。
"""

from __future__ import annotations

import time

from app.core.config import settings


class Qwen2Generator:
    """大模型生成封装，用于合成最终回答。"""

    def __init__(self) -> None:
        temp = (
            settings.gen_temperature_competition
            if settings.gen_competition_mode
            else settings.gen_temperature_casual
        )
        self._competition = settings.gen_competition_mode
        self._max_retries = max(1, settings.gen_max_retries)

        if settings.bailian_api_key:
            self.client = self._build_bailian_client(temp)
        else:
            self.client = self._build_ollama_client(temp)

    @staticmethod
    def _build_bailian_client(temp: float):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise RuntimeError(
                "百炼 API 需要 langchain-openai 包，请执行: pip install langchain-openai"
            ) from err

        return ChatOpenAI(
            model=settings.gen_model,
            base_url=settings.bailian_base_url,
            api_key=settings.bailian_api_key,
            temperature=temp,
            max_tokens=settings.gen_max_tokens,
        )

    @staticmethod
    def _build_ollama_client(temp: float):
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.gen_model,
            base_url=settings.ollama_base_url,
            temperature=temp,
            num_predict=settings.gen_max_tokens,
        )

    def generate(self, prompt: str) -> str:
        """调用生成模型，返回纯文本。"""
        attempts = self._max_retries if self._competition else 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                result = self.client.invoke(prompt)
            except Exception as exc:
                last_exc = exc
                if i + 1 >= attempts:
                    raise RuntimeError(
                        f"生成失败（已重试 {attempts} 次）: {exc}"
                    ) from exc
                # 指数退避：1s、2s、4s…
                delay = min(2**i, 8.0)
                time.sleep(delay)
                continue
            text = getattr(result, "content", "") or ""
            text = text.strip()
            if text and (not self._competition or len(text) >= 20):
                return text
            if not text:
                last_exc = RuntimeError("模型返回空内容")
            else:
                last_exc = RuntimeError(f"模型返回内容过短（{len(text)} 字符），可能被截断")
            if i + 1 >= attempts:
                break
            time.sleep(1.0 * (i + 1))
        raise RuntimeError(
            f"生成失败：连续 {attempts} 次无有效输出"
        ) from last_exc
