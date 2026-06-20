"""多模态 embedding 客户端（Jina / DashScope）。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from app.core.config import settings


class JinaMultimodalEmbeddingClient:
    """封装多模态 embedding API。

    兼容历史类名：默认 provider 由 ``MULTIMODAL_EMBED_PROVIDER`` 控制。
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.provider = (provider or settings.multimodal_embed_provider or "jina_api").strip().lower()
        default_api_key = (
            settings.jina_api_key
            if self.provider == "jina_api"
            else (settings.dashscope_api_key or settings.bailian_api_key)
        )
        default_endpoint = (
            settings.jina_embedding_endpoint
            if self.provider == "jina_api"
            else settings.dashscope_multimodal_embedding_endpoint
        )
        default_timeout = (
            settings.jina_embedding_timeout_sec
            if self.provider == "jina_api"
            else settings.dashscope_embedding_timeout_sec
        )
        self.api_key = api_key if api_key is not None else default_api_key
        self.model = model or settings.multimodal_embed_model
        self.dimension = settings.multimodal_embed_dimension
        self.endpoint = endpoint or default_endpoint
        self.timeout = timeout or default_timeout
        self.session = session or requests.Session()

    def embed_text(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return []
        return self._embed_one(text)

    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        if self.provider == "jina_api":
            return self._embed_one({"image": image_b64})
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        image_fmt = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        return self._embed_one({"image": f"data:image/{image_fmt};base64,{image_b64}"})

    def embed_image_input(self, image_b64_or_data_uri: str) -> list[float]:
        raw = (image_b64_or_data_uri or "").strip()
        if not raw:
            return []
        if raw.startswith(("http://", "https://")):
            return self._embed_one({"image": raw})
        if self.provider == "jina_api":
            if raw.startswith("data:image/") and "," in raw:
                raw = raw.split(",", 1)[1]
            return self._embed_one({"image": raw})
        if raw.startswith("data:image/"):
            return self._embed_one({"image": raw})
        return self._embed_one({"image": f"data:image/jpeg;base64,{raw}"})

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = [text.strip() for text in texts if text and text.strip()]
        if not inputs:
            return []
        return self._embed_many(inputs)

    def _embed_one(self, item: str | dict[str, str]) -> list[float]:
        vectors = self._embed_many([item])
        return vectors[0] if vectors else []

    def _embed_many(self, inputs: list[str | dict[str, str]]) -> list[list[float]]:
        if self.provider == "jina_api":
            return self._embed_many_jina(inputs)
        if self.provider in {"dashscope_multimodal", "bailian_multimodal"}:
            return self._embed_many_dashscope(inputs)
        raise RuntimeError(f"不支持的 MULTIMODAL_EMBED_PROVIDER: {self.provider}")

    def _embed_many_jina(self, inputs: list[str | dict[str, str]]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("缺少 JINA_API_KEY，无法调用 Jina 图文 embedding API")
        payload = {"model": self.model, "input": inputs}
        return self._post_and_parse_vectors(
            payload=payload,
            auth_error="Jina embedding 认证失败：请检查 JINA_API_KEY",
            rate_limit_error="Jina embedding 额度或速率受限：请稍后重试或升级额度",
            generic_error_prefix="Jina embedding API",
            result_path=("data",),
        )

    def _embed_many_dashscope(self, inputs: list[str | dict[str, str]]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY），无法调用百炼多模态 embedding API")
        contents: list[dict[str, str]] = []
        for item in inputs:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    contents.append({"text": text})
                continue
            if isinstance(item, dict):
                if "text" in item:
                    text_value = str(item.get("text") or "").strip()
                    if text_value:
                        contents.append({"text": text_value})
                elif "image" in item:
                    image_value = str(item.get("image") or "").strip()
                    if image_value:
                        contents.append({"image": image_value})
        parameters: dict[str, Any] = {"output_type": "dense"}
        # multimodal-embedding-v1 固定 1024 维，传 dimension 会报错。
        if self.model != "multimodal-embedding-v1":
            parameters["dimension"] = self.dimension
        payload = {
            "model": self.model,
            "input": {"contents": contents},
            "parameters": parameters,
        }
        return self._post_and_parse_vectors(
            payload=payload,
            auth_error="百炼 embedding 认证失败：请检查 DASHSCOPE_API_KEY/BAILIAN_API_KEY",
            rate_limit_error="百炼 embedding 额度或速率受限：请稍后重试",
            generic_error_prefix="百炼 embedding API",
            result_path=("output", "embeddings"),
        )

    def _post_and_parse_vectors(
        self,
        *,
        payload: dict[str, Any],
        auth_error: str,
        rate_limit_error: str,
        generic_error_prefix: str,
        result_path: tuple[str, ...],
    ) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "")
            detail = self._format_http_error_detail(exc.response)
            if status in {401, 403}:
                raise RuntimeError(f"{auth_error}{detail}") from exc
            if status == 429:
                raise RuntimeError(f"{rate_limit_error}{detail}") from exc
            if status == 400 and self.provider in {"dashscope_multimodal", "bailian_multimodal"}:
                hint = (
                    "；请确认 MULTIMODAL_EMBED_MODEL 为百炼多模态 embedding 模型"
                    "（如 qwen3-vl-embedding / multimodal-embedding-v1），"
                    "qwen3.5-omni-flash 等对话模型不能用于 embedding API"
                )
                raise RuntimeError(
                    f"{generic_error_prefix} HTTP 失败 status={status}{detail}{hint}"
                ) from exc
            raise RuntimeError(f"{generic_error_prefix} HTTP 失败 status={status}{detail}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"{generic_error_prefix} 请求失败：{exc}") from exc

        data = resp.json()
        items: Any = data
        for key in result_path:
            if isinstance(items, dict):
                items = items.get(key)
            else:
                items = None
                break
        if not isinstance(items, list):
            raise RuntimeError(f"{generic_error_prefix} 返回格式异常：缺少向量列表")
        vectors: list[list[float]] = []
        for item in items:
            embedding: Any = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError(f"{generic_error_prefix} 返回格式异常：缺少 embedding")
            vectors.append([float(v) for v in embedding])
        return vectors

    @staticmethod
    def _format_http_error_detail(response: requests.Response | None) -> str:
        if response is None:
            return ""
        try:
            data = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return f": {text[:300]}" if text else ""
        if not isinstance(data, dict):
            return ""
        code = str(data.get("code") or "").strip()
        message = str(data.get("message") or "").strip()
        if code and message:
            return f": {code} - {message}"
        if message:
            return f": {message}"
        if code:
            return f": {code}"
        return ""
