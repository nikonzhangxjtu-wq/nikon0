"""MCP 联网评价 Provider（skills 模块版）。"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.online_review_skill import ReviewHit, ReviewSearchProvider


def _to_hit(item: dict[str, Any]) -> ReviewHit | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title", "")).strip()
    snippet = str(item.get("snippet", "")).strip()
    if not title and not snippet:
        return None
    url = str(item.get("url", "")).strip()
    source = str(item.get("source", "")).strip()
    published_at = str(item.get("published_at", "")).strip()
    score_raw = item.get("score", 0.0)
    try:
        score = float(score_raw or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return ReviewHit(
        title=title,
        url=url,
        snippet=snippet,
        source=source,
        published_at=published_at,
        score=score,
    )


class MCPReviewProvider(ReviewSearchProvider):
    """通过 HTTP endpoint 调 MCP 检索评价。"""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        api_key: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        ep = (endpoint or settings.mcp_review_endpoint).strip()
        if not ep:
            raise ValueError("MCP review endpoint 不能为空")
        self._endpoint = ep
        self._api_key = (api_key or settings.mcp_review_api_key).strip()
        self._timeout_sec = timeout_sec or settings.mcp_review_timeout_sec

    def search_reviews(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
        import requests as _req

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"query": query, "top_k": top_k}
        resp = _req.post(
            self._endpoint,
            json=payload,
            headers=headers,
            timeout=self._timeout_sec,
        )
        resp.raise_for_status()
        body = resp.json()

        rows: list[dict[str, Any]] = []
        if isinstance(body, list):
            rows = [x for x in body if isinstance(x, dict)]
        elif isinstance(body, dict):
            hits = body.get("hits")
            if isinstance(hits, list):
                rows = [x for x in hits if isinstance(x, dict)]
            else:
                data = body.get("data")
                if isinstance(data, dict) and isinstance(data.get("hits"), list):
                    rows = [x for x in data["hits"] if isinstance(x, dict)]

        out: list[ReviewHit] = []
        for row in rows:
            hit = _to_hit(row)
            if hit is not None:
                out.append(hit)
        return out

