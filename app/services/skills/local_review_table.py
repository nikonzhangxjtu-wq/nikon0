"""本地评价表：内置若干商品评论，按 query 关键词检索，不访问外网。

与 ``OnlineReviewSkill`` 配合：实现 ``ReviewSearchProvider``，返回 ``ReviewHit``。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.services.online_review_skill import ReviewHit, ReviewSearchProvider


@dataclass(frozen=True)
class ReviewTableRow:
    """评价表中的一行（一条用户评论）。"""

    product_id: str
    product_name: str
    rating: float
    title: str
    snippet: str
    source: str = "本地评价表"
    published_at: str = ""
    url: str = ""

    def to_hit(self, *, row_index: int) -> ReviewHit:
        url = (self.url or "").strip() or f"local://review/{self.product_id}/{row_index}"
        title = (self.title or "").strip() or f"{self.product_name} 评价"
        snippet = (self.snippet or "").strip()
        return ReviewHit(
            title=f"【{self.product_name}】{title}",
            url=url,
            snippet=snippet,
            source=(self.source or "本地评价表").strip()[:40],
            published_at=(self.published_at or "").strip()[:32],
            score=float(self.rating or 0.0),
        )


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


_NOISE_PHRASES = (
    "真实评价",
    "口碑",
    "优缺点",
    "怎么样",
    "如何",
    "值得买吗",
    "值得",
    "买吗",
    "网上",
    "大家",
    "用户",
    "评测",
)


def _tokenize_query(query: str) -> list[str]:
    q = _clean(query)
    for p in _NOISE_PHRASES:
        q = q.replace(p, " ")
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9-]{1,}", q)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _row_text(row: ReviewTableRow) -> str:
    return _clean(f"{row.product_id} {row.product_name} {row.title} {row.snippet}")


def _score_row(query: str, row: ReviewTableRow) -> float:
    blob = _row_text(row)
    score = 0.0
    for tok in _tokenize_query(query):
        if tok in blob:
            score += 3.0
            if tok in row.product_name:
                score += 4.0
            if tok in row.title:
                score += 1.5
    # 仅作排序微调：无 token 命中时不得加分，否则任意 query 会匹配全表
    if score <= 0:
        return 0.0
    score += float(row.rating or 0.0) * 0.05
    return score


def _dict_to_row(obj: dict[str, Any]) -> ReviewTableRow | None:
    if not isinstance(obj, dict):
        return None
    pid = str(obj.get("product_id", "")).strip()
    pname = str(obj.get("product_name", "")).strip()
    title = str(obj.get("title", "")).strip()
    snippet = str(obj.get("snippet", "")).strip()
    if not pname or (not title and not snippet):
        return None
    if not pid:
        pid = pname[:16] or "unknown"
    try:
        rating = float(obj.get("rating", 0.0) or 0.0)
    except (TypeError, ValueError):
        rating = 0.0
    return ReviewTableRow(
        product_id=pid,
        product_name=pname,
        rating=rating,
        title=title or "评价",
        snippet=snippet,
        source=str(obj.get("source", "本地评价表")).strip()[:40] or "本地评价表",
        published_at=str(obj.get("published_at", "")).strip()[:32],
        url=str(obj.get("url", "")).strip(),
    )


# 内置示例数据：多商品、多评论，便于离线演示与单测
_BUILTIN_ROWS: list[dict[str, Any]] = [
    {
        "product_id": "vacuum-x1",
        "product_name": "追觅扫地机器人 X1",
        "rating": 4.6,
        "title": "扫得干净，地图准",
        "snippet": "避障比上一代好，地毯增压有用；集尘声音略大，能接受。",
        "source": "本地评价表",
        "published_at": "2024-11",
    },
    {
        "product_id": "vacuum-x1",
        "product_name": "追觅扫地机器人 X1",
        "rating": 3.8,
        "title": "缠头发问题",
        "snippet": "长发家庭主刷偶尔要手动清理，APP 功能多但学习成本有一点。",
        "source": "本地评价表",
        "published_at": "2024-10",
    },
    {
        "product_id": "drill-d500",
        "product_name": "博世手电钻 D500",
        "rating": 4.4,
        "title": "家用足够",
        "snippet": "动力足，续航不错；噪音偏大，晚上用会吵邻居。",
        "source": "本地评价表",
        "published_at": "2024-09",
    },
    {
        "product_id": "drill-d500",
        "product_name": "博世手电钻 D500",
        "rating": 4.1,
        "title": "做工稳",
        "snippet": "同心度好，打孔不飘；价格小贵，配件要另买。",
        "source": "本地评价表",
        "published_at": "2024-08",
    },
    {
        "product_id": "kettle-s3",
        "product_name": "米家电水壶 S3",
        "rating": 4.0,
        "title": "烧水快",
        "snippet": "一键保温好用；壶嘴略短，倒水要习惯一下。",
        "source": "本地评价表",
        "published_at": "2024-07",
    },
]


class LocalReviewTable:
    """内存中的评价表，支持按 query 简单打分检索。"""

    def __init__(self, rows: Sequence[ReviewTableRow]) -> None:
        self._rows = list(rows)

    @classmethod
    def default(cls) -> LocalReviewTable:
        parsed = [_dict_to_row(d) for d in _BUILTIN_ROWS]
        rows = [r for r in parsed if r is not None]
        return cls(rows)

    @classmethod
    def from_json_file(cls, path: str | Path) -> LocalReviewTable:
        p = Path(path)
        if not p.is_file():
            return cls.default()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return cls.default()
        if not isinstance(raw, list):
            return cls.default()
        rows: list[ReviewTableRow] = []
        for item in raw:
            r = _dict_to_row(item) if isinstance(item, dict) else None
            if r is not None:
                rows.append(r)
        base = cls.default()._rows
        return cls(base + rows)

    def all_rows(self) -> list[ReviewTableRow]:
        return list(self._rows)

    def search(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
        q = _clean(query)
        if not q:
            return []
        scored: list[tuple[float, int, ReviewTableRow]] = []
        for i, row in enumerate(self._rows):
            s = _score_row(q, row)
            if s > 0:
                scored.append((s, i, row))
        scored.sort(key=lambda x: (-x[0], -x[2].rating, x[1]))
        out: list[ReviewHit] = []
        for _, idx, row in scored[: max(1, top_k)]:
            out.append(row.to_hit(row_index=idx))
        return out


class LocalReviewTableProvider(ReviewSearchProvider):
    """从本地评价表检索，不发起 HTTP。"""

    def __init__(self, table: LocalReviewTable | None = None) -> None:
        if table is not None:
            self._table = table
            return
        # 旧口碑分支已从主 Pipeline 断开；保留 provider 仅供独立测试/实验使用，
        # 因此不再读取已废弃的 LOCAL_REVIEW_TABLE_PATH 配置。
        self._table = LocalReviewTable.default()

    def search_reviews(self, query: str, *, top_k: int = 8) -> list[ReviewHit]:
        return self._table.search(query, top_k=top_k)
