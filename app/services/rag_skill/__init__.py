"""RAG 相关小技能模块（查询构造、后处理等）。

注意：子模块文件名为 ``query_construction.py``，因此请勿写::

    from app.services.rag_skill import query_construction

这会把 ``query_construction`` 绑定为 **模块对象**（子模块），调用时会报
``TypeError: 'module' object is not callable``。

应使用::

    from app.services.rag_skill.query_construction import query_construction
"""

from __future__ import annotations

__all__: list[str] = []


def __getattr__(name: str):
    """支持 ``import app.services.rag_skill as r; r.query_construction(...)``。"""
    if name == "query_construction":
        from app.services.rag_skill.query_construction import query_construction as qc

        return qc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
