"""Cross-encoder 精排：直接使用 sentence-transformers CrossEncoder。

绕过 langchain 的 HuggingFaceCrossEncoder / CrossEncoderReranker 包装，
避免 langchain-core 版本升级导致 ``langchain_core.pydantic_v1`` 缺失问题。

依赖：``sentence-transformers``、PyTorch。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from app.core.config import settings

_CE_MODEL_CACHE: dict[tuple[str, bool], object] = {}
_MISSING_DEPS_WARNED = False


def _hub_cache_root() -> Path:
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache/huggingface/hub").resolve()


def _cached_snapshot_dir(repo_id: str) -> Path | None:
    rid = repo_id.strip()
    if not rid or "/" not in rid:
        return None
    snaps = _hub_cache_root() / f"models--{rid.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for child in snaps.iterdir():
        if not child.is_dir():
            continue
        if not (child / "config.json").is_file():
            continue
        mtime = child.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best = child
    return best


def _resolve_load_path(model_name: str) -> tuple[str, dict]:
    mn = model_name.strip()
    direct = Path(mn).expanduser()
    if direct.is_dir() and (direct / "config.json").is_file():
        return str(direct.resolve()), {}

    snap = _cached_snapshot_dir(mn)
    if snap is not None:
        return str(snap.resolve()), {}

    mkw: dict = {}
    if settings.rerank_hf_local_files_only:
        mkw["local_files_only"] = True
    return mn, mkw


def _load_cross_encoder(model_name: str):
    """加载 sentence-transformers CrossEncoder，带缓存。"""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        print(f"[ERROR] sentence-transformers 未安装，无法加载 CrossEncoder: {e}")
        return None

    cache_key = (model_name, settings.rerank_hf_local_files_only)
    if cache_key not in _CE_MODEL_CACHE:
        try:
            load_path, mkw = _resolve_load_path(model_name)
            if settings.rerank_hf_local_files_only:
                lp = Path(load_path)
                if not lp.is_dir() or not (lp / "config.json").is_file():
                    print(
                        "[ERROR] RERANK_HF_LOCAL_FILES_ONLY：未找到可用本地快照目录"
                        f" repo_id={model_name!r} hub={_hub_cache_root()}"
                    )
                    return None
            _CE_MODEL_CACHE[cache_key] = CrossEncoder(
                load_path,
                model_kwargs=mkw or None,
            )
        except Exception as e:
            print(f"[ERROR] CrossEncoder 初始化失败: {e}")
            return None

    return _CE_MODEL_CACHE[cache_key]


def rerank_fused_hits(fused_hits: list[dict], query: str, top_k: int) -> list[dict]:
    """用 sentence-transformers CrossEncoder 对融合 hit 列表精排。

    未安装依赖、关闭开关或失败时退回融合顺序（截断 top_k）。
    使用 sigmoid(raw_score) → (0, 1) 保留真实相关性差异。
    """
    global _MISSING_DEPS_WARNED
    if not fused_hits or top_k <= 0:
        return []
    q = query.strip()
    if not q:
        return fused_hits[:top_k]

    if not settings.rerank_enabled:
        return fused_hits[:top_k]

    model_name = (settings.rerank_model_name or "").strip()
    if not model_name:
        return fused_hits[:top_k]

    ce_model = _load_cross_encoder(model_name)
    if ce_model is None:
        if not _MISSING_DEPS_WARNED:
            print("[WARN] 跳过精排：CrossEncoder 加载失败，使用融合顺序")
            _MISSING_DEPS_WARNED = True
        return fused_hits[:top_k]

    # 收集有效文本
    texts: list[str] = []
    indices: list[int] = []
    for i, hit in enumerate(fused_hits):
        entity = hit.get("entity", {}) or {}
        text = str(entity.get("text", "")).strip()
        if not text:
            continue
        texts.append(text)
        indices.append(i)

    if not texts:
        return fused_hits[:top_k]

    # CrossEncoder 打分 + sigmoid 归一化
    try:
        raw_scores = ce_model.predict(list(zip([q] * len(texts), texts)))
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        else:
            raw_scores = list(raw_scores)
    except Exception as exc:
        print(f"[WARN] CrossEncoder 打分失败，使用融合顺序: {exc}")
        return fused_hits[:top_k]

    # 按原始分数降序排列
    scored = sorted(
        zip(indices, raw_scores, texts),
        key=lambda x: x[1],
        reverse=True,
    )

    out: list[dict] = []
    for idx, raw, _ in scored[:top_k]:
        base = fused_hits[idx]
        entity = base.get("entity", {}) or {}
        # sigmoid 归一化到 (0, 1)，数值溢出保护
        clamped = max(min(float(raw), 50.0), -50.0)
        sigmoid_score = 1.0 / (1.0 + math.exp(-clamped))
        out.append({"entity": entity, "score": sigmoid_score})

    return out if out else fused_hits[:top_k]
