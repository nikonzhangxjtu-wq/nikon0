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


def _keyword_overlap_score(query: str, chunk_text: str) -> float:
    """计算 query 与 chunk 的关键词重叠度 (Jaccard-like)。

    中文用字符 bigram，英文用单词，返回 [0, 1] 之间的得分。
    """
    import re

    q_lower = query.lower()
    c_lower = chunk_text.lower()

    # 英文单词
    en_words_q = set(re.findall(r'[a-zA-Z0-9]{2,}', q_lower))
    en_words_c = set(re.findall(r'[a-zA-Z0-9]{2,}', c_lower))

    # 中文字符 bigram
    cn_chars_q = re.sub(r'[a-zA-Z0-9\s]', '', q_lower)
    cn_chars_c = re.sub(r'[a-zA-Z0-9\s]', '', c_lower)
    cn_bigrams_q = {cn_chars_q[i:i+2] for i in range(len(cn_chars_q) - 1)} if len(cn_chars_q) >= 2 else set()
    cn_bigrams_c = {cn_chars_c[i:i+2] for i in range(len(cn_chars_c) - 1)} if len(cn_chars_c) >= 2 else set()

    # 数字（型号、参数等）
    nums_q = set(re.findall(r'\d+', q_lower))
    nums_c = set(re.findall(r'\d+', c_lower))

    all_q = en_words_q | cn_bigrams_q | nums_q
    all_c = en_words_c | cn_bigrams_c | nums_c

    if not all_q:
        return 0.0

    intersection = all_q & all_c
    # Jaccard: intersection / (union 按 query 大小归一化避免长文档偏差)
    return len(intersection) / max(len(all_q), 1)


def rerank_fused_hits(fused_hits: list[dict], query: str, top_k: int) -> list[dict]:
    """用 CrossEncoder + 关键词重叠混合精排。

    未安装依赖、关闭开关或失败时退回融合顺序（截断 top_k）。

    混合策略：
    - softmax(CrossEncoder raw scores, temperature) → 语义相关性
    - Jaccard keyword overlap → 精确术语匹配
    - 加权融合：keyword_weight 控制术语匹配权重（默认 0.3）
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

    # CrossEncoder 打分
    try:
        raw_scores = ce_model.predict(list(zip([q] * len(texts), texts)))
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        else:
            raw_scores = list(raw_scores)
    except Exception as exc:
        print(f"[WARN] CrossEncoder 打分失败，使用融合顺序: {exc}")
        return fused_hits[:top_k]

    # 按 CrossEncoder 原始分数排序（直接用 raw score，保持最大区分度）
    scored = sorted(
        zip(indices, raw_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # softmax 归一化（仅用于输出 score 字段，不影响排序）
    temperature = settings.rerank_temperature
    clamped = [max(min(float(s), 50.0), -50.0) for _, s in scored]
    max_raw = max(clamped) if clamped else 0.0
    exp_scores = [math.exp((s - max_raw) / temperature) for s in clamped]
    exp_sum = sum(exp_scores) or 1.0
    softmax_scores = [es / exp_sum for es in exp_scores]

    blended = [(scored[i][0], softmax_scores[i]) for i in range(len(scored))]

    out: list[dict] = []
    for idx, score in blended[:top_k]:
        base = fused_hits[idx]
        entity = base.get("entity", {}) or {}
        out.append({"entity": entity, "score": round(score, 6)})

    return out if out else fused_hits[:top_k]
