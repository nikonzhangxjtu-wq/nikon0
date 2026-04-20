"""Cross-encoder 精排：LangChain ``Document`` + ``CrossEncoderReranker``。

依赖（请自行安装）：``langchain-community``、``sentence-transformers``、PyTorch。

加载模型时优先使用 Hugging Face **Hub 本地快照目录的绝对路径**（只读磁盘，不访问
``huggingface.co``）；若无缓存再回退到仓库 ID（可能联网）。
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import settings

_COMPRESSOR_CACHE: dict[tuple[str, bool], object] = {}
_MISSING_DEPS_WARNED = False


def _hub_cache_root() -> Path:
    """与 ``huggingface_hub`` 一致：``HF_HUB_CACHE`` 或 ``HF_HOME/hub`` 或默认路径。"""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache/huggingface/hub").resolve()


def _cached_snapshot_dir(repo_id: str) -> Path | None:
    """在 Hub 缓存里解析 ``repo_id`` 对应的本地 snapshot 目录（纯文件系统，无网络）。"""
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


def _resolve_cross_encoder_load_path(model_name: str) -> tuple[str, dict]:
    """返回 ``(传给 CrossEncoder 的路径或 repo_id, model_kwargs)``.

    若能在本机缓存找到快照，则使用**绝对路径**，避免对 Hub 发 HEAD/metadata 请求。
    """
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


def _cross_encoder_reranker(model_name: str):
    try:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as e:
        print(f"[ERROR] 导入失败，真实原因是: {e}")
        return None

    cache_key = (model_name, settings.rerank_hf_local_files_only)
    if cache_key not in _COMPRESSOR_CACHE:
        try:
            load_path, mkw = _resolve_cross_encoder_load_path(model_name)
            if settings.rerank_hf_local_files_only:
                lp = Path(load_path)
                if not lp.is_dir() or not (lp / "config.json").is_file():
                    print(
                        "[ERROR] RERANK_HF_LOCAL_FILES_ONLY：未找到可用本地快照目录（需含 "
                        "config.json）。请先完整下载模型或设置 HF_HOME/HF_HUB_CACHE。"
                        f" repo_id={model_name!r} hub={_hub_cache_root()}"
                    )
                    return None
            model = HuggingFaceCrossEncoder(model_name=load_path, model_kwargs=mkw)
            _COMPRESSOR_CACHE[cache_key] = CrossEncoderReranker(
                model=model,
                top_n=10_000,
            )
        except Exception as e:
            print(f"[ERROR] Reranker 初始化失败: {e}")
            return None

    return _COMPRESSOR_CACHE[cache_key]


def rerank_fused_hits(fused_hits: list[dict], query: str, top_k: int) -> list[dict]:
    """将 Milvus 融合后的 ``hit`` 列表（``entity`` + ``score``）按 query 精排。

    未安装依赖、关闭开关或失败时退回融合顺序（截断 ``top_k``）。
    写入 ``RetrievedChunk.score`` 时使用名次映射到 ``(0, 1]``，避免 CE logit 与阈值过滤不一致。
    """
    global _MISSING_DEPS_WARNED
    if not fused_hits or top_k <= 0:
        return []
    q = query.strip()
    if not q:
        return fused_hits[:top_k]

    if not settings.rerank_enabled:
        return fused_hits[:top_k]

    model = (settings.rerank_model_name or "").strip()
    if not model:
        return fused_hits[:top_k]

    compressor = _cross_encoder_reranker(model)
    if compressor is None:
        if not _MISSING_DEPS_WARNED:
            print(
                "[WARN] 跳过精排：未安装 langchain-community / sentence-transformers "
                "（或导入失败）。"
            )
            _MISSING_DEPS_WARNED = True
        return fused_hits[:top_k]

    docs: list[Document] = []
    for i, hit in enumerate(fused_hits):
        entity = hit.get("entity", {}) or {}
        text = str(entity.get("text", "")).strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"_hit_index": i},
            )
        )

    if not docs:
        return fused_hits[:top_k]

    try:
        reranked = compressor.compress_documents(documents=docs, query=q)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] CrossEncoder rerank 失败，使用融合顺序: {exc}")
        return fused_hits[:top_k]

    out: list[dict] = []
    for rank, doc in enumerate(reranked[:top_k]):
        idx = doc.metadata.get("_hit_index")
        if idx is None:
            # metadata 若被覆盖，按正文回退匹配
            for i, hit in enumerate(fused_hits):
                ent = hit.get("entity", {}) or {}
                if str(ent.get("text", "")).strip() == doc.page_content.strip():
                    idx = i
                    break
        if idx is None:
            continue
        base = fused_hits[int(idx)]
        entity = base.get("entity", {}) or {}
        # 名次分：与 retriever_context_filter 兼容（均在 (0,1]）
        denom = float(top_k) if top_k else 1.0
        score = float(top_k - rank) / denom
        out.append({"entity": entity, "score": score})
    return out if out else fused_hits[:top_k]

