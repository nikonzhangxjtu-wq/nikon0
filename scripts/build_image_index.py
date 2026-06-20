"""构建手册图片多模态索引。

流程：解析文本 chunk 的 `<IMG:...>` 关系 → 扫描插图目录 → VLM 结构理解 →
Jina 图文向量 → 写入 Milvus `manual_images_v1`。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pymilvus import MilvusClient

from app.core.config import settings
from app.services.ingestion import ManualChunk, ManualIngestionService
from app.services.multimodal.types import ManualImageAsset
from app.services.milvus_create import build_image_collection, create_image_vector_index
from app.services.multimodal.catalog import build_manual_image_catalog
from app.services.multimodal.embeddings import JinaMultimodalEmbeddingClient
from app.services.multimodal.understanding import (
    ManualImageInterpreter,
    ManualImageUnderstanding,
    ManualImageUnderstandingCache,
)

VLM_LOG_RAW_MAX_CHARS = 1200

IMAGE_META_MAX_LEN = 8192
MILVUS_INSERT_BATCH = 16


class _BuildProgress:
    """索引构建进度：优先 tqdm，否则简易终端进度条。"""

    def __init__(self, *, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.done = 0
        self._t0 = time.monotonic()
        self._pbar: Any = None
        if tqdm is not None and total > 0:
            self._pbar = tqdm(
                total=total,
                desc=desc,
                unit="张",
                dynamic_ncols=True,
                file=sys.stdout,
            )

    def advance(self, *, image_id: str, stats: dict[str, int]) -> None:
        self.done += 1
        if self._pbar is not None:
            self._pbar.set_postfix(
                ok=stats.get("success", 0),
                fail=stats.get("failed", 0),
                cache=stats.get("vlm_cache", 0),
                flush=stats.get("milvus_flush", 0),
                refresh=False,
            )
            self._pbar.update(1)
            return
        if self.done == 1 or self.done == self.total or self.done % max(1, self.total // 80) == 0:
            self._print_plain(image_id=image_id, stats=stats)

    def _print_plain(self, *, image_id: str, stats: dict[str, int]) -> None:
        width = 32
        ratio = self.done / self.total if self.total else 1.0
        filled = int(width * ratio)
        bar = "=" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self._t0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0.0
        msg = (
            f"\r[{bar}] {self.done}/{self.total} ({ratio * 100:.1f}%) "
            f"ok={stats.get('success', 0)} fail={stats.get('failed', 0)} "
            f"cache={stats.get('vlm_cache', 0)} flush={stats.get('milvus_flush', 0)} "
            f"elapsed={_fmt_duration(elapsed)} eta={_fmt_duration(eta)} "
            f"last={image_id[:24]}"
        )
        print(msg, end="", flush=True)

    def close(self) -> float:
        if self._pbar is not None:
            self._pbar.close()
        elif self.total > 0:
            print()
        return time.monotonic() - self._t0

    @staticmethod
    def log(message: str) -> None:
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _step_banner(step: int, total_steps: int, title: str) -> None:
    _BuildProgress.log(f"[{step}/{total_steps}] {title}")


def _milvus_client() -> MilvusClient:
    kwargs: dict = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _truncate(value: str) -> str:
    return value[:IMAGE_META_MAX_LEN]


def _chunk_text_index(chunks: list[ManualChunk]) -> dict[str, ManualChunk]:
    return {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_id}

# TODO：信息提取
def _parent_context_text(asset: ManualImageAsset, chunk_by_id: dict[str, ManualChunk]) -> str:
    """拼接父 chunk 正文，供 VLM 判断图片在手册中的语义意图。"""
    parts: list[str] = []
    seen: set[str] = set()
    for chunk_id in asset.parent_chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        if not chunk:
            continue
        text = (chunk.text or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    merged = "\n\n".join(parts)
    return merged[:IMAGE_META_MAX_LEN]


def _image_row(
    *,
    asset: ManualImageAsset,
    image_vec: list[float],
    semantic_vec: list[float],
    understanding: ManualImageUnderstanding,
    semantic_text: str,
) -> dict:
    """与 milvus_create.build_image_collection 字段对齐，避免 insert 缺列。"""
    return {
        "image_id": asset.image_id,
        "image_vector": image_vec,
        "semantic_vector": semantic_vec,
        "image_path": str(asset.image_path),
        "manual_name": asset.parent_manual_names[0] if asset.parent_manual_names else "",
        "parent_chunk_ids": _truncate(_json(asset.parent_chunk_ids)),
        "parent_context_text": _truncate(understanding.parent_context_text or ""),
        "context_intent": _truncate(understanding.context_intent or ""),
        "image_type": understanding.image_type,
        "semantic_text": _truncate(semantic_text),
        "ocr_text": _truncate(_json(understanding.ocr_text)),
        "visual_entities": _truncate(_json(understanding.visual_entities())),
        "operation_steps": _truncate(_json(understanding.operation_steps)),
        "warnings": _truncate(_json(understanding.warnings)),
    }


def _write_report(report) -> None:
    path = Path(settings.manual_image_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "missing_images": report.missing_images,
                "orphan_images": report.orphan_images,
                "duplicate_ids": report.duplicate_ids,
                "case_conflicts": report.case_conflicts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _vlm_log_record(
    understanding: ManualImageUnderstanding,
    *,
    from_cache: bool,
) -> dict[str, object]:
    """单条可序列化记录，写入 jsonl 便于事后 grep/分析。"""
    record: dict[str, object] = {
        "image_id": understanding.image_id,
        "source": "cache" if from_cache else "api",
        "image_type": understanding.image_type,
        "context_intent": understanding.context_intent,
        "parent_context_text": understanding.parent_context_text,
        "ocr_text": understanding.ocr_text,
        "buttons": understanding.buttons,
        "indicators": understanding.indicators,
        "parts": understanding.parts,
        "operation_steps": understanding.operation_steps,
        "warnings": understanding.warnings,
        "relations": understanding.relations,
        "semantic_text": understanding.to_semantic_text(),
        "visual_entities": understanding.visual_entities(),
    }
    raw = (understanding.raw_text or "").strip()
    if raw:
        record["raw_text"] = raw[:VLM_LOG_RAW_MAX_CHARS]
        if len(raw) > VLM_LOG_RAW_MAX_CHARS:
            record["raw_text_truncated"] = True
    return record


def _append_vlm_log_jsonl(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_vlm_understanding(
    understanding: ManualImageUnderstanding,
    *,
    from_cache: bool,
    log_path: Path | None,
) -> None:
    if not settings.manual_image_vlm_log_enabled:
        return

    record = _vlm_log_record(understanding, from_cache=from_cache)
    if log_path is not None:
        _append_vlm_log_jsonl(log_path, record)

    if not settings.manual_image_vlm_log_console:
        return

    source = record["source"]
    _BuildProgress.log(
        f"[VLM] ===== {understanding.image_id} ({source}) "
        f"type={understanding.image_type} ====="
    )
    if understanding.context_intent:
        _BuildProgress.log(f"[VLM] intent: {understanding.context_intent}")
    if understanding.ocr_text:
        _BuildProgress.log(f"[VLM] ocr: {understanding.ocr_text}")
    if understanding.operation_steps:
        _BuildProgress.log(f"[VLM] steps: {understanding.operation_steps}")
    if understanding.warnings:
        _BuildProgress.log(f"[VLM] warnings: {understanding.warnings}")
    semantic = str(record.get("semantic_text") or "").strip()
    if semantic:
        for line in semantic.splitlines():
            _BuildProgress.log(f"[VLM] {line}")
    else:
        _BuildProgress.log("[VLM] (empty semantic_text — API/解析失败或模型未返回 JSON)")
    if understanding.raw_text and not from_cache:
        preview = understanding.raw_text.replace("\n", " ")[:300]
        _BuildProgress.log(f"[VLM] raw_preview: {preview}")


def _understand_image(
    *,
    image_id: str,
    image_path: Path,
    parent_context_text: str,
    cache: ManualImageUnderstandingCache,
    interpreter: ManualImageInterpreter,
) -> tuple[ManualImageUnderstanding, bool]:
    cached = cache.get(
        image_id=image_id,
        image_path=image_path,
        parent_context_text=parent_context_text,
    )
    if cached is not None:
        return cached, True
    try:
        item = interpreter.understand_image(
            image_id=image_id,
            image_path=image_path,
            parent_context_text=parent_context_text,
        )
    except Exception as exc:  # noqa: BLE001
        _BuildProgress.log(
            f"[WARN] 图片结构理解失败，使用空结构降级 image_id={image_id}: {exc}"
        )
        item = ManualImageUnderstanding(
            image_id=image_id,
            parent_context_text=parent_context_text,
        )
    cache.set(
        image_id=image_id,
        image_path=image_path,
        parent_context_text=parent_context_text,
        understanding=item,
    )
    return item, False


def main() -> None:
    t_start = time.monotonic()
    total_steps = 4

    provider = (settings.multimodal_embed_provider or "").strip().lower()
    if provider not in {"jina_api", "dashscope_multimodal", "bailian_multimodal"}:
        _BuildProgress.log(
            "[ERROR] 不支持的 MULTIMODAL_EMBED_PROVIDER；"
            f"实际={settings.multimodal_embed_provider}"
        )
        return
    if provider == "jina_api" and not settings.jina_api_key:
        _BuildProgress.log("[ERROR] 缺少 JINA_API_KEY，请在本地 .env 或 shell 环境中设置后重试。")
        return
    if provider in {"dashscope_multimodal", "bailian_multimodal"} and not (
        settings.dashscope_api_key or settings.bailian_api_key
    ):
        _BuildProgress.log("[ERROR] 缺少 DASHSCOPE_API_KEY（或 BAILIAN_API_KEY），请在本地 .env 或 shell 环境中设置后重试。")
        return

    _step_banner(1, total_steps, "解析手册 chunk 并扫描图片资产")
    ingestion = ManualIngestionService(settings.manual_dir)
    chunks = ingestion.parse_and_chunk()
    catalog, report = build_manual_image_catalog(
        chunks=chunks,
        image_dir=settings.manual_image_dir,
    )
    _write_report(report)
    _BuildProgress.log(
        "[INFO] 图片资产: "
        f"可索引={len(catalog)} missing={len(report.missing_images)} "
        f"orphan={len(report.orphan_images)} duplicate={len(report.duplicate_ids)} "
        f"case_conflict={len(report.case_conflicts)}"
    )
    if not catalog:
        _BuildProgress.log("[WARN] 未发现可索引图片，请检查 MANUAL_DIR / MANUAL_IMAGE_DIR 与 <IMG:id>。")
        return

    _step_banner(2, total_steps, "初始化 VLM 缓存与多模态 embedding 客户端")
    cache = ManualImageUnderstandingCache(settings.manual_image_cache_path)
    interpreter = ManualImageInterpreter()
    embedder = JinaMultimodalEmbeddingClient()
    client = _milvus_client()
    _BuildProgress.log(
        f"[INFO] VLM={interpreter.model or '(未配置)'} "
        f"embed_provider={provider} "
        f"embed={settings.multimodal_embed_model} "
        f"cache={settings.manual_image_cache_path}"
    )

    _step_banner(3, total_steps, "探测向量维度并创建 Milvus collection")
    assets = list(catalog.values())
    first_asset = assets[0]
    _BuildProgress.log(f"[INFO] 探测首张图片 embedding: {first_asset.image_id}")
    probe_vec = embedder.embed_image(first_asset.image_path)
    if not probe_vec:
        _BuildProgress.log("[ERROR] Jina 图片 embedding 返回空向量。")
        return
    dim = len(probe_vec)
    _BuildProgress.log(
        f"[INFO] 图片向量维度={dim} collection={settings.multimodal_image_collection}"
    )
    build_image_collection(client, vector_dim=dim)

    _step_banner(
        4,
        total_steps,
        f"VLM 结构理解 + 双向量 + 写入 Milvus（共 {len(assets)} 张，每 {MILVUS_INSERT_BATCH} 条 flush）",
    )
    if tqdm is None:
        _BuildProgress.log("[INFO] 未安装 tqdm，使用简易进度条；可选: pip install tqdm")

    chunk_by_id = _chunk_text_index(chunks)

    vlm_log_path: Path | None = None
    if settings.manual_image_vlm_log_enabled:
        vlm_log_path = Path(settings.manual_image_vlm_log_path)
        vlm_log_path.parent.mkdir(parents=True, exist_ok=True)
        vlm_log_path.write_text("", encoding="utf-8")
        _BuildProgress.log(
            f"[INFO] VLM 结构理解日志 jsonl: {vlm_log_path.resolve()} "
            f"(console={'on' if settings.manual_image_vlm_log_console else 'off'})"
        )

    success = 0
    failed = 0
    vlm_cache = 0
    milvus_flush = 0
    rows: list[dict] = []
    progress = _BuildProgress(total=len(assets), desc="索引图片")

    for idx, asset in enumerate(assets):
        try:
            # image_vec创建图片向量
            image_vec = probe_vec if idx == 0 else embedder.embed_image(asset.image_path)
            # 拼父chunk正文
            parent_context = _parent_context_text(asset, chunk_by_id)
            # vlm结构理解
            understanding, from_cache = _understand_image(
                image_id=asset.image_id,
                image_path=asset.image_path,
                parent_context_text=parent_context,
                cache=cache,
                interpreter=interpreter,
            )
            if from_cache:
                vlm_cache += 1
            _log_vlm_understanding(
                understanding,
                from_cache=from_cache,
                log_path=vlm_log_path,
            )
            # 根据vlm结构理解拼接semantic_text
            semantic_text = understanding.to_semantic_text() or asset.image_id
            # 根据semantic_text创建semantic_vec
            semantic_vec = embedder.embed_text(semantic_text)
            if len(image_vec) != dim or len(semantic_vec) != dim:
                raise RuntimeError(
                    f"向量维度不一致 image={len(image_vec)} semantic={len(semantic_vec)} expected={dim}"
                )
            rows.append(
                _image_row(
                    asset=asset,
                    image_vec=image_vec,
                    semantic_vec=semantic_vec,
                    understanding=understanding,
                    semantic_text=semantic_text,
                )
            )
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _BuildProgress.log(f"[ERROR] 图片索引行构建失败 image_id={asset.image_id}: {exc}")

        if len(rows) >= MILVUS_INSERT_BATCH:
            client.insert(collection_name=settings.multimodal_image_collection, data=rows)
            milvus_flush += len(rows)
            rows.clear()

        progress.advance(
            image_id=asset.image_id,
            stats={
                "success": success,
                "failed": failed,
                "vlm_cache": vlm_cache,
                "milvus_flush": milvus_flush,
            },
        )

    loop_elapsed = progress.close()

    if rows:
        client.insert(collection_name=settings.multimodal_image_collection, data=rows)
        milvus_flush += len(rows)
        rows.clear()

    _BuildProgress.log("[INFO] 创建图片 collection 向量索引…")
    index_ok = True
    try:
        create_image_vector_index(client)
    except Exception as exc:  # noqa: BLE001
        index_ok = False
        _BuildProgress.log(f"[ERROR] 图片 collection 建索引/load 失败: {exc}")

    total_elapsed = time.monotonic() - t_start
    _BuildProgress.log(
        "[INFO] 图片索引完成 "
        f"success={success} failed={failed} vlm_cache={vlm_cache} "
        f"milvus_rows={milvus_flush} collection={settings.multimodal_image_collection} "
        f"index_ok={index_ok} loop={_fmt_duration(loop_elapsed)} total={_fmt_duration(total_elapsed)}"
    )


if __name__ == "__main__":
    main()
