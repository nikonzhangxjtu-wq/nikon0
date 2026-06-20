"""手册图片资产解析。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from app.services.ingestion import ManualChunk
from app.services.multimodal.types import ManualImageAsset, ManualImageAssetReport

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def file_sha256(path: Path) -> str:
    """计算文件 hash，用于判断图片理解缓存是否失效。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def scan_image_files(image_dir: str | Path) -> tuple[dict[str, Path], list[list[str]], list[list[str]]]:
    """扫描插图目录，返回 stem->path 以及重复/大小写冲突报告。"""
    root = Path(image_dir).resolve()
    by_id: dict[str, Path] = {}
    duplicate_paths: defaultdict[str, list[str]] = defaultdict(list)
    case_groups: defaultdict[str, list[str]] = defaultdict(list)
    # 目录不存在
    if not root.exists():
        return {}, [], []
    # 遍历目录下的所有文件
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        image_id = path.stem  # Blower_02.png → "Blower_02"
        duplicate_paths[image_id].append(str(path)) # 同一stem出现多次就记录下来
        case_groups[image_id.lower()].append(image_id) # 把真实stem归到小写id桶内
        """setdefault 意味着：若目录里同时有 Blower_02.png 和 Blower_02.jpg，by_id["Blower_02"] 只会指向排序后先遇到的那个文件。"""
        by_id.setdefault(image_id, path.resolve()) # 只保留第一次见到那个stem对应的路径

    duplicate_ids = [paths for paths in duplicate_paths.values() if len(paths) > 1]
    case_conflicts: list[list[str]] = []
    for ids in case_groups.values():
        unique = sorted(set(ids))
        if len(unique) > 1:
            case_conflicts.append(unique)

    return by_id, duplicate_ids, case_conflicts


def build_parent_links(chunks: list[ManualChunk]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """从 chunk.image_ids 建立 image_id 到父 chunk/手册名的反向索引。"""
    parent_chunks: defaultdict[str, list[str]] = defaultdict(list)
    parent_manuals: defaultdict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for image_id in chunk.image_ids:
            if not image_id:
                continue
            if chunk.chunk_id not in parent_chunks[image_id]:
                parent_chunks[image_id].append(chunk.chunk_id)
            if chunk.manual_name and chunk.manual_name not in parent_manuals[image_id]:
                parent_manuals[image_id].append(chunk.manual_name)
    return dict(parent_chunks), dict(parent_manuals)

# 一张图如何挂载到父chunk上
def build_manual_image_catalog(
    *, chunks: list[ManualChunk], image_dir: str | Path
) -> tuple[dict[str, ManualImageAsset], ManualImageAssetReport]:
    """建立 image_id -> ManualImageAsset，并返回资产质量报告。"""
    files_by_id, duplicate_ids, case_conflicts = scan_image_files(image_dir)
    parent_chunks, parent_manuals = build_parent_links(chunks)
    referenced = set(parent_chunks)
    available = set(files_by_id)

    missing = sorted(referenced - available)
    orphan = sorted(available - referenced)

    catalog: dict[str, ManualImageAsset] = {}
    for image_id in sorted(referenced & available):
        path = files_by_id[image_id]
        width, height = _image_size(path)
        catalog[image_id] = ManualImageAsset(
            image_id=image_id,
            image_path=path,
            parent_chunk_ids=parent_chunks.get(image_id, []),
            parent_manual_names=parent_manuals.get(image_id, []),
            width=width,
            height=height,
            file_hash=file_sha256(path),
        )

    return catalog, ManualImageAssetReport(
        missing_images=missing,
        orphan_images=orphan,
        duplicate_ids=duplicate_ids,
        case_conflicts=case_conflicts,
    )
