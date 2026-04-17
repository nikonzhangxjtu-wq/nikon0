"""手册解析与切块：将 `手册/*.txt` 转为可建索引的 `ManualChunk`。

每份手册文件约定为 **单行 Python 字面量**，形如::

    [\"长文本...<PIC>...\", [\"Manual1_0\", \"drill0_01\", ...]]

- 正文里用 `<PIC>` 占位表示插图位置；
- 尾部列表按 **出现顺序** 与每个 `<PIC>` 一一对应（若数量不一致，按较短一侧对齐并记录警告）。

切块策略（V1 实用版）：
1. 用 `ast.literal_eval` 解析整文件；
2. 扫描全文记录每个 `<PIC>` 的起始下标，顺序绑定 `image_ids[i]`；
3. 按最大字符数滑动切块，并尽量在换行处截断，避免硬切句中；
4. 每个块携带「落在该块字符区间内的」图片 ID（去重、保序）。
"""

from __future__ import annotations

import ast
import logging
from operator import truth
import re
from dataclasses import dataclass, field
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

PIC_MARKER = "<PIC>"


@dataclass
class ManualChunk:
    """用于建索引的标准化切片。"""

    chunk_id: str
    manual_name: str
    text: str
    image_ids: list[str] = field(default_factory=list)


def _parse_manual_file_raw(content: str) -> tuple[str, list[str]]:
    """解析单行 Python 列表字面量，返回 (正文, 图片ID列表)。"""
    content = content.strip();
    if(not content):
        raise ValueError("文件为空")
    try:
        # 使用 ast.literal_eval 解析字符串为 Python 对象
        data = ast.literal_eval(content)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"无法用 ast.literal_eval 解析: {e}") from e

    if not isinstance(data, (list, tuple)) or len(data) < 1:
        raise ValueError("根节点应为长度>=1的 list/tuple")
    # 取第一个元素作为整本手册的正文字符串。
    body = data[0]
    # 第一个元素应为 str
    if not isinstance(body, str):
        raise ValueError(f"第一个元素应为 str，实际为 {type(body)}")

    # 第二个元素应为图片 ID 列表，先声明「图片 ID 是字符串列表」，并默认成空
    image_ids: list[str] = []

    if len(data) >= 2 and data[1] is not None:
        if not isinstance(data[1], list):
            raise ValueError("第二个元素应为图片 ID 列表")
        image_ids = [str(x) for x in data[1]]

    return body, image_ids


def _pic_positions_and_binding(text: str, image_ids: list[str]) -> list[tuple[int, str]]:
    """扫描 `<PIC>`，按顺序绑定 image_ids。返回 [(起始下标, 图片id), ...]。"""
    positions: list[tuple[int, str]] = []
    search_from = 0
    idx = 0
    while True:
        pos = text.find(PIC_MARKER, search_from)
        if pos == -1:
            break
        pid = image_ids[idx] if idx < len(image_ids) else ""
        if idx >= len(image_ids):
            logger.warning(
                "`<PIC>` 数量多于 image_ids：多出的占位无对应 ID（从第 %s 个 PIC 起）",
                idx + 1,
            )
        positions.append((pos, pid))
        idx += 1
        search_from = pos + len(PIC_MARKER)

    if idx < len(image_ids):
        logger.warning(
            "image_ids 数量多于 `<PIC>`：多余 %s 个 ID 将被忽略",
            len(image_ids) - idx,
        )
    return positions


def _chunk_spans(text: str, max_chars: int, min_chunk_chars: int) -> list[tuple[int, int]]:
    """将全文切成 [start, end) 区间列表。尽量在换行处断开。"""
    if max_chars < 200:
        max_chars = 200
    n = len(text)
    if n == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 在 (start, end] 内从后往前找换行，避免切得太碎
            window_start = start + min_chunk_chars
            nl = text.rfind("\n", window_start, end)
            if nl != -1 and nl > start:
                end = nl + 1
        spans.append((start, end))
        start = end
    return spans


def _image_ids_for_span(
    pic_positions: list[tuple[int, str]],
    span_start: int,
    span_end: int,
) -> list[str]:
    """落在 [span_start, span_end) 内的 PIC 所绑定的图片 ID，去重保序。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for pos, pid in pic_positions:
        if span_start <= pos < span_end and pid:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
    return ordered


def _sanitize_chunk_text(text: str, keep_pic: bool) -> str:
    """块内文本：可选保留或移除 `<PIC>` 占位（检索用常保留，便于模型知道有图）。"""
    if keep_pic:
        return text.strip()
    return re.sub(re.escape(PIC_MARKER) + r"\s*", "", text).strip()


class ManualIngestionService:
    """解析 `manual_dir` 下的 `.txt` 手册并切块。"""

    def __init__(self, manual_dir: str | None = None) -> None:
        # 允许传入覆盖 settings，便于测试
        self.manual_dir = Path(manual_dir or settings.manual_dir).resolve()

    def load_manual_files(self) -> list[Path]:
        """列出手册目录下所有 `.txt` 文件。"""
        return sorted(self.manual_dir.glob("*.txt"))

    def parse_one_file(
        self,
        path: Path,
        *,
        max_chars: int = 1200,
        min_chunk_chars: int = 400,
        strip_pic_markers: bool = False,
    ) -> list[ManualChunk]:
        """解析单个手册文件为 `ManualChunk` 列表。"""
        raw = path.read_text(encoding="utf-8")
        # 解析单行 Python 列表字面量，返回 (正文, 图片ID列表)。
        body, image_ids = _parse_manual_file_raw(raw)
        # 文件名作为手册名称
        manual_name = path.stem

        # 从正文中提取 `<PIC>` 位置，与尾部列表做顺序/局部对齐（先实现一种简单规则即可）
        pic_positions = _pic_positions_and_binding(body, image_ids)
        spans = _chunk_spans(body, max_chars=max_chars, min_chunk_chars=min_chunk_chars)

        chunks: list[ManualChunk] = []
        for i, (s, e) in enumerate(spans):
            chunk_text = body[s:e]
            chunk_text = _sanitize_chunk_text(chunk_text, keep_pic=not strip_pic_markers)
            if not chunk_text:
                continue
            ids = _image_ids_for_span(pic_positions, s, e)
            cid = f"{manual_name}_{i:04d}"
            chunks.append(
                ManualChunk(
                    chunk_id=cid,
                    manual_name=manual_name,
                    text=chunk_text,
                    image_ids=ids,
                )
            )
        return chunks

    def parse_and_chunk(
        self,
        *,
        max_chars: int = 1200,
        min_chunk_chars: int = 400,
        strip_pic_markers: bool = False,
    ) -> list[ManualChunk]:
        """遍历 `manual_dir` 下全部 `.txt`，合并为 `ManualChunk` 列表。"""
        all_chunks: list[ManualChunk] = []
        for path in self.load_manual_files():
            try:
                all_chunks.extend(
                    self.parse_one_file(
                        path,
                        max_chars=max_chars,
                        min_chunk_chars=min_chunk_chars,
                        strip_pic_markers=strip_pic_markers,
                    )
                )
            except Exception as exc:
                logger.exception("跳过无法解析的手册文件 %s: %s", path, exc)
        return all_chunks
