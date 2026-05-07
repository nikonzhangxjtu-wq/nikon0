"""手册解析与切块：将 `手册/*.txt` 转为可建索引的 `ManualChunk`。

支持的文件格式：

1. **单行**：与其它手册一致，整文件一行 Python list/tuple 字面量::

    [\"长文本...<PIC>...\", [\"Manual1_0\", \"drill0_01\", ...]]

2. **多行汇总**：每行一条独立的 `[正文, image_ids]` 字面量（常见于 ``汇总*.txt``），解析后按顺序拼接正文并串联 ``image_ids``，当作一本虚拟手册参与切块。

- 正文里用 `<PIC>` 占位表示插图位置；
- 尾部列表按 **出现顺序** 与每个 `<PIC>` 一一对应（若数量不一致，按较短一侧对齐并记录警告）。

切块策略（V2）：
1. 用 `ast.literal_eval` 解析字面量（单行 / 合法折行单行 / 或多行汇总合并后）；
2. 扫描全文记录每个 `<PIC>` 的起始下标，顺序绑定 `image_ids[i]`；
3. 使用 `RecursiveCharacterTextSplitter`（与 `app/test/test_splite.py` 相同的 separators / chunk_size / overlap）切分正文；
4. 每个块按块内 `<PIC>` 出现顺序，从全局 `<PIC>` 序列中依次取对应的图片 ID（去重、保序）。
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PIC_MARKER = "<PIC>"
_IMG_ID_PATTERN = re.compile(r"<IMG:([^>]+)>")


@dataclass
class ManualChunk:
    """用于建索引的标准化切片。"""

    chunk_id: str
    manual_name: str
    text: str
    image_ids: list[str] = field(default_factory=list)


def _root_literal_to_body_and_ids(data: object) -> tuple[str, list[str]]:
    """把 ``ast.literal_eval`` 得到的 list/tuple 根节点转为 (正文, image_ids)。"""
    if not isinstance(data, (list, tuple)) or len(data) < 1:
        raise ValueError("根节点应为长度>=1的 list/tuple")
    body = data[0]
    if not isinstance(body, str):
        raise ValueError(f"第一个元素应为 str，实际为 {type(body)}")
    image_ids: list[str] = []
    if len(data) >= 2 and data[1] is not None:
        if not isinstance(data[1], list):
            raise ValueError("第二个元素应为图片 ID 列表")
        image_ids = [str(x) for x in data[1]]
    return body, image_ids


def _parse_manual_file_raw(content: str) -> tuple[str, list[str]]:
    """解析文件为 (合并后正文, 合并后 image_ids)。

    - 单行：整文件一个 ``[str, list]`` 字面量（与其它 ``*.txt`` 一致）。
    - 多行汇总：每行一个完整字面量；多行合法的「单条 list 折行」整文件仍可被
      ``literal_eval`` 接受时走整文件路径。
    """
    content = content.strip()
    if not content:
        raise ValueError("文件为空")

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    # 最常见：单行字面量（含一行结束无换行的文件）
    if len(lines) == 1:
        try:
            data = ast.literal_eval(lines[0])
            return _root_literal_to_body_and_ids(data)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法用 ast.literal_eval 解析: {e}") from e

    # 多物理行：先尝试「整文件作为一个表达式」（合法折行的单个 list/tuple）
    try:
        data = ast.literal_eval(content)
        return _root_literal_to_body_and_ids(data)
    except (ValueError, SyntaxError):
        pass

    # 多行汇总：每行一条 [正文, ids]，顺序合并为一本手册
    bodies: list[str] = []
    all_ids: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        try:
            data = ast.literal_eval(line)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法用 ast.literal_eval 解析（第 {lineno} 行）: {e}") from e
        body, ids = _root_literal_to_body_and_ids(data)
        bodies.append(body)
        all_ids.extend(ids)

    merged_body = "\n\n".join(bodies)
    return merged_body, all_ids


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


def _embed_image_ids_in_text(text: str, image_ids: list[str]) -> str:
    """分块前将 `<PIC>` 替换为 `<IMG:actual_id>`，从后往前避免位置偏移。"""
    pic_positions = _pic_positions_and_binding(text, image_ids)
    result = text
    for pos, img_id in reversed(pic_positions):
        if not img_id:
            continue
        result = result[:pos] + f"<IMG:{img_id}>" + result[pos + len(PIC_MARKER):]
    return result


def _repair_split_img_tags(chunks: list[str]) -> list[str]:
    """修复分块导致的 `<IMG:...>` 标签断裂（末尾未闭合则接到下一块开头）。"""
    repaired: list[str] = []
    carry = ""
    for chunk in chunks:
        chunk = carry + chunk
        carry = ""
        last_open = chunk.rfind("<IMG:")
        if last_open >= 0:
            close = chunk.find(">", last_open)
            if close == -1:
                carry = chunk[last_open:]
                chunk = chunk[:last_open]
        repaired.append(chunk)
    if carry:
        repaired[-1] += carry
    return [c for c in repaired if c.strip()]


def _image_ids_from_chunk_text(chunk_text: str) -> list[str]:
    """从已嵌入 `<IMG:xxx>` 的文本中提取 image_id 列表（保序去重）。"""
    ids = _IMG_ID_PATTERN.findall(chunk_text)
    seen: set[str] = set()
    result: list[str] = []
    for iid in ids:
        if iid and iid not in seen:
            seen.add(iid)
            result.append(iid)
    return result


_CUSTOM_SPLIT_SEPARATORS = [
    r"\n+#\s+",  # 匹配带有一个或多个换行的标题
    r"\n+\d+[\.）\)]?\s+",  # 匹配正常步骤
    r"\n+[a-zA-Z][\.）\)]\s+",  # 匹配字母步骤
    r"\n+[·\-\u00b7\u25cf\u2022]\s+",  # 匹配带换行的无序列表
    r"\s+[·\u00b7\u25cf\u2022]\s+",  # 专门抓取没换行的内联无序列表
    r"\n\n",
    r"\n",
    r" ",
    r"",
]


def _make_recursive_splitter(*, chunk_size: int, chunk_overlap: int) -> "RecursiveCharacterTextSplitter":
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "缺少依赖 `langchain-text-splitters`：请按 README 安装后再运行建索引/切块流程。"
        ) from exc

    return _RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_CUSTOM_SPLIT_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
    )


def _image_ids_for_chunk_by_pic_occurrence(
    chunk_text: str,
    pic_positions: list[tuple[int, str]],
    occ_index: list[int],
) -> list[str]:
    """按 chunk 内 `<PIC>` 的出现顺序，从全局 `<PIC>` 序列依次取绑定 ID。"""
    if not chunk_text or not pic_positions:
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    idx = occ_index[0]
    for _ in re.finditer(re.escape(PIC_MARKER), chunk_text):
        pid = pic_positions[idx][1] if idx < len(pic_positions) else ""
        idx += 1
        if pid and pid not in seen:
            seen.add(pid)
            ordered.append(pid)

    occ_index[0] = idx
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
        body, image_ids = _parse_manual_file_raw(raw)
        # 文件名作为手册名称
        manual_name = path.stem

        # 分块前：将 `<PIC>` 替换为 `<IMG:actual_id>`，图片信息直接嵌入文本
        body_with_imgs = _embed_image_ids_in_text(body, image_ids)

        _ = (max_chars, min_chunk_chars)
        splitter = _make_recursive_splitter(chunk_size=800, chunk_overlap=120)
        split_texts = splitter.split_text(body_with_imgs)

        # 修复分块可能导致的 `<IMG:...>` 标签断裂
        split_texts = _repair_split_img_tags(split_texts)

        chunks: list[ManualChunk] = []
        for i, chunk_text in enumerate(split_texts):
            chunk_text = _sanitize_chunk_text(chunk_text, keep_pic=not strip_pic_markers)
            if not chunk_text:
                continue
            # 从文本中直接提取 image_ids，天然一致
            ids = _image_ids_from_chunk_text(chunk_text)
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
