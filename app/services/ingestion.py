"""Manual ingestion and chunk preparation.

This module is intentionally left as a structured TODO map so you can
implement your own parsing/chunking strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ManualChunk:
    """Normalized chunk object for indexing."""

    chunk_id: str
    manual_name: str
    text: str
    image_ids: list[str] = field(default_factory=list)


class ManualIngestionService:
    """Parse and chunk manual files under MANUAL_DIR.

    TODO (you):
    1) Parse each `.txt` manual file.
    2) Extract core text, `<PIC>` positions, and trailing image id list.
    3) Chunk by headings/steps/troubleshooting blocks.
    4) Bind chunk-level image_ids.
    5) Return clean ManualChunk objects for indexing.
    """

    def __init__(self, manual_dir: str) -> None:
        self.manual_dir = Path(manual_dir)

    def load_manual_files(self) -> list[Path]:
        """List manual files in source directory."""
        return sorted(self.manual_dir.glob("*.txt"))

    def parse_and_chunk(self) -> list[ManualChunk]:
        """Main ingestion method.

        Placeholder implementation:
        - returns an empty list intentionally.
        """
        # TODO: implement your parser/chunker here.
        return []
