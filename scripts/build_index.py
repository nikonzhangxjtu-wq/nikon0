"""Build vector index from manuals.

This script is a guided TODO:
- It wires ingestion and indexing entry points.
- You should complete Milvus + llama-index storage here.
"""

from __future__ import annotations

from app.core.config import settings
from app.services.ingestion import ManualIngestionService


def main() -> None:
    """Build index entry."""
    ingestion = ManualIngestionService(settings.manual_dir)
    chunks = ingestion.parse_and_chunk()

    print(f"[INFO] parsed chunks: {len(chunks)}")
    if not chunks:
        print("[WARN] no chunks found. Implement parser/chunker in ingestion.py first.")
        return

    # TODO (you):
    # 1) Convert ManualChunk -> llama-index Document.
    # 2) Build embedding with nomic-embed-text.
    # 3) Write vectors + metadata into Milvus collection.
    # 4) Print indexing summary (count, failed docs, collection name).
    print("[TODO] Milvus indexing is not implemented yet.")


if __name__ == "__main__":
    main()
