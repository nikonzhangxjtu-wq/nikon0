"""Retrieval abstraction.

This file defines interfaces and V1 placeholder implementations so you can
focus on one piece at a time:
1) first run with placeholder retrieval,
2) then replace with Milvus + llama-index retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """Single retrieved context unit."""

    chunk_id: str
    text: str
    score: float
    manual_name: str = ""
    image_ids: list[str] = field(default_factory=list)


class VectorRetriever:
    """Retriever service contract.

    TODO (you):
    - Integrate Milvus connection.
    - Embed query with nomic-embed-text.
    - Retrieve top-k chunks and fill score/manual/image metadata.
    """

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        # Placeholder behavior for scaffold:
        # returns one fake chunk to keep the end-to-end API runnable.
        # Replace this with real vector retrieval.
        return [
            RetrievedChunk(
                chunk_id="placeholder_001",
                text=(
                    "This is a placeholder retrieval chunk. "
                    "Implement Milvus retrieval in app/services/retriever.py."
                ),
                score=0.0,
                manual_name="placeholder_manual",
                image_ids=[],
            )
        ]
