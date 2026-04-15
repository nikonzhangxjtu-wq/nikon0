"""Run batch inference over question_public.csv and export id,ret.

This script is intentionally simple and readable for first-project learning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.pipeline import ChatPipeline

INPUT_PATH = Path("question_public.csv")
OUTPUT_PATH = Path("submission_v1.csv")


def main() -> None:
    """Batch prediction entry."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    if "id" not in df.columns or "question" not in df.columns:
        raise ValueError("question_public.csv must include columns: id, question")

    pipeline = ChatPipeline()
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        qid = row["id"]
        question = str(row["question"])

        # NOTE:
        # This path uses empty images because public csv is text-only.
        # If you later have image-based evaluation, extend this row schema.
        result = pipeline.run(question=question, images=[])
        rows.append({"id": qid, "ret": result.answer})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[INFO] generated: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
