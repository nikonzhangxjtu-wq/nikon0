"""对 question_public.csv 批量推理，导出 id,ret。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许在项目根执行 `python scripts/run_batch_submission.py` 时找到 `app` 包
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from app.services.pipeline import ChatPipeline

INPUT_PATH = Path("question_public.csv")
OUTPUT_PATH = Path("submission_v1.csv")


def load_questions(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    df = pd.read_csv(input_path)
    if "id" not in df.columns or "question" not in df.columns:
        raise ValueError("question_public.csv 必须包含列: id, question")

    df = df[["id", "question"]].copy()
    if df["id"].isna().any() or df["question"].isna().any():
        raise ValueError("id/question 列存在空值，请先清洗输入数据")

    duplicated = df[df.duplicated(subset=["id"], keep=False)]
    if not duplicated.empty:
        sample_ids = duplicated["id"].head(10).tolist()
        raise ValueError(f"id 存在重复，示例: {sample_ids}")

    return df


def run_batch(
    df: pd.DataFrame,
    pipeline: ChatPipeline,
    *,
    sleep_seconds: float = 0.0,
    progress_every: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        qid = row.id
        question = str(row.question)

        result = pipeline.run(question=question, images=[])
        rows.append({"id": qid, "ret": result.answer})

        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[INFO] 已处理 {idx}/{total}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    out_df = pd.DataFrame(rows, columns=["id", "ret"])
    return out_df


def validate_submission(input_df: pd.DataFrame, out_df: pd.DataFrame) -> None:
    if list(out_df.columns) != ["id", "ret"]:
        raise ValueError("输出列必须为: id, ret")

    if len(out_df) != len(input_df):
        raise ValueError(f"输出条数不匹配: input={len(input_df)} output={len(out_df)}")

    if out_df["id"].duplicated().any():
        raise ValueError("输出中 id 出现重复")

    in_ids = input_df["id"].tolist()
    out_ids = out_df["id"].tolist()
    if out_ids != in_ids:
        raise ValueError("输出 id 顺序或内容与 question_public.csv 不一致")


def print_sampling_for_manual_check(input_df: pd.DataFrame, out_df: pd.DataFrame) -> None:
    merged = input_df.merge(out_df, on="id", how="left")

    multi_turn = merged[merged["question"].astype(str).str.contains("\n", na=False)].head(3)
    english = merged[merged["question"].astype(str).str.contains(r"[A-Za-z]", regex=True, na=False)].head(3)
    cs = merged[
        merged["question"].astype(str).str.contains(
            r"退货|退款|换货|发票|物流|投诉|售后|保修|运费|维修", regex=True, na=False
        )
    ].head(3)

    def _print_group(name: str, frame: pd.DataFrame) -> None:
        print(f"[CHECK] {name} 抽样 {len(frame)} 条")
        for row in frame.itertuples(index=False):
            q = str(row.question).replace("\n", " ")
            r = str(row.ret).replace("\n", " ")
            print(f"  - id={row.id} | q={q[:80]} | ret={r[:80]}")

    _print_group("多轮问答类", multi_turn)
    _print_group("英文题", english)
    _print_group("纯客服题", cs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 submission 文件")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="输入 CSV，默认 question_public.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出 CSV，默认 submission_v1.csv")
    parser.add_argument("--sleep", type=float, default=0.0, help="每条请求间 sleep 秒数（限流）")
    parser.add_argument("--progress-every", type=int, default=50, help="进度打印间隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_df = load_questions(args.input)
    pipeline = ChatPipeline()

    out_df = run_batch(
        input_df,
        pipeline,
        sleep_seconds=args.sleep,
        progress_every=args.progress_every,
    )
    validate_submission(input_df, out_df)

    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[INFO] 已生成: {args.output.resolve()}")
    print("[INFO] 输出校验通过：id 一一对应，无遗漏、无重复，列名为 id,ret")

    print_sampling_for_manual_check(input_df, out_df)


if __name__ == "__main__":
    main()
