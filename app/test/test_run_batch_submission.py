from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

# 避免导入真实 pipeline（其依赖可能触发外部模型连接）
fake_pipeline_module = types.ModuleType("app.services.pipeline")


class _FakePipelineForImport:
    def run(self, question: str, images: list[str]):
        class _R:
            answer = "stub"

        return _R()


fake_pipeline_module.ChatPipeline = _FakePipelineForImport
sys.modules.setdefault("app.services.pipeline", fake_pipeline_module)

from scripts import run_batch_submission as rbs


class DummyPipeline:
    def run(self, question: str, images: list[str]):
        class R:
            answer = f"ans:{question[:8]}"

        return R()


def test_load_questions_ok_and_duplicate_guard():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "q.csv"
        p.write_text("id,question\n1,hello\n2,world\n", encoding="utf-8")
        df = rbs.load_questions(p)
        assert list(df.columns) == ["id", "question"]
        assert len(df) == 2

        dup = Path(td) / "dup.csv"
        dup.write_text("id,question\n1,a\n1,b\n", encoding="utf-8")
        try:
            rbs.load_questions(dup)
            assert False, "expected duplicate id ValueError"
        except ValueError as exc:
            assert "重复" in str(exc)


def test_run_batch_and_validate_submission():
    input_df = pd.DataFrame(
        [
            {"id": 101, "question": "请问如何安装"},
            {"id": 102, "question": "我要退款"},
        ]
    )

    out_df = rbs.run_batch(input_df, DummyPipeline(), sleep_seconds=0.0, progress_every=1)
    assert list(out_df.columns) == ["id", "ret"]
    assert out_df["id"].tolist() == [101, 102]
    assert out_df["ret"].iloc[0].startswith("ans:")

    rbs.validate_submission(input_df, out_df)


def test_validate_submission_rejects_id_mismatch():
    input_df = pd.DataFrame(
        [
            {"id": 1, "question": "q1"},
            {"id": 2, "question": "q2"},
        ]
    )
    out_df = pd.DataFrame(
        [
            {"id": 2, "ret": "a2"},
            {"id": 1, "ret": "a1"},
        ]
    )
    try:
        rbs.validate_submission(input_df, out_df)
        assert False, "expected id order/content mismatch ValueError"
    except ValueError as exc:
        assert "顺序或内容" in str(exc)


def test_print_sampling_for_manual_check_runs(capsys):
    input_df = pd.DataFrame(
        [
            {"id": 1, "question": "line1\nline2"},
            {"id": 2, "question": "How to install?"},
            {"id": 3, "question": "我要退款"},
        ]
    )
    out_df = pd.DataFrame(
        [
            {"id": 1, "ret": "r1"},
            {"id": 2, "ret": "r2"},
            {"id": 3, "ret": "r3"},
        ]
    )

    rbs.print_sampling_for_manual_check(input_df, out_df)
    out = capsys.readouterr().out
    assert "多轮问答类" in out
    assert "英文题" in out
    assert "纯客服题" in out


if __name__ == "__main__":
    test_load_questions_ok_and_duplicate_guard()
    test_run_batch_and_validate_submission()
    test_validate_submission_rejects_id_mismatch()

    class _Cap:
        class _R:
            out = ""

        def readouterr(self):
            return self._R()

    print("[OK] test_run_batch_submission basic tests passed")
