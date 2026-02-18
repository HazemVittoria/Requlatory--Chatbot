from __future__ import annotations

from pathlib import Path

from src.eval_runner import load_golden


def test_golden_file_loads():
    cases = load_golden(Path("golden_set.jsonl"))
    assert len(cases) >= 1
    assert all(c.question for c in cases)
