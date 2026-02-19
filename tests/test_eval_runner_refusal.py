from __future__ import annotations

import json
from pathlib import Path

from src import eval_runner as er


class _FakeResult:
    def __init__(self, text: str, citations: list):
        self.text = text
        self.citations = citations


def test_expected_not_found_passes_on_exact_refusal(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": 1,
                "question": "q",
                "expected_not_found": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _FakeResult("Not found in provided PDFs", []),
    )
    out = er.run_eval(golden)
    assert out["passed"] == 1
    assert out["metrics"]["correct_refusal"] == 1
    assert out["metrics"]["hallucination"] == 0


def test_expected_not_found_fails_if_content_returned(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": 1,
                "question": "q",
                "expected_not_found": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _C:
        def __init__(self):
            self.doc_id = "x.pdf"
            self.page = 1
            self.chunk_id = "p1_c1"

    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _FakeResult("Some answer", [_C()]),
    )
    out = er.run_eval(golden)
    assert out["passed"] == 0
    assert out["metrics"]["hallucination"] == 1
    assert out["results"][0]["outcome"] == "hallucination"
