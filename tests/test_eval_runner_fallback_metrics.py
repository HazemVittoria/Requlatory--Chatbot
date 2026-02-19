from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import eval_runner as er


class _FakeCitation:
    def __init__(self):
        self.doc_id = "x.pdf"
        self.page = 1
        self.chunk_id = "p1_c1"


class _FakeResult:
    def __init__(self, text: str, citations: list, phase_a_fallback_used: bool):
        self.text = text
        self.citations = citations
        self.phase_a_fallback_used = phase_a_fallback_used


def test_fallback_usage_metrics_and_threshold_not_exceeded(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    rows = []
    for i in range(1, 11):
        rows.append({"id": i, "question": f"q{i}", "expected_not_found": False})
    rows.append({"id": 11, "question": "q11", "expected_not_found": True})
    golden.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def _fake_answer(question: str, scope: str = "MIXED", case_id: str | None = None):
        del scope, case_id
        # 1/10 answerable queries uses fallback => 0.10 (below 0.15 threshold).
        if question == "q1":
            return _FakeResult("Some answer", [_FakeCitation()], True)
        if question == "q11":
            return _FakeResult("Not found in provided PDFs", [], True)
        return _FakeResult("Some answer", [_FakeCitation()], False)

    monkeypatch.setattr(er, "answer", _fake_answer)
    out = er.run_eval(golden)
    assert out["metrics"]["phase_a_fallback_used_count"] == 2
    assert out["metrics"]["phase_a_fallback_used_rate"] == pytest.approx(2 / 11)
    assert out["metrics"]["fallback_used_rate_answerable"] == pytest.approx(1 / 10)
    assert out["alerts"]["fallback_used_rate_answerable_exceeded"] is False
    assert out["ci_fail"] is False


def test_fallback_usage_threshold_exceeded_sets_ci_fail(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    rows = [
        {"id": 1, "question": "q1", "expected_not_found": False},
        {"id": 2, "question": "q2", "expected_not_found": False},
        {"id": 3, "question": "q3", "expected_not_found": False},
        {"id": 4, "question": "q4", "expected_not_found": False},
        {"id": 5, "question": "q5", "expected_not_found": False},
    ]
    golden.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def _fake_answer(question: str, scope: str = "MIXED", case_id: str | None = None):
        del scope, case_id
        if question in {"q1", "q2"}:
            return _FakeResult("Some answer", [_FakeCitation()], True)
        return _FakeResult("Some answer", [_FakeCitation()], False)

    monkeypatch.setattr(er, "answer", _fake_answer)
    out = er.run_eval(golden)
    assert out["metrics"]["fallback_used_rate_answerable"] == pytest.approx(2 / 5)
    assert out["alerts"]["fallback_used_rate_answerable_exceeded"] is True
    assert out["alerts"]["message"] == (
        "Fallback retrieval triggered too often; check embeddings/index changes "
        "or similarity calibration."
    )
    assert out["ci_fail"] is True
