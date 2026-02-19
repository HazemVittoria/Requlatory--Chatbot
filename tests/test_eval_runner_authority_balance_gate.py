from __future__ import annotations

import json
from pathlib import Path

from src import eval_runner as er


class _FakeIndexReport:
    total_documents = 1
    chunks_per_document = {"x.pdf": 1}
    total_chunks = 1
    embedding_model_version = "test"
    index_hash = "abc"


class _FakeCitation:
    def __init__(self):
        self.doc_id = "x.pdf"
        self.page = 1
        self.chunk_id = "p1_c1"


class _FakeResult:
    def __init__(self, phase_a_trace: dict):
        self.text = "Some answer"
        self.citations = [_FakeCitation()]
        self.phase_a_fallback_used = False
        self.debug_trace = {"phase_a": phase_a_trace}


def _write_case(path: Path) -> None:
    row = {
        "id": 1,
        "question": "What does GMP require for validation?",
        "expected_not_found": False,
        "phase_a_require_authorities_when_eligible": ["EU_GMP", "FDA"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_balance_gate_fails_when_eligible_but_missing(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    _write_case(golden)

    phase_a_trace = {
        "attempts": [
            {
                "min_similarity": 0.2,
                "top_results": [
                    {"authority": "FDA", "score": 0.45},
                    {"authority": "EU_GMP", "score": 0.43},
                    {"authority": "ICH", "score": 0.41},
                ],
            }
        ],
        "fallback_used": False,
        "balancing": {
            "soft_floor_score": 0.40,
            "authorities_kept": ["FDA"],
        },
    }

    monkeypatch.setattr(er, "validate_index_integrity", lambda print_report=True: _FakeIndexReport())
    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _FakeResult(phase_a_trace),
    )

    out = er.run_eval(golden)
    assert out["passed"] == 0
    assert "phase_a_balance_missing:EU_GMP" in out["results"][0]["fail_reasons"]


def test_balance_gate_skips_when_not_eligible(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    _write_case(golden)

    phase_a_trace = {
        "attempts": [
            {
                "min_similarity": 0.2,
                "top_results": [
                    {"authority": "FDA", "score": 0.45},
                    {"authority": "EU_GMP", "score": 0.35},
                ],
            }
        ],
        "fallback_used": False,
        "balancing": {
            "soft_floor_score": 0.40,
            "authorities_kept": ["FDA"],
        },
    }

    monkeypatch.setattr(er, "validate_index_integrity", lambda print_report=True: _FakeIndexReport())
    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _FakeResult(phase_a_trace),
    )

    out = er.run_eval(golden)
    assert out["passed"] == 1
    assert out["results"][0]["ok"] is True
