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


class _FakeResult:
    def __init__(self):
        self.text = "Not found in provided PDFs"
        self.citations = []
        self.phase_a_fallback_used = False
        self.relevant_facts = []
        self.debug_trace = {
            "phase_a": {
                "queries_used": ["q", "q expanded"],
                "thresholds_attempted": [0.2, 0.18],
                "attempts": [
                    {"min_similarity": 0.2, "top_results": []},
                    {"min_similarity": 0.18, "top_results": []},
                ],
                "fallback_used": False,
            },
            "phase_b": {"relevant_facts_kept": []},
            "phase_c": {"answer_sentences": [], "citations": []},
        }


def test_failure_case_writes_debug_artifact(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"id": 1, "question": "q", "expected_not_found": False}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(er, "validate_index_integrity", lambda print_report=True: _FakeIndexReport())
    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _FakeResult(),
    )

    out = er.run_eval(golden)
    assert out["passed"] == 0
    assert out["results"][0]["outcome"] == "incorrect_refusal"
    assert out["failure_artifacts"]["count"] == 1

    artifact_dir = tmp_path / out["failure_artifacts"]["path"]
    artifact_path = artifact_dir / "1.json"
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["question"] == "q"
    assert payload["expected_not_found"] is False
    assert payload["outcome"] == "incorrect_refusal"
    assert "phase_a" in payload and "queries_used" in payload["phase_a"]
    assert "phase_b" in payload and "relevant_facts_kept" in payload["phase_b"]
    assert "phase_c" in payload and "answer_sentences" in payload["phase_c"]
    assert payload["validator"]["ok"] is False
    assert payload["validator"]["reasons"] == ["incorrect_refusal"]


def test_success_only_run_has_no_failure_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"id": 1, "question": "q", "expected_not_found": True}) + "\n",
        encoding="utf-8",
    )

    class _OkResult(_FakeResult):
        def __init__(self):
            super().__init__()
            self.text = "Not found in provided PDFs"

    monkeypatch.setattr(er, "validate_index_integrity", lambda print_report=True: _FakeIndexReport())
    monkeypatch.setattr(
        er,
        "answer",
        lambda question, scope="MIXED", case_id=None: _OkResult(),
    )

    out = er.run_eval(golden)
    assert out["passed"] == 1
    assert "failure_artifacts" not in out
