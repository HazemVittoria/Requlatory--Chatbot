from __future__ import annotations

import json
from pathlib import Path

from src import eval_runner as er


def test_load_golden_file_has_cases():
    cases = er.load_golden(Path("golden_set.jsonl"))
    assert len(cases) >= 1


def test_run_eval_citation_display_prefers_page_label(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"id": 1, "question": "q", "expected_not_found": False}) + "\n",
        encoding="utf-8",
    )

    class _C:
        doc_id = "x.pdf"
        page = 3
        pdf_index = 2
        page_label = "iii"
        chunk_id = "p3_c1"

    class _R:
        text = "Some answer"
        citations = [_C()]
        phase_a_fallback_used = False
        debug_trace = {}

    monkeypatch.setattr(
        er,
        "validate_index_integrity",
        lambda print_report=True: type(
            "_Idx",
            (),
            {
                "total_documents": 1,
                "chunks_per_document": {"x.pdf": 1},
                "total_chunks": 1,
                "embedding_model_version": "x",
                "index_hash": "y",
            },
        )(),
    )
    monkeypatch.setattr(er, "answer", lambda question, scope="MIXED", case_id=None: _R())

    out = er.run_eval(golden)
    assert out["results"][0]["citations"] == ["x.pdf|iii|p3_c1"]
