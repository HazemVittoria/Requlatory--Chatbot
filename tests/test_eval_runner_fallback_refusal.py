from __future__ import annotations

import json
from pathlib import Path

from src import eval_runner as er
from src import qa


def test_fallback_retrieval_still_results_in_correct_refusal(monkeypatch, tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": 49,
                "question": "What is the market share of PIC/S member countries?",
                "expected_not_found": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[float] = []

    def _fake_search_chunks(
        query: str,
        *,
        data_dir: Path = Path("data"),
        scope: str = "MIXED",
        top_k: int = 8,
        max_context_chunks: int = 6,
        min_similarity: float = 0.20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.60,
        allowed_docs: list[str] | None = None,
        anchor_terms: list[str] | None = None,
        chunk_chars: int = 1000,
        min_chunk_chars: int = 220,
    ) -> list[dict]:
        del query, data_dir, scope, top_k, max_context_chunks, use_mmr, mmr_lambda, allowed_docs, anchor_terms, chunk_chars, min_chunk_chars
        calls.append(float(min_similarity))
        if min_similarity >= 0.20:
            return []
        return [
            {
                "text": (
                    "This section describes validation lifecycle activities, cleaning verification, "
                    "and equipment calibration controls for manufacturing systems."
                ),
                "file": "member_quality_guidance.pdf",
                "page": 12,
                "chunk_id": "p12_c1",
                "_score": 0.181,
            }
        ]

    monkeypatch.setattr(qa, "search_chunks", _fake_search_chunks)

    out = er.run_eval(golden)
    assert calls == [0.20, 0.20, 0.18]
    assert out["passed"] == 1
    assert out["metrics"]["correct_refusal"] == 1
    assert out["metrics"]["hallucination"] == 0
    assert out["results"][0]["outcome"] == "correct_refusal"
    assert out["results"][0]["answer"] == "Not found in provided PDFs"
    assert out["results"][0]["citations"] == []
