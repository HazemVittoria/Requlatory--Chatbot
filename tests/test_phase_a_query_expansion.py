from __future__ import annotations

from pathlib import Path

from src import qa as qa_module


def test_phase_a_expands_continuous_processing_query(monkeypatch):
    calls: list[tuple[str, float]] = []

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
        del data_dir, scope, top_k, max_context_chunks, use_mmr, mmr_lambda, allowed_docs, anchor_terms, chunk_chars, min_chunk_chars
        calls.append((query, float(min_similarity)))
        ql = query.lower()
        if min_similarity >= 0.20 and "continuous manufacturing" in ql and "ich q13" in ql:
            return [
                {
                    "text": "Continuous manufacturing is described in ICH Q13.",
                    "file": "ICH_Q13_Step4_Guideline_2022_1116.pdf",
                    "page": 6,
                    "chunk_id": "p6_c1",
                    "_score": 0.21,
                }
            ]
        return []

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "What does ICH Q13 say about continuous processing?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert fallback_used is False
    assert len(facts) == 1
    assert facts[0].pdf == "ICH_Q13_Step4_Guideline_2022_1116.pdf"
    assert trace["queries_used"]
    assert any(ms == 0.20 and "continuous processing" in q.lower() for q, ms in calls)
    assert any(ms == 0.20 and "continuous manufacturing" in q.lower() and "ich q13" in q.lower() for q, ms in calls)
    assert not any(ms == 0.18 for _, ms in calls)
