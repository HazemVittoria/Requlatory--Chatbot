from __future__ import annotations

from pathlib import Path

from src import qa as qa_module


def test_phase_a_expands_generic_gmp_validation_query(monkeypatch):
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
        if (
            min_similarity >= 0.20
            and "annex 15" in ql
            and "21 cfr 211" in ql
            and "in accordance with" in ql
            and "must" in ql
            and "shall" in ql
        ):
            return [
                {
                    "text": "It is a GMP requirement that manufacturers control critical aspects through qualification and validation.",
                    "file": "pe-009-17-gmp-guide-xannexes.pdf",
                    "page": 219,
                    "chunk_id": "p219_c1",
                    "_score": 0.22,
                }
            ]
        return []

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "GMP require for validation",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert len(facts) == 1
    assert fallback_used is False
    assert facts[0].pdf == "pe-009-17-gmp-guide-xannexes.pdf"
    assert "gmp require for validation" in trace["queries_used"]
    assert any(
        "annex 15" in q
        and "21 cfr 211" in q
        and "in accordance with" in q
        and "must" in q
        and "shall" in q
        and "required" in q
        for q in trace["queries_used"]
    )
    assert not any(ms == 0.18 for _, ms in calls)


def test_phase_a_does_not_add_requirement_anchor_expansion_for_procedural_query():
    variants = qa_module._phase_a_query_variants("How to investigate OOS?")
    assert variants
    assert not any(
        "annex 15" in q or "21 cfr 211" in q or "in accordance with" in q for q in variants
    )


def test_phase_a_requirement_anchor_expansion_is_deterministic():
    q = "What does GMP require for validation?"
    v1 = qa_module._phase_a_query_variants(q)
    v2 = qa_module._phase_a_query_variants(q)
    assert v1 == v2
