from __future__ import annotations

from pathlib import Path

from src import qa as qa_module


def test_is_apr_query_deterministic_rules():
    assert qa_module._is_apr_query("What is required in APR?")
    assert qa_module._is_apr_query("What is required in PQR?")
    assert qa_module._is_apr_query("What is required in an annual product review?")
    assert qa_module._is_apr_query("What data is in product quality review?")
    assert qa_module._is_apr_query("What should annual review include?")
    assert qa_module._is_apr_query("What does product review cover?")
    assert not qa_module._is_apr_query("What does GMP require for process validation?")


def test_phase_a_apr_query_passes_apr_anchor_terms(monkeypatch):
    monkeypatch.setattr(qa_module, "route_docs", lambda *args, **kwargs: [])
    calls: list[list[str] | None] = []
    call_queries: list[str] = []

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
        del data_dir, scope, top_k, max_context_chunks, min_similarity, use_mmr, mmr_lambda
        del allowed_docs, chunk_chars, min_chunk_chars
        calls.append(anchor_terms)
        call_queries.append(query)
        return [
            {
                "text": "Annual product review should include trend analysis and deviations.",
                "file": "Manual-022-Annual-Product-Reviews.pdf",
                "page": 2,
                "chunk_id": "p2_c1",
                "_score": 0.34,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "What is required in an Annual Product Review (APR) / Product Quality Review (PQR)?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert fallback_used is False
    assert len(facts) == 1
    assert trace["anchor_terms_used"] == [
        "annual product review",
        "product quality review",
        "pqr",
        "annual review",
        "quality review",
    ]
    assert "annual product review" in trace["expanded_query_used"]
    assert "product quality review" in trace["expanded_query_used"]
    assert all(calls)
    assert all("pqr" in (a or []) for a in calls)
    assert any("annual product review" in q and "product quality review" in q for q in call_queries)


def test_phase_a_non_apr_query_does_not_pass_apr_anchor_terms(monkeypatch):
    monkeypatch.setattr(qa_module, "route_docs", lambda *args, **kwargs: [])
    calls: list[list[str] | None] = []

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
        del query, data_dir, scope, top_k, max_context_chunks, min_similarity, use_mmr, mmr_lambda
        del allowed_docs, chunk_chars, min_chunk_chars
        calls.append(anchor_terms)
        return [
            {
                "text": "Manufacturers must establish written procedures for process validation.",
                "file": "Process-Validation--General-Principles-and-Practices.pdf",
                "page": 10,
                "chunk_id": "p10_c1",
                "_score": 0.41,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "What does GMP require for process validation?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert fallback_used is False
    assert len(facts) == 1
    apr_terms = {
        "annual product review",
        "product quality review",
        "pqr",
        "annual review",
        "quality review",
    }
    assert all(t not in trace["anchor_terms_used"] for t in apr_terms)
    assert all(t not in trace["expanded_query_used"] for t in apr_terms)
    assert all(not any(t in (a or []) for t in apr_terms) for a in calls)
