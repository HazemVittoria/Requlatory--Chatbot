from __future__ import annotations

from pathlib import Path

from src import qa as qa_module


def test_deviation_trigger_and_anchor_terms(monkeypatch):
    monkeypatch.setattr(qa_module, "route_docs", lambda *args, **kwargs: [])
    queries_seen: list[str] = []
    anchors_seen: list[list[str] | None] = []

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
        queries_seen.append(query)
        anchors_seen.append(anchor_terms)
        return [
            {
                "text": "Deviation investigation should identify root cause and CAPA.",
                "file": "qms.pdf",
                "page": 4,
                "chunk_id": "p4_c1",
                "_score": 0.33,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    _, _, trace = qa_module._phase_a_retrieval(
        "What is required for deviation investigation and root cause?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert "deviation" in trace["anchor_terms_used"]
    assert "root cause" in trace["anchor_terms_used"]
    assert "capa" in trace["anchor_terms_used"]
    assert "deviation report" in trace["anchor_terms_used"]
    assert "deviation investigation" in trace["expanded_query_used"]
    assert "impact assessment" in trace["expanded_query_used"]
    assert any("deviation investigation" in q and "root cause" in q for q in queries_seen)
    assert all(a and "deviation" in a for a in anchors_seen)


def test_equipment_trigger_and_anchor_terms(monkeypatch):
    monkeypatch.setattr(qa_module, "route_docs", lambda *args, **kwargs: [])
    queries_seen: list[str] = []
    anchors_seen: list[list[str] | None] = []

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
        queries_seen.append(query)
        anchors_seen.append(anchor_terms)
        return [
            {
                "text": "IQ/OQ/PQ protocols should define acceptance criteria.",
                "file": "annex15.pdf",
                "page": 6,
                "chunk_id": "p6_c1",
                "_score": 0.31,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    _, _, trace = qa_module._phase_a_retrieval(
        "What are IQ, OQ, and PQ and what is required in each stage?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert "equipment qualification" in trace["anchor_terms_used"]
    assert "installation qualification" in trace["anchor_terms_used"]
    assert "iq" in trace["anchor_terms_used"]
    assert "oq" in trace["anchor_terms_used"]
    assert "pq" in trace["anchor_terms_used"]
    assert "qualification protocol" in trace["expanded_query_used"]
    assert any("equipment qualification" in q and "acceptance criteria" in q for q in queries_seen)
    assert all(a and "equipment qualification" in a for a in anchors_seen)


def test_computerized_systems_query_includes_data_integrity_trigger_anchors(monkeypatch):
    monkeypatch.setattr(qa_module, "route_docs", lambda *args, **kwargs: [])

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
        del allowed_docs, chunk_chars, min_chunk_chars, anchor_terms
        return [
            {
                "text": "Computerized systems should maintain audit trails.",
                "file": "annex11.pdf",
                "page": 3,
                "chunk_id": "p3_c1",
                "_score": 0.35,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    _, _, trace = qa_module._phase_a_retrieval(
        "What controls are expected for audit trails in computerized systems?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert "data integrity" in trace["anchor_terms_used"]
    assert "audit trail" in trace["anchor_terms_used"]
    assert "metadata" in trace["anchor_terms_used"]
    assert "audit trail" in trace["expanded_query_used"]
