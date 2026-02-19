from __future__ import annotations

from pathlib import Path

from src import qa as qa_module


def test_phase_a_balancing_keeps_multiple_authorities_when_available(monkeypatch):
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
        del allowed_docs, anchor_terms, chunk_chars, min_chunk_chars
        return [
            {"text": "FDA requirement 1", "file": "fda_1.pdf", "page": 1, "chunk_id": "p1_c1", "authority": "FDA", "_score": 0.50},
            {"text": "FDA requirement 2", "file": "fda_2.pdf", "page": 2, "chunk_id": "p2_c1", "authority": "FDA", "_score": 0.49},
            {"text": "FDA requirement 3", "file": "fda_3.pdf", "page": 3, "chunk_id": "p3_c1", "authority": "FDA", "_score": 0.48},
            {"text": "FDA requirement 4", "file": "fda_4.pdf", "page": 4, "chunk_id": "p4_c1", "authority": "FDA", "_score": 0.47},
            {"text": "FDA requirement 5", "file": "fda_5.pdf", "page": 5, "chunk_id": "p5_c1", "authority": "FDA", "_score": 0.46},
            {"text": "FDA requirement 6", "file": "fda_6.pdf", "page": 6, "chunk_id": "p6_c1", "authority": "FDA", "_score": 0.45},
            {"text": "EU GMP validation requirement", "file": "eu_annex15.pdf", "page": 10, "chunk_id": "p10_c1", "authority": "EU_GMP", "_score": 0.44},
            {"text": "ICH validation requirement", "file": "ich_q7.pdf", "page": 20, "chunk_id": "p20_c1", "authority": "ICH", "_score": 0.43},
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "What does GMP require for validation?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert fallback_used is False
    assert len(facts) == qa_module.MAX_CONTEXT_CHUNKS
    assert "eu_annex15.pdf" in {f.pdf for f in facts}
    assert "ich_q7.pdf" in {f.pdf for f in facts}
    assert set(trace["balancing"]["authorities_kept"]) >= {"EU_GMP", "FDA", "ICH"}


def test_phase_a_balancing_preserves_single_authority_top_results(monkeypatch):
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
        del allowed_docs, anchor_terms, chunk_chars, min_chunk_chars
        return [
            {"text": f"FDA requirement {i}", "file": "fda_only.pdf", "page": i, "chunk_id": f"p{i}_c1", "authority": "FDA", "_score": 0.60 - (i * 0.01)}
            for i in range(1, 9)
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    facts, fallback_used, trace = qa_module._phase_a_retrieval(
        "What does FDA require for process validation?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    assert fallback_used is False
    assert len(facts) == qa_module.MAX_CONTEXT_CHUNKS
    assert [f.chunk_id for f in facts] == [f"p{i}_c1" for i in range(1, qa_module.MAX_CONTEXT_CHUNKS + 1)]
    assert trace["balancing"]["authorities_kept"] == ["FDA"]


def test_phase_a_trace_exposes_requirement_boost_fields(monkeypatch):
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
        del allowed_docs, anchor_terms, chunk_chars, min_chunk_chars
        return [
            {
                "text": "Manufacturers must ensure validation is in accordance with requirements.",
                "file": "fda.pdf",
                "page": 1,
                "chunk_id": "p1_c1",
                "authority": "FDA",
                "_score": 0.46,
                "_base_score": 0.45,
                "_req_signal_hits": 3,
                "_req_boost": 0.015,
                "_final_score": 0.465,
            }
        ]

    monkeypatch.setattr(qa_module, "search_chunks", _fake_search_chunks)

    _, _, trace = qa_module._phase_a_retrieval(
        "What does GMP require for validation?",
        scope="MIXED",
        data_dir=Path("data"),
        chunk_chars=1000,
        min_chunk_chars=220,
    )

    top = trace["attempts"][0]["top_results"][0]
    assert "authority" in top
    assert "base_score" in top
    assert "req_signal_hits" in top
    assert "req_boost" in top
    assert "final_score" in top


def test_phase_a_refill_prefers_requirement_density_within_base_gap_guardrail():
    chunks = [
        {"file": "fda_1.pdf", "page": 1, "chunk_id": "p1_c1", "authority": "FDA", "_score": 0.90, "_final_score": 0.90, "_base_score": 0.90, "_req_signal_hits": 0},
        {"file": "eu_1.pdf", "page": 1, "chunk_id": "p1_c1", "authority": "EU_GMP", "_score": 0.88, "_final_score": 0.88, "_base_score": 0.88, "_req_signal_hits": 0},
        {"file": "fda_2.pdf", "page": 2, "chunk_id": "p2_c1", "authority": "FDA", "_score": 0.87, "_final_score": 0.87, "_base_score": 0.87, "_req_signal_hits": 0},
        {"file": "fda_3.pdf", "page": 3, "chunk_id": "p3_c1", "authority": "FDA", "_score": 0.86, "_final_score": 0.86, "_base_score": 0.86, "_req_signal_hits": 0},
        {"file": "fda_4_low_hits.pdf", "page": 4, "chunk_id": "p4_c1", "authority": "FDA", "_score": 0.400, "_final_score": 0.400, "_base_score": 0.400, "_req_signal_hits": 0},
        {"file": "fda_5_high_hits.pdf", "page": 5, "chunk_id": "p5_c1", "authority": "FDA", "_score": 0.399, "_final_score": 0.399, "_base_score": 0.389, "_req_signal_hits": 3},
    ]

    out, trace = qa_module._balance_phase_a_chunks(chunks, target_k=5)

    assert len(out) == 5
    assert trace["output_count"] == 5
    assert "p5_c1" in {c["chunk_id"] for c in out}
    assert "p4_c1" not in {c["chunk_id"] for c in out}
