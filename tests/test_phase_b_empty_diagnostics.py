from __future__ import annotations

from src import qa as qa_module
from src.qa import answer
from src.qa_types import Fact


def test_phase_b_empty_emits_reason_summary(monkeypatch):
    facts = [
        Fact(
            quote="This section provides background context and overview language only.",
            pdf="fda/Process-Validation--General-Principles-and-Practices.pdf",
            page=4,
            chunk_id="p4_c1",
            score=0.41,
        ),
        Fact(
            quote="General introductory statements are described for awareness.",
            pdf="eu_gmp/2015-10_annex15.pdf",
            page=2,
            chunk_id="p2_c1",
            score=0.36,
        ),
    ]
    trace = {
        "queries_used": ["what does gmp require for validation?"],
        "thresholds_attempted": [0.2],
        "attempts": [
            {
                "min_similarity": 0.2,
                "top_results": [
                    {
                        "pdf": "Process-Validation--General-Principles-and-Practices.pdf",
                        "authority": "FDA",
                        "page": 4,
                        "chunk_id": "p4_c1",
                        "score": 0.41,
                        "base_score": 0.41,
                        "req_signal_hits": 0,
                        "req_boost": 0.0,
                        "final_score": 0.41,
                    },
                    {
                        "pdf": "2015-10_annex15.pdf",
                        "authority": "EU_GMP",
                        "page": 2,
                        "chunk_id": "p2_c1",
                        "score": 0.36,
                        "base_score": 0.36,
                        "req_signal_hits": 0,
                        "req_boost": 0.0,
                        "final_score": 0.36,
                    },
                ],
            }
        ],
        "fallback_used": False,
        "balancing": {
            "input_count": 2,
            "output_count": 2,
            "authorities_seen": ["EU_GMP", "FDA"],
            "authorities_kept": ["EU_GMP", "FDA"],
            "soft_floor_score": 0.2,
        },
    }

    monkeypatch.setattr(qa_module, "_phase_a_retrieval", lambda *args, **kwargs: (facts, False, trace))

    res = answer("What does GMP require for validation?")

    assert res.text == "Not found in provided PDFs"
    phase_b = (res.debug_trace or {}).get("phase_b", {})
    summary = phase_b.get("empty_reason_summary", {})
    assert summary.get("reason") == "phase_b_filters_removed_all_candidates"
    assert summary.get("phase_a_candidate_count") == 2
    assert summary.get("phase_a_pool_count") == 2
    assert "requirement_bearing_count" in summary
    assert "topic_coherence_count" in summary
    assert isinstance(summary.get("top_candidates"), list)
    assert len(summary.get("top_candidates", [])) <= 5
