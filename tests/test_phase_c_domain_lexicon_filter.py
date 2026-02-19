from __future__ import annotations

from src import qa as qa_module
from src.qa_types import Fact


def test_domain_filter_applies_for_computerized_systems_when_enough_domain_sentences():
    facts = [
        Fact(
            quote=(
                "The computerized system must maintain an audit trail and access control. "
                "CAPA should be documented by the quality unit."
            ),
            pdf="annex11.pdf",
            page=5,
            chunk_id="p5_c1",
            score=0.61,
        ),
        Fact(
            quote="Electronic records should implement authentication and authorization for user privileges.",
            pdf="part11.pdf",
            page=12,
            chunk_id="p12_c1",
            score=0.59,
        ),
    ]

    selected = qa_module._select_phase_c_facts(facts)
    candidates, stats = qa_module._phase_c_sentence_candidates(
        question="What controls are expected for audit trails in computerized systems?",
        selected_facts=selected,
    )

    assert stats["domain_key"] == "computerized_systems"
    assert stats["N_requirement_bearing"] >= 2
    assert stats["N_domain_eligible"] <= stats["N_requirement_bearing"]
    assert stats["domain_filter_used"] is True
    texts = [c[0].lower() for c in candidates]
    assert all("capa" not in t for t in texts)
    assert any("audit trail" in t for t in texts)


def test_domain_filter_fails_open_when_domain_eligible_too_small():
    facts = [
        Fact(
            quote="Data integrity controls must ensure attributable and accurate records.",
            pdf="di.pdf",
            page=3,
            chunk_id="p3_c1",
            score=0.63,
        ),
        Fact(
            quote="Investigations must be completed and documented in writing.",
            pdf="qms.pdf",
            page=7,
            chunk_id="p7_c1",
            score=0.58,
        ),
    ]

    selected = qa_module._select_phase_c_facts(facts)
    candidates, stats = qa_module._phase_c_sentence_candidates(
        question="What constitutes data integrity and what controls ensure ALCOA?",
        selected_facts=selected,
    )

    assert stats["domain_key"] == "data_integrity"
    assert stats["N_requirement_bearing"] >= 1
    assert stats["N_domain_eligible"] < qa_module.MIN_DOMAIN_SENTENCES
    assert stats["domain_filter_used"] is False
    assert len(candidates) == stats["N_requirement_bearing"]


def test_no_domain_key_skips_domain_filter_and_preserves_non_domain_behavior():
    facts = [
        Fact(
            quote=(
                "Quality risk management is a systematic process for the assessment, control, "
                "communication and review of risks to quality."
            ),
            pdf="Q9.pdf",
            page=7,
            chunk_id="p7_c3",
            score=0.70,
        ),
    ]

    selected = qa_module._select_phase_c_facts(facts)
    candidates, stats = qa_module._phase_c_sentence_candidates(
        question="What is quality risk management under ICH Q9?",
        selected_facts=selected,
    )

    assert stats["domain_key"] is None
    assert stats["domain_filter_used"] is False
    assert len(candidates) >= 1
