from __future__ import annotations

from src.phase_llm import _enforce_phase_b_requirement_filter
from src.qa_types import Fact


def test_boilerplate_fact_is_removed():
    facts = [
        Fact(
            quote="This guidance represents the Food and Drug Administration's current thinking on this topic.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=4,
            chunk_id="p4_c1",
            score=0.5,
        ),
        Fact(
            quote="Manufacturers must establish written procedures for process validation.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.45,
        ),
    ]
    out = _enforce_phase_b_requirement_filter("What does GMP require for validation?", facts, source_facts=facts)
    assert len(out) == 1
    assert out[0].chunk_id == "p10_c1"


def test_requirement_language_facts_are_kept():
    facts = [
        Fact(
            quote="The manufacturer shall maintain validation records.",
            pdf="Q7 Guideline.pdf",
            page=20,
            chunk_id="p20_c1",
            score=0.6,
        ),
        Fact(
            quote="The process must be verified at defined stages.",
            pdf="Q7 Guideline.pdf",
            page=21,
            chunk_id="p21_c1",
            score=0.55,
        ),
    ]
    out = _enforce_phase_b_requirement_filter("validation requirements", facts, source_facts=facts)
    assert len(out) == 2
    assert {f.chunk_id for f in out} == {"p20_c1", "p21_c1"}


def test_fallback_keeps_best_topic_matching_non_boilerplate():
    selected = [
        Fact(
            quote="Introduction to this guidance document.",
            pdf="X.pdf",
            page=1,
            chunk_id="p1_c1",
            score=0.9,
        )
    ]
    source = [
        Fact(
            quote="This section discusses validation lifecycle activities.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c2",
            score=0.4,
        ),
        Fact(
            quote="Validation planning across stages is described.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=11,
            chunk_id="p11_c1",
            score=0.7,
        ),
    ]
    out = _enforce_phase_b_requirement_filter(
        "What is validation lifecycle planning?",
        selected,
        source_facts=source,
    )
    assert len(out) == 1
    assert out[0].chunk_id == "p11_c1"


def test_validation_gate_accepts_primary_or_two_secondary_tokens():
    facts = [
        Fact(
            quote="The manufacturer must establish process validation protocols and maintain records.",
            pdf="fda.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.7,
        ),
        Fact(
            quote="Firms shall define qualification plans with acceptance criteria before execution.",
            pdf="eu.pdf",
            page=6,
            chunk_id="p6_c1",
            score=0.65,
        ),
        Fact(
            quote="The process must include qualification before release.",
            pdf="eu.pdf",
            page=7,
            chunk_id="p7_c1",
            score=0.6,
        ),
    ]
    out = _enforce_phase_b_requirement_filter(
        "What does GMP require for validation?",
        facts,
        source_facts=facts,
    )
    assert {f.chunk_id for f in out} == {"p10_c1", "p6_c1"}


def test_validation_gate_rejects_negative_patterns_unless_exception():
    facts = [
        Fact(
            quote="Manufacturers must keep validation records; email subject line should identify batch.",
            pdf="x.pdf",
            page=2,
            chunk_id="p2_c1",
            score=0.72,
        ),
        Fact(
            quote="Manufacturers must perform process validation with predefined acceptance criteria.",
            pdf="x.pdf",
            page=3,
            chunk_id="p3_c1",
            score=0.70,
        ),
    ]
    out = _enforce_phase_b_requirement_filter(
        "What does GMP require for validation?",
        facts,
        source_facts=facts,
    )
    assert {f.chunk_id for f in out} == {"p3_c1"}


def test_validation_gate_negative_filter_bypassed_for_imp_or_adulteration_query():
    facts = [
        Fact(
            quote="Batches shall be deemed to be adulterated if validation controls are missing.",
            pdf="fda.pdf",
            page=11,
            chunk_id="p11_c1",
            score=0.71,
        ),
    ]
    out = _enforce_phase_b_requirement_filter(
        "Under GMP, when are products adulterated due to failed validation?",
        facts,
        source_facts=facts,
    )
    assert len(out) == 1
    assert out[0].chunk_id == "p11_c1"


def test_procedural_query_does_not_trigger_validation_gate():
    facts = [
        Fact(
            quote="Investigations must be documented and reviewed by quality unit.",
            pdf="oos.pdf",
            page=4,
            chunk_id="p4_c1",
            score=0.63,
        ),
    ]
    out = _enforce_phase_b_requirement_filter(
        "How to investigate OOS?",
        facts,
        source_facts=facts,
    )
    assert len(out) == 1
    assert out[0].chunk_id == "p4_c1"


def test_validation_gate_is_deterministic():
    facts = [
        Fact(
            quote="Manufacturers must establish process validation procedures and maintain records.",
            pdf="fda.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.70,
        ),
        Fact(
            quote="Qualification plans should include acceptance criteria before execution.",
            pdf="eu.pdf",
            page=6,
            chunk_id="p6_c1",
            score=0.68,
        ),
    ]
    out1 = _enforce_phase_b_requirement_filter(
        "What does GMP require for validation?",
        facts,
        source_facts=facts,
    )
    out2 = _enforce_phase_b_requirement_filter(
        "What does GMP require for validation?",
        facts,
        source_facts=facts,
    )
    assert [(f.pdf, f.page, f.chunk_id, f.score) for f in out1] == [
        (f.pdf, f.page, f.chunk_id, f.score) for f in out2
    ]
