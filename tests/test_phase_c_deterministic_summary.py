from __future__ import annotations

import re

from src.qa import _deterministic_confidence_label, _phase_c_synthesis
from src.qa_types import Fact


def test_phase_c_deterministic_generates_three_plus_sentences_when_enough_facts():
    facts = [
        Fact(
            quote="Manufacturers must establish written procedures for process validation.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.72,
        ),
        Fact(
            quote="Validation activities should be reviewed throughout the product lifecycle.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c2",
            score=0.68,
        ),
        Fact(
            quote="Firms shall maintain validation records and supporting evidence.",
            pdf="pe-009-17-gmp-guide-xannexes.pdf",
            page=219,
            chunk_id="p219_c1",
            score=0.66,
        ),
        Fact(
            quote="The process requires verification at defined stages before routine use.",
            pdf="Q7 Guideline.pdf",
            page=23,
            chunk_id="p23_c2",
            score=0.64,
        ),
        Fact(
            quote="Validation expectations include documented procedures and periodic review.",
            pdf="Q11 Guideline.pdf",
            page=31,
            chunk_id="p31_c1",
            score=0.61,
        ),
    ]
    text, cits = _phase_c_synthesis("What does GMP require for validation?", facts)
    assert text.startswith("ANSWER:")
    body = [ln for ln in text.splitlines() if ln.strip().startswith(tuple(f"{i})" for i in range(1, 7)))]
    assert len(body) >= 3
    assert len(cits) >= 3
    assert "INTRODUCTION" not in text
    assert "Guidance for Industry" not in text
    assert "with a." not in text.lower()
    for ln in body:
        assert ln.endswith(")")
        assert re.search(r"[.;:]\s+\([^)]+\)$", ln)


def test_phase_c_deterministic_returns_not_found_when_fewer_than_two_usable_facts():
    facts = [
        Fact(
            quote="INTRODUCTION:",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=1,
            chunk_id="p1_c1",
            score=0.7,
        )
    ]
    text, cits = _phase_c_synthesis("What does GMP require for validation?", facts)
    assert text == "Not found in provided PDFs"
    assert cits == []


def test_deterministic_confidence_label_thresholds():
    assert _deterministic_confidence_label(
        num_sentences=3,
        distinct_docs=2,
        avg_similarity=0.25,
    ) == "High"
    assert _deterministic_confidence_label(
        num_sentences=2,
        distinct_docs=1,
        avg_similarity=0.22,
    ) == "Medium"
    assert _deterministic_confidence_label(
        num_sentences=1,
        distinct_docs=1,
        avg_similarity=0.40,
    ) == "Low"
