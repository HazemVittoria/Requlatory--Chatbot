from __future__ import annotations

from src.phase_llm import _validate_synthesis
from src.qa_types import Fact


def test_heading_like_sentence_is_removed():
    rel = [
        Fact(
            quote="Manufacturers must establish validation procedures and maintain records for each stage.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.8,
        )
    ]
    obj = {
        "answer_sentences": [
            {"sentence": "INTRODUCTION:", "pdf": rel[0].pdf, "page": rel[0].page, "chunk_id": rel[0].chunk_id},
            {
                "sentence": "Manufacturers should establish and maintain validation procedures, with records reviewed across each lifecycle stage.",
                "pdf": rel[0].pdf,
                "page": rel[0].page,
                "chunk_id": rel[0].chunk_id,
            },
        ],
        "confidence": "Medium",
    }
    text, cits = _validate_synthesis(obj=obj, relevant_facts=rel, question="What does GMP require for validation procedures?")
    assert text.startswith("ANSWER:")
    assert "INTRODUCTION:" not in text
    assert len(cits) == 1


def test_sentence_word_limit_enforced():
    rel = [
        Fact(
            quote="Validation activities are defined in procedures and reviewed across the lifecycle.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c2",
            score=0.8,
        ),
        Fact(
            quote="Firms should define validation procedures and review records periodically.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=11,
            chunk_id="p11_c1",
            score=0.7,
        ),
    ]
    long_sentence = (
        "Manufacturers should establish and maintain comprehensive validation procedures across multiple "
        "product lifecycle phases to ensure continued process performance, data quality, and ongoing "
        "review in all operational contexts."
    )
    obj = {
        "answer_sentences": [
            {"sentence": long_sentence, "pdf": rel[0].pdf, "page": rel[0].page, "chunk_id": rel[0].chunk_id},
            {"sentence": "Firms should define validation procedures and review records periodically.", "pdf": rel[1].pdf, "page": rel[1].page, "chunk_id": rel[1].chunk_id},
        ],
        "confidence": "Low",
    }
    text, cits = _validate_synthesis(obj=obj, relevant_facts=rel, question="What are validation procedures and review requirements?")
    assert text.startswith("ANSWER:")
    assert len(cits) == 1
    body = [ln for ln in text.splitlines() if ln.strip().startswith("1) ")]
    assert body
    # Ensure kept sentence is short.
    words = body[0].split()
    assert len(words) <= 40  # line includes numbering and citation.


def test_quote_dump_sentence_is_removed():
    rel = [
        Fact(
            quote="The manufacturer must establish and follow written procedures for process validation.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c3",
            score=0.9,
        ),
        Fact(
            quote="Validation procedures should be reviewed periodically and updated when needed.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=11,
            chunk_id="p11_c2",
            score=0.7,
        ),
    ]
    obj = {
        "answer_sentences": [
            {
                "sentence": "The manufacturer must establish and follow written procedures for process validation.",
                "pdf": rel[0].pdf,
                "page": rel[0].page,
                "chunk_id": rel[0].chunk_id,
            },
            {
                "sentence": "Firms should review validation procedures periodically and update them when needed.",
                "pdf": rel[1].pdf,
                "page": rel[1].page,
                "chunk_id": rel[1].chunk_id,
            },
        ],
        "confidence": "High",
    }
    text, cits = _validate_synthesis(obj=obj, relevant_facts=rel, question="What validation procedures should be reviewed?")
    assert text.startswith("ANSWER:")
    assert len(cits) == 1
    assert "follow written procedures for process validation" not in text


def test_all_sentences_removed_returns_not_found():
    rel = [
        Fact(
            quote="Guidance for Industry Process Validation: General Principles and Practices.",
            pdf="Process-Validation--General-Principles-and-Practices.pdf",
            page=1,
            chunk_id="p1_c1",
            score=0.5,
        )
    ]
    obj = {
        "answer_sentences": [
            {"sentence": "INTRODUCTION:", "pdf": rel[0].pdf, "page": rel[0].page, "chunk_id": rel[0].chunk_id},
        ],
        "confidence": "Low",
    }
    text, cits = _validate_synthesis(obj=obj, relevant_facts=rel, question="What does GMP require for validation?")
    assert text == "Not found in provided PDFs"
    assert cits == []
