from __future__ import annotations

from src import qa as qa_module
from src.qa import _phase_c_synthesis, answer, format_facts, format_relevant_facts
from src.qa_types import Fact


def test_fact_contract_formatters():
    facts = [
        Fact(quote="A systematic process for quality risk management.", pdf="Q9.pdf", page=9, chunk_id="p9_c1", score=0.71)
    ]
    ftxt = format_facts(facts)
    rtxt = format_relevant_facts(facts)
    assert ftxt.startswith("FACTS:")
    assert '(pdf="Q9.pdf", page=9, chunk_id="p9_c1", score=0.7100)' in ftxt
    assert rtxt.startswith("RELEVANT FACTS:")
    assert '(pdf="Q9.pdf", page=9, chunk_id="p9_c1")' in rtxt


def test_phase_c_requires_cited_sentences():
    facts = [
        Fact(
            quote="Quality risk management is a systematic process for the assessment, control, communication, and review of risks to quality.",
            pdf="Q9.pdf",
            page=9,
            chunk_id="p9_c2",
            score=0.8,
        ),
        Fact(
            quote="Risk management is used throughout the product lifecycle to support science- and risk-based decisions.",
            pdf="Q9.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.7,
        ),
    ]
    text, cits = _phase_c_synthesis("what is risk management", facts)
    assert text.startswith("ANSWER:")
    assert "1) " in text
    assert "(Q9.pdf, p9, p9_c2)" in text or "(Q9.pdf, p10, p10_c1)" in text
    assert "CONFIDENCE:" in text
    assert len(cits) >= 1


def test_not_found_message_shape(monkeypatch):
    monkeypatch.setattr(
        qa_module,
        "_phase_a_retrieval",
        lambda *args, **kwargs: ([], False, {"queries_used": [], "thresholds_attempted": [], "attempts": [], "fallback_used": False}),
    )
    res = answer("any", scope="FDA")
    assert res.text == "Not found in provided PDFs"


def test_live_external_metric_query_refuses(monkeypatch):
    facts = [
        Fact(
            quote="For the purposes of this guidance, data integrity refers to the completeness, consistency, and accuracy of data.",
            pdf="Data Integrity and Compliance_cGMP.pdf",
            page=8,
            chunk_id="p8_c1",
            score=0.35,
        )
    ]
    monkeypatch.setattr(
        qa_module,
        "_phase_a_retrieval",
        lambda *args, **kwargs: (facts, False, {"queries_used": [], "thresholds_attempted": [], "attempts": [], "fallback_used": False}),
    )
    res = answer("What is the current FDA inspection failure rate for data integrity in 2025?", scope="MIXED")
    assert res.text == "Not found in provided PDFs"
