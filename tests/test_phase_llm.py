from __future__ import annotations

from src.phase_llm import LLMPhaseRunner, _validate_relevant_facts, _validate_synthesis
from src.qa_types import Fact


def test_validate_relevant_facts_keeps_only_source_fact_keys():
    src = [
        Fact(quote="Quality risk management should be applied across the lifecycle.", pdf="Q9.pdf", page=7, chunk_id="p7_c3", score=0.7),
        Fact(quote="Risk control.", pdf="Q9.pdf", page=11, chunk_id="p11_c2", score=0.6),
    ]
    obj = {
        "relevant_facts": [
            {"quote": "Quality risk management should be applied across the lifecycle.", "pdf": "Q9.pdf", "page": 7, "chunk_id": "p7_c3"},
            {"quote": "fake", "pdf": "X.pdf", "page": 1, "chunk_id": "p1_c1"},
        ]
    }
    out = _validate_relevant_facts(obj=obj, source_facts=src, question="what is risk management")
    assert len(out) == 1
    assert out[0].pdf == "Q9.pdf"
    assert out[0].chunk_id == "p7_c3"


def test_validate_synthesis_requires_known_citations_and_formats_output():
    rel = [
        Fact(
            quote="Quality risk management is a systematic process for assessment, control, communication, and review of quality risks.",
            pdf="Q9.pdf",
            page=7,
            chunk_id="p7_c3",
            score=0.8,
        )
    ]
    obj = {
        "answer_sentences": [
            {
                "sentence": "Quality risk management supports assessing, controlling, communicating, and reviewing product quality risks.",
                "pdf": "Q9.pdf",
                "page": 7,
                "chunk_id": "p7_c3",
            },
            {"sentence": "Bad row", "pdf": "X.pdf", "page": 1, "chunk_id": "c1"},
        ],
        "confidence": "High",
    }
    text, cits = _validate_synthesis(obj=obj, relevant_facts=rel)
    assert text.startswith("ANSWER:")
    assert "CONFIDENCE: High" in text
    assert len(cits) == 1
    assert cits[0].doc_id == "Q9.pdf"


def test_phase_runner_retries_with_fallback_temperature_zero():
    calls: list[float] = []

    def _fake_request(**kwargs):
        calls.append(float(kwargs.get("temperature", -1.0)))
        if len(calls) == 1:
            return "not json"
        return '{"relevant_facts":[{"quote":"Risk management should be applied.","pdf":"Q9.pdf","page":7,"chunk_id":"p7_c3"}]}'

    runner = LLMPhaseRunner(request_fn=_fake_request)
    facts = [Fact(quote="Risk management should be applied.", pdf="Q9.pdf", page=7, chunk_id="p7_c3", score=0.8)]
    out = runner.phase_b_filter("what is risk management", facts)
    assert len(out) == 1
    assert calls[0] == 0.2
    assert calls[1] == 0.0
