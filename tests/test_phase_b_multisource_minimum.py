from __future__ import annotations

from src.phase_llm import _enforce_phase_b_requirement_filter
from src.qa_types import Fact


def test_generic_gmp_validation_adds_second_document_when_available():
    selected = [
        Fact(
            quote="Manufacturers must establish written procedures for process validation.",
            pdf="fda/Process-Validation--General-Principles-and-Practices.pdf",
            page=10,
            chunk_id="p10_c1",
            score=0.74,
        )
    ]
    phase_a_facts = [
        selected[0],
        Fact(
            quote=(
                "Qualification and validation protocols should define acceptance criteria "
                "and review plans before implementation."
            ),
            pdf="eu_gmp/2015-10_annex15.pdf",
            page=6,
            chunk_id="p6_c2",
            score=0.63,
        ),
    ]

    out = _enforce_phase_b_requirement_filter(
        "What does GMP require for validation?",
        selected,
        source_facts=phase_a_facts,
    )

    assert len(out) >= 2
    assert len({f.pdf for f in out}) >= 2
    assert {f.chunk_id for f in out} == {"p10_c1", "p6_c2"}
