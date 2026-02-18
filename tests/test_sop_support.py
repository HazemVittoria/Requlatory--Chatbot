from __future__ import annotations

from src.ingestion import authority_from_folder
from src.search import _query_prior


def test_authority_mapping_supports_sop_and_sops():
    assert authority_from_folder("sop") == "SOP"
    assert authority_from_folder("sops") == "SOP"


def test_internal_sop_query_boosts_sop_authority():
    q = "use our internal sop for equipment qualification"
    sop_score = _query_prior(q, authority="SOP", file_name="Manual-053-Laboratory-Equipment-Qualification.pdf", chunk_text="internal procedure and records")
    fda_score = _query_prior(q, authority="FDA", file_name="Q9.pdf", chunk_text="regulatory guideline")
    assert sop_score > fda_score
