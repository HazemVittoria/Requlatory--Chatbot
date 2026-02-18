from __future__ import annotations

import json
from pathlib import Path

from src.doc_router import match_topics, route_docs
from src.eval_runner import run_eval
from src.search import search_chunks
from src.topic_taxonomy import TOPICS


def _norm_paths(paths: list[str]) -> list[str]:
    return [str(p).replace("\\", "/").lower() for p in paths]


def _path_match_any(docs: list[str], paths: list[str]) -> bool:
    norm = _norm_paths(docs)
    pnorm = [str(p).replace("\\", "/").lower() for p in paths]
    for d in norm:
        for p in pnorm:
            if d.startswith(p) or p in d:
                return True
    return False


def test_route_docs_continuous_manufacturing_matches_process_validation_topic_and_ich_path():
    query = "what does ich q13 say about continuous manufacturing?"
    topics = match_topics([query])
    assert "process_validation" in topics
    docs = route_docs([query], top_docs=5)
    assert _path_match_any(docs, TOPICS["process_validation"]["paths"])
    norm = _norm_paths(docs)
    assert any(p.startswith("ich/") for p in norm)


def test_route_docs_audit_trail_matches_computerized_systems_topic_paths():
    query = "audit trail expectations for electronic records"
    topics = match_topics([query])
    assert "computerized_systems" in topics
    docs = route_docs([query], top_docs=5)
    assert _path_match_any(docs, TOPICS["computerized_systems"]["paths"])


def test_route_docs_oos_retest_matches_oos_topic_and_fda_path():
    query = "how to handle oos retest investigation?"
    topics = match_topics([query])
    assert "oos" in topics
    docs = route_docs([query], top_docs=5)
    assert _path_match_any(docs, TOPICS["oos"]["paths"])
    norm = _norm_paths(docs)
    assert any(p.startswith("fda/") for p in norm)


def test_route_docs_generic_gmp_validation_includes_annex15():
    query = "gmp require for validation"
    topics = match_topics([query])
    assert "process_validation" in topics
    docs = route_docs([query], top_docs=5)
    norm = _norm_paths(docs)
    assert any(p.startswith("eu_gmp/") and "annex15" in p.replace("-", "").replace("_", "") for p in norm)


def test_search_chunks_allowlist_restricts_documents():
    allowed = ["ich/ICH_Q13_Step4_Guideline_2022_1116.pdf"]
    hits = search_chunks(
        "continuous manufacturing ich q13",
        scope="MIXED",
        top_k=8,
        max_context_chunks=6,
        min_similarity=0.0,
        allowed_docs=allowed,
    )
    assert hits
    assert all(str(h.get("file") or "").lower() == "ich_q13_step4_guideline_2022_1116.pdf" for h in hits)


def test_routing_does_not_introduce_refusal_for_q13_eval_case(tmp_path: Path):
    golden = tmp_path / "golden_router.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": 9001,
                "question": "What is continuous manufacturing according to ICH Q13?",
                "expected_not_found": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = run_eval(golden)
    assert out["count"] == 1
    assert out["results"][0]["outcome"] == "answer_ok"
