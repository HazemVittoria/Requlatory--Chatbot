from __future__ import annotations

import numpy as np

from src import search as search_module


class _FakeVectorizer:
    def transform(self, rows):
        del rows
        return np.zeros((1, 1), dtype=float)


def _configure_fake_index(monkeypatch, corpus: list[dict], sims_word: list[float], sims_char: list[float]) -> None:
    monkeypatch.setattr(search_module, "_ensure_index", lambda **kwargs: None)
    monkeypatch.setattr(search_module, "_CORPUS", list(corpus))
    monkeypatch.setattr(search_module, "_VECT_WORD", _FakeVectorizer())
    monkeypatch.setattr(search_module, "_VECT_CHAR", _FakeVectorizer())
    monkeypatch.setattr(search_module, "_X_WORD", np.zeros((len(corpus), 1), dtype=float))
    monkeypatch.setattr(search_module, "_X_CHAR", np.zeros((len(corpus), 1), dtype=float))
    monkeypatch.setattr(search_module, "_query_prior", lambda **kwargs: 0.0)
    monkeypatch.setattr(search_module, "_tokenize", lambda text: set())

    word = np.asarray([sims_word], dtype=float)
    char = np.asarray([sims_char], dtype=float)

    state = {"calls": 0}

    def _fake_cosine_similarity(a, b):
        del a, b
        out = word if (state["calls"] % 2 == 0) else char
        state["calls"] += 1
        return out

    monkeypatch.setattr(search_module, "cosine_similarity", _fake_cosine_similarity)


def test_broad_requirement_boost_promotes_requirement_chunks_across_authorities(monkeypatch):
    corpus = [
        {
            "authority": "OTHER",
            "file": "other.pdf",
            "page": 1,
            "chunk_id": "p1_c1",
            "text": "General GMP validation overview and lifecycle context.",
        },
        {
            "authority": "FDA",
            "file": "fda.pdf",
            "page": 2,
            "chunk_id": "p2_c1",
            "text": "Manufacturers must ensure validation is in accordance with written requirements.",
        },
        {
            "authority": "EU_GMP",
            "file": "eu.pdf",
            "page": 3,
            "chunk_id": "p3_c1",
            "text": "Qualification and validation shall be performed and should be documented as required.",
        },
    ]
    _configure_fake_index(monkeypatch, corpus, sims_word=[0.500, 0.493, 0.492], sims_char=[0.500, 0.493, 0.492])

    out = search_module.search_chunks(
        "What does GMP require for validation?",
        scope="MIXED",
        top_k=2,
        max_context_chunks=2,
        min_similarity=0.20,
        use_mmr=False,
    )

    assert len(out) == 2
    assert [c["authority"] for c in out] == ["FDA", "EU_GMP"]
    assert all(float(c["_req_boost"]) > 0.0 for c in out)
    assert all(int(c["_req_signal_hits"]) > 0 for c in out)
    assert all(float(c["_final_score"]) >= float(c["_base_score"]) for c in out)


def test_procedural_query_keeps_boost_inactive_and_preserves_base_order(monkeypatch):
    corpus = [
        {
            "authority": "OTHER",
            "file": "other.pdf",
            "page": 1,
            "chunk_id": "p1_c1",
            "text": "General GMP validation overview and lifecycle context.",
        },
        {
            "authority": "FDA",
            "file": "fda.pdf",
            "page": 2,
            "chunk_id": "p2_c1",
            "text": "Manufacturers must ensure validation is in accordance with written requirements.",
        },
        {
            "authority": "EU_GMP",
            "file": "eu.pdf",
            "page": 3,
            "chunk_id": "p3_c1",
            "text": "Qualification and validation shall be performed and should be documented as required.",
        },
    ]
    _configure_fake_index(monkeypatch, corpus, sims_word=[0.500, 0.493, 0.492], sims_char=[0.500, 0.493, 0.492])

    out = search_module.search_chunks(
        "How to investigate OOS?",
        scope="MIXED",
        top_k=2,
        max_context_chunks=2,
        min_similarity=0.20,
        use_mmr=False,
    )

    assert len(out) == 2
    assert [c["authority"] for c in out] == ["OTHER", "FDA"]
    assert all(int(c["_req_signal_hits"]) == 0 for c in out)
    assert all(float(c["_req_boost"]) == 0.0 for c in out)
    assert all(float(c["_final_score"]) == float(c["_base_score"]) for c in out)


def test_threshold_invariance_blocks_below_threshold_even_with_boost(monkeypatch):
    corpus = [
        {
            "authority": "FDA",
            "file": "fda.pdf",
            "page": 1,
            "chunk_id": "p1_c1",
            "text": "Validation guidance summary text.",
        },
        {
            "authority": "EU_GMP",
            "file": "eu.pdf",
            "page": 2,
            "chunk_id": "p2_c1",
            "text": "Manufacturers must ensure validation is in accordance with required controls.",
        },
    ]
    # base scores become 0.252 and 0.189; second would cross 0.20 only if boost were allowed into eligibility.
    _configure_fake_index(monkeypatch, corpus, sims_word=[0.28, 0.21], sims_char=[0.28, 0.21])

    out = search_module.search_chunks(
        "What does GMP require for validation?",
        scope="MIXED",
        top_k=2,
        max_context_chunks=2,
        min_similarity=0.20,
        use_mmr=False,
    )

    assert len(out) == 1
    assert out[0]["chunk_id"] == "p1_c1"


def test_boost_guardrail_prevents_large_gap_leapfrog(monkeypatch):
    corpus = [
        {
            "authority": "OTHER",
            "file": "other.pdf",
            "page": 1,
            "chunk_id": "p1_c1",
            "text": "General GMP validation lifecycle expectations.",
        },
        {
            "authority": "FDA",
            "file": "fda.pdf",
            "page": 2,
            "chunk_id": "p2_c1",
            "text": "Manufacturers must ensure validation is in accordance with required controls.",
        },
    ]
    # base scores become 0.54 and 0.504 (gap 0.036 > 0.02), so rank must keep base order.
    _configure_fake_index(monkeypatch, corpus, sims_word=[0.60, 0.56], sims_char=[0.60, 0.56])

    out = search_module.search_chunks(
        "What does GMP require for validation?",
        scope="MIXED",
        top_k=2,
        max_context_chunks=2,
        min_similarity=0.20,
        use_mmr=False,
    )

    assert len(out) == 2
    assert [c["authority"] for c in out] == ["OTHER", "FDA"]


def test_broad_requirement_ranking_is_deterministic(monkeypatch):
    corpus = [
        {
            "authority": "OTHER",
            "file": "other.pdf",
            "page": 1,
            "chunk_id": "p1_c1",
            "text": "General GMP validation overview and lifecycle context.",
        },
        {
            "authority": "FDA",
            "file": "fda.pdf",
            "page": 2,
            "chunk_id": "p2_c1",
            "text": "Manufacturers must ensure validation is in accordance with written requirements.",
        },
        {
            "authority": "EU_GMP",
            "file": "eu.pdf",
            "page": 3,
            "chunk_id": "p3_c1",
            "text": "Qualification and validation shall be performed and should be documented as required.",
        },
    ]
    _configure_fake_index(monkeypatch, corpus, sims_word=[0.500, 0.493, 0.492], sims_char=[0.500, 0.493, 0.492])

    out1 = search_module.search_chunks(
        "What does GMP require for validation?",
        scope="MIXED",
        top_k=3,
        max_context_chunks=3,
        min_similarity=0.20,
        use_mmr=False,
    )
    out2 = search_module.search_chunks(
        "What does GMP require for validation?",
        scope="MIXED",
        top_k=3,
        max_context_chunks=3,
        min_similarity=0.20,
        use_mmr=False,
    )

    sig1 = [(c["chunk_id"], c["authority"], c["_base_score"], c["_final_score"]) for c in out1]
    sig2 = [(c["chunk_id"], c["authority"], c["_base_score"], c["_final_score"]) for c in out2]
    assert sig1 == sig2
