from __future__ import annotations

from functools import cmp_to_key
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import build_corpus

_CACHE_FILE = Path(".cache") / "retrieval_index.pkl"
_CACHE_VERSION = "v2-with-internal-sop-tagging"

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")
_BROAD_REQUIREMENT_INCLUDE_TERMS = (
    "require",
    "required",
    "requirements",
    "must",
    "shall",
    "should",
    "validation",
    "qualification",
)
_BROAD_REQUIREMENT_EXCLUDE_PHRASES = (
    "how to",
    "steps",
    "procedure",
    "investigate",
    "handle",
    "process for",
    "method for",
)
_REQUIREMENT_SIGNALS = (
    "shall",
    "must",
    "required",
    "should",
    "ensure",
    "in accordance with",
)
_REQ_SIGNAL_HITS_CAP = 3
_REQ_SIGNAL_BOOST_PER_HIT = 0.005
_REQ_SIGNAL_MAX_BOOST = 0.015
_BOOST_REORDER_GAP_MAX = 0.02

_CORPUS: list[dict[str, Any]] | None = None
_VECT_WORD: TfidfVectorizer | None = None
_VECT_CHAR: TfidfVectorizer | None = None
_X_WORD = None
_X_CHAR = None
_LOGGER = logging.getLogger(__name__)
_EMPTY_RETRIEVAL_EVENTS: list[dict[str, Any]] = []
_MAX_EMPTY_RETRIEVAL_EVENTS = 256


def _record_empty_retrieval_event(event: dict[str, Any]) -> None:
    _EMPTY_RETRIEVAL_EVENTS.append(event)
    overflow = len(_EMPTY_RETRIEVAL_EVENTS) - _MAX_EMPTY_RETRIEVAL_EVENTS
    if overflow > 0:
        del _EMPTY_RETRIEVAL_EVENTS[:overflow]
    _LOGGER.debug("phase_a_empty_retrieval=%s", event)


def pop_empty_retrieval_events() -> list[dict[str, Any]]:
    if not _EMPTY_RETRIEVAL_EVENTS:
        return []
    out = list(_EMPTY_RETRIEVAL_EVENTS)
    _EMPTY_RETRIEVAL_EVENTS.clear()
    return out


_TOPIC_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "signals": ("risk management", "risk managment", "quality risk management", "ich q9", "q9"),
        "authority_boost": {"ICH": 0.12},
        "file_boost": (("q9", 0.26), ("q10", 0.04)),
        "text_boost": (
            ("quality risk management is a systematic process for the assessment, control, communication", 0.45),
            ("quality risk management (qrm): a systematic process", 0.30),
            ("systematic process for the assessment, control, communication and review of risks to quality", 0.30),
        ),
    },
    {
        "signals": ("continuous manufacturing", "continuous processing", "ich q13", "q13"),
        "authority_boost": {"ICH": 0.08},
        "file_boost": (("q13", 0.08),),
        "text_boost": (
            ("continuous manufacturing (cm)", 0.12),
            ("development, implementation, operation, and lifecycle management of continuous manufacturing", 0.16),
        ),
    },
    {
        "signals": ("data integrity", "alcoa", "audit trail", "part 11", "annex 11"),
        "authority_boost": {"FDA": 0.07, "PIC_S": 0.05, "WHO": 0.05, "EMA": 0.03},
        "file_boost": (
            ("data integrity and compliance", 0.16),
            ("trs1033-annex4-guideline-on-data-integrity", 0.12),
            ("pi 041", 0.08),
            ("data integrity", 0.08),
            ("part-11", 0.06),
            ("annex11", 0.06),
        ),
    },
    {
        "signals": ("process validation", "validation lifecycle", "continued process verification"),
        "authority_boost": {"FDA": 0.07, "ICH": 0.06, "EU_GMP": 0.05},
        "file_boost": (("process-validation", 0.14), ("annex15", 0.12), ("q8", 0.06), ("q11", 0.06)),
    },
    {
        "signals": ("deviation", "capa", "corrective action", "preventive action"),
        "authority_boost": {"FDA": 0.06, "ICH": 0.05, "EU_GMP": 0.04},
        "file_boost": (("q10", 0.10), ("deviation", 0.07), ("capa", 0.07)),
    },
    {
        "signals": ("sop", "internal sop", "our procedure", "work instruction", "company procedure"),
        "authority_boost": {"SOP": 0.30},
        "file_boost": (("sop", 0.18), ("manual-", 0.12), ("validation and verification", 0.10)),
    },
)


def _scope_authorities(scope: str) -> set[str]:
    s = (scope or "MIXED").strip().upper()
    if s == "FDA":
        return {"FDA"}
    if s == "ICH":
        return {"ICH"}
    if s == "EMA":
        return {"EMA", "EU_GMP", "PIC_S"}
    if s == "SOPS":
        return {"SOP"}
    return {"FDA", "EMA", "EU_GMP", "ICH", "PIC_S", "WHO", "SOP", "OTHER"}


def _data_fingerprint(data_dir: Path) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.rglob("*.pdf"), key=lambda x: str(x).lower()):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p.relative_to(data_dir)).replace("\\", "/"), int(st.st_size), int(st.st_mtime_ns)))
    return out


def _cache_meta(data_dir: Path, chunk_chars: int, min_chunk_chars: int) -> dict[str, Any]:
    return {
        "version": _CACHE_VERSION,
        "data_dir": str(data_dir.resolve()),
        "fingerprint": _data_fingerprint(data_dir),
        "chunk_chars": chunk_chars,
        "min_chunk_chars": min_chunk_chars,
        "word_vect": {"stop_words": "english", "ngram_range": (1, 2), "min_df": 1},
        "char_vect": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 1},
    }


def _load_cache(meta: dict[str, Any]) -> bool:
    global _CORPUS, _VECT_WORD, _VECT_CHAR, _X_WORD, _X_CHAR
    if not _CACHE_FILE.exists():
        return False
    try:
        payload = pickle.loads(_CACHE_FILE.read_bytes())
    except Exception:
        return False
    if not isinstance(payload, dict) or payload.get("meta") != meta:
        return False
    _CORPUS = payload.get("corpus")
    _VECT_WORD = payload.get("vect_word")
    _VECT_CHAR = payload.get("vect_char")
    _X_WORD = payload.get("x_word")
    _X_CHAR = payload.get("x_char")
    return all(x is not None for x in (_CORPUS, _VECT_WORD, _VECT_CHAR, _X_WORD, _X_CHAR))


def _save_cache(meta: dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "corpus": _CORPUS,
        "vect_word": _VECT_WORD,
        "vect_char": _VECT_CHAR,
        "x_word": _X_WORD,
        "x_char": _X_CHAR,
    }
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def rebuild_index(data_dir: Path, chunk_chars: int = 1000, min_chunk_chars: int = 220, force: bool = False) -> None:
    global _CORPUS, _VECT_WORD, _VECT_CHAR, _X_WORD, _X_CHAR

    meta = _cache_meta(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars)
    if not force and _load_cache(meta):
        return

    _CORPUS = build_corpus(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars)
    texts = [(c.get("text") or "") for c in _CORPUS]
    _VECT_WORD = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    _VECT_CHAR = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    _X_WORD = _VECT_WORD.fit_transform(texts)
    _X_CHAR = _VECT_CHAR.fit_transform(texts)
    _save_cache(meta)


def _ensure_index(data_dir: Path, chunk_chars: int, min_chunk_chars: int) -> None:
    if _CORPUS is None or _VECT_WORD is None or _VECT_CHAR is None or _X_WORD is None or _X_CHAR is None:
        rebuild_index(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars, force=False)


def _tokenize(text: str) -> set[str]:
    stop = {"what", "how", "when", "where", "which", "should", "could", "would"}
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in stop}


def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").lower().split())


def _is_broad_requirement_query(query: str) -> bool:
    q = _normalize_spaces(query)
    if not q:
        return False
    if not re.search(r"\bgmp\b", q):
        return False
    if not any(re.search(rf"\b{re.escape(term)}\b", q) for term in _BROAD_REQUIREMENT_INCLUDE_TERMS):
        return False
    if any(phrase in q for phrase in _BROAD_REQUIREMENT_EXCLUDE_PHRASES):
        return False
    return True


def _requirement_signal_hits(text: str) -> int:
    s = _normalize_spaces(text)
    if not s:
        return 0
    hits = 0
    for signal in _REQUIREMENT_SIGNALS:
        hits += len(re.findall(rf"\b{re.escape(signal)}\b", s))
    return min(_REQ_SIGNAL_HITS_CAP, hits)


def _requirement_signal_boost(hits: int) -> float:
    if hits <= 0:
        return 0.0
    return min(_REQ_SIGNAL_MAX_BOOST, float(hits) * _REQ_SIGNAL_BOOST_PER_HIT)


def _candidate_tie_break_key(c: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(c.get("file") or ""),
        int(c.get("page") or 0),
        str(c.get("chunk_id") or ""),
    )


def _rank_candidates_with_guardrail(
    candidate_idx: list[int],
    *,
    base_scores: np.ndarray,
    final_scores: np.ndarray,
    corpus: list[dict[str, Any]],
) -> list[int]:
    if len(candidate_idx) <= 1:
        return list(candidate_idx)

    eps = 1e-12

    def _cmp(i: int, j: int) -> int:
        bi = float(base_scores[i])
        bj = float(base_scores[j])
        if abs(bi - bj) > _BOOST_REORDER_GAP_MAX:
            if bi > bj:
                return -1
            if bi < bj:
                return 1
        fi = float(final_scores[i])
        fj = float(final_scores[j])
        if fi > fj + eps:
            return -1
        if fi + eps < fj:
            return 1
        if bi > bj + eps:
            return -1
        if bi + eps < bj:
            return 1
        ti = _candidate_tie_break_key(corpus[i])
        tj = _candidate_tie_break_key(corpus[j])
        if ti < tj:
            return -1
        if ti > tj:
            return 1
        return 0

    return sorted(candidate_idx, key=cmp_to_key(_cmp))


def _mmr_select(
    candidate_idx: list[int],
    relevance_scores: np.ndarray,
    x_word,
    max_items: int,
    mmr_lambda: float,
) -> list[int]:
    if not candidate_idx:
        return []
    if len(candidate_idx) <= 1 or max_items <= 1:
        return candidate_idx[: max(1, max_items)]

    cand = list(candidate_idx)
    max_items = max(1, min(max_items, len(cand)))
    lam = max(0.0, min(1.0, float(mmr_lambda)))

    cand_mat = x_word[cand]
    pair_sim = cosine_similarity(cand_mat, cand_mat)

    selected_pos: list[int] = []
    remaining_pos = list(range(len(cand)))

    first_pos = max(remaining_pos, key=lambda p: float(relevance_scores[cand[p]]))
    selected_pos.append(first_pos)
    remaining_pos.remove(first_pos)

    while remaining_pos and len(selected_pos) < max_items:
        best_pos = None
        best_val = -1e9
        for p in remaining_pos:
            rel = float(relevance_scores[cand[p]])
            div = max(float(pair_sim[p, s]) for s in selected_pos)
            mmr = (lam * rel) - ((1.0 - lam) * div)
            if mmr > best_val:
                best_val = mmr
                best_pos = p
        assert best_pos is not None
        selected_pos.append(best_pos)
        remaining_pos.remove(best_pos)

    return [cand[p] for p in selected_pos]


def _matching_profiles(query_l: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in _TOPIC_PROFILES:
        if any(sig in query_l for sig in profile.get("signals", ())):
            out.append(profile)
    return out


def _query_prior(query_l: str, authority: str, file_name: str, chunk_text: str) -> float:
    score = 0.0
    a = (authority or "OTHER").strip().upper()
    fn = (file_name or "").lower()
    text_l = (chunk_text or "").lower()

    # Query-aware authority + canonical-file priors.
    for p in _matching_profiles(query_l):
        score += float(p.get("authority_boost", {}).get(a, 0.0))
        for pattern, boost in p.get("file_boost", ()):
            if pattern in fn:
                score += float(boost)
        for pattern, boost in p.get("text_boost", ()):
            if pattern in text_l:
                score += float(boost)

    # Explicit internal-SOP intent: prioritize SOP authority regardless of profile match granularity.
    if any(sig in query_l for sig in ("internal sop", "our sop", "sop", "internal procedure", "work instruction")):
        if a == "SOP":
            score += 0.22

    # Generic quality priors (small).
    if "aide-memoire" in fn or "aide-memoire" in text_l:
        score -= 0.06
    if "manual-" in fn:
        score -= 0.04
    if "table of contents" in text_l:
        score -= 0.06

    # Penalize likely extraction artifacts / diagram fragments.
    if re.search(r"\b(output / result|risk assessment initiate|risk communication risk control)\b", text_l):
        score -= 0.10
    if len(text_l) < 80:
        score -= 0.03

    return score


def search_chunks(
    query: str,
    *,
    data_dir: Path = Path("data"),
    scope: str = "MIXED",
    top_k: int = 8,
    max_context_chunks: int = 6,
    min_similarity: float = 0.20,
    use_mmr: bool = True,
    mmr_lambda: float = 0.60,
    allowed_docs: list[str] | None = None,
    anchor_terms: list[str] | None = None,
    chunk_chars: int = 1000,
    min_chunk_chars: int = 220,
) -> list[dict[str, Any]]:
    _ensure_index(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars)
    assert _CORPUS is not None and _VECT_WORD is not None and _VECT_CHAR is not None and _X_WORD is not None and _X_CHAR is not None

    anchors = [a.strip() for a in (anchor_terms or []) if a and a.strip()]
    q = (query or "").strip()
    q_aug = f"{q} {' '.join(anchors)}".strip()
    q_l = q_aug.lower()
    broad_requirement = _is_broad_requirement_query(q)

    q_word = _VECT_WORD.transform([q_aug])
    q_char = _VECT_CHAR.transform([q_aug])
    sims_word = cosine_similarity(q_word, _X_WORD).flatten()
    sims_char = cosine_similarity(q_char, _X_CHAR).flatten()

    q_tokens = _tokenize(q_aug)
    overlap = np.zeros(len(_CORPUS), dtype=float)
    if q_tokens:
        for i, c in enumerate(_CORPUS):
            c_tokens = _tokenize(str(c.get("text") or ""))
            overlap[i] = len(q_tokens & c_tokens) / max(1, len(q_tokens))

    base_scores = (0.62 * sims_word) + (0.28 * sims_char) + (0.10 * overlap)
    base_rank_scores = np.asarray(base_scores, dtype=float)
    for i, c in enumerate(_CORPUS):
        base_rank_scores[i] += _query_prior(
            query_l=q_l,
            authority=str(c.get("authority") or "OTHER"),
            file_name=str(c.get("file") or ""),
            chunk_text=str(c.get("text") or ""),
        )
    req_hits = np.zeros(len(_CORPUS), dtype=int)
    req_boost = np.zeros(len(_CORPUS), dtype=float)
    if broad_requirement:
        for i, c in enumerate(_CORPUS):
            hits = _requirement_signal_hits(str(c.get("text") or ""))
            req_hits[i] = int(hits)
            req_boost[i] = _requirement_signal_boost(hits)
    final_rank_scores = np.asarray(base_rank_scores + req_boost, dtype=float)

    allowed = _scope_authorities(scope)
    allowed_doc_norm: set[str] | None = None
    if allowed_docs:
        allowed_doc_norm = set()
        for d in allowed_docs:
            if not d:
                continue
            ds = str(d).replace("\\", "/").strip().lower()
            if not ds:
                continue
            allowed_doc_norm.add(ds)
            allowed_doc_norm.add(Path(ds).name.lower())

    total_chunks = len(_CORPUS)
    authority_pass_count = 0
    docs_pass_count = 0
    eligible_similarity_count = 0
    candidate_idx: list[int] = []
    for i in range(len(_CORPUS)):
        c = _CORPUS[i]
        authority = str(c.get("authority") or "OTHER")
        if authority not in allowed:
            continue
        authority_pass_count += 1
        if allowed_doc_norm is not None:
            f = str(c.get("file") or "").strip().lower()
            if f not in allowed_doc_norm:
                continue
        docs_pass_count += 1
        # Threshold invariance: eligibility is based on pre-boost score only.
        if float(base_rank_scores[i]) < float(min_similarity):
            continue
        eligible_similarity_count += 1
        candidate_idx.append(i)

    if not candidate_idx:
        scope_filtered_count = max(0, total_chunks - authority_pass_count)
        doc_filter_removed_count = max(0, authority_pass_count - docs_pass_count)
        similarity_filtered_count = max(0, docs_pass_count - eligible_similarity_count)
        _record_empty_retrieval_event(
            {
                "query_used": q_aug,
                "scope_used": str(scope or "MIXED"),
                "anchor_terms_used": list(anchors),
                "candidate_chunks_considered": int(docs_pass_count),
                "authority_pass_count": int(authority_pass_count),
                "scope_filter_removed_count": int(scope_filtered_count),
                "scope_filter_removed_any": bool(scope_filtered_count > 0),
                "allowed_docs_filter_active": bool(allowed_doc_norm is not None),
                "allowed_docs_filter_removed_count": int(doc_filter_removed_count),
                "similarity_filtered_count": int(similarity_filtered_count),
                "min_similarity": float(min_similarity),
            }
        )
        return []

    ranked_candidates = _rank_candidates_with_guardrail(
        candidate_idx=candidate_idx,
        base_scores=base_rank_scores,
        final_scores=final_rank_scores,
        corpus=_CORPUS,
    )
    top_n = max(1, int(top_k))
    candidate_idx = ranked_candidates[:top_n]

    max_ctx = max(1, int(max_context_chunks))
    if use_mmr:
        selected_idx = _mmr_select(
            candidate_idx=candidate_idx,
            relevance_scores=final_rank_scores,
            x_word=_X_WORD,
            max_items=max_ctx,
            mmr_lambda=mmr_lambda,
        )
    else:
        selected_idx = candidate_idx[:max_ctx]

    out: list[dict[str, Any]] = []
    for i in selected_idx:
        c = dict(_CORPUS[i])
        c["_base_score"] = float(base_rank_scores[i])
        c["_req_signal_hits"] = int(req_hits[i])
        c["_req_boost"] = float(req_boost[i])
        c["_final_score"] = float(final_rank_scores[i])
        c["_score"] = float(final_rank_scores[i])
        out.append(c)
    return out
