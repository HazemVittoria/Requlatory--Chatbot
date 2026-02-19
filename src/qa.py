from __future__ import annotations

import logging
import os
import re
from functools import cmp_to_key
from pathlib import Path

from .domain_lexicons import DOMAIN_LEXICONS
from .doc_router import route_docs
from .phase_llm import (
    LLMPhaseRunner,
    _enforce_phase_b_requirement_filter,
    _has_requirement_language as _phase_b_has_requirement_language,
    _is_validation_topic_gate_active as _phase_b_validation_gate_active,
    _validation_topic_coherent as _phase_b_topic_coherent,
    llm_phase_available,
)
from .query_normalization import expand_query, normalize_text
from .qa_types import AnswerResult, Citation, Fact
from . import search as search_module
from .search import search_chunks

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LOGGER = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 8
MAX_CONTEXT_CHUNKS = 8
MIN_CONTEXT_CHUNKS = 2
MIN_SIMILARITY = 0.20
FALLBACK_MIN_SIMILARITY = 0.18
FALLBACK_ONLY_IF_EMPTY = True
USE_MMR = True
MMR_LAMBDA = 0.60
PHASE_A_BALANCE_PER_AUTHORITY = 3
PHASE_A_BALANCE_POOL_CHUNKS = 80
PHASE_A_BALANCE_SOFT_FLOOR_DELTA = 0.22
MAX_SENTENCES = 6
DETERMINISTIC_PHASE_C_MAX_FACTS = 5
DETERMINISTIC_PHASE_C_MIN_SENTENCES = 3
DETERMINISTIC_MIN_DISTINCT_DOCS = 2
DETERMINISTIC_MAX_SENTENCES_PER_DOC = 2
MIN_DOMAIN_SENTENCES = 2

_REQ_TERMS = (
    "shall",
    "must",
    "should",
    "required",
    "requirement",
    "expect",
    "ensure",
    "defined",
    "responsible",
    "procedure",
    "establish",
    "maintain",
    "verify",
    "review",
)

_PHASE_C_HEADING_PATTERNS = (
    "introduction",
    "purpose",
    "scope",
    "recommendations",
    "guidance for industry",
    "table of contents",
)

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

_BROAD_REQUIREMENT_ANCHORS = (
    "validation",
    "qualification",
    "shall",
    "must",
    "required",
    "in accordance with",
    "annex 15",
    "21 cfr 211",
)

_APR_QUERY_ANCHORS = (
    "annual product review",
    "product quality review",
    "pqr",
    "annual review",
    "quality review",
)

_DATA_INTEGRITY_QUERY_ANCHORS = (
    "data integrity",
    "alcoa",
    "audit trail",
    "metadata",
    "attributable",
    "legible",
    "contemporaneous",
    "original",
    "accurate",
    "record retention",
)

_DEVIATION_QUERY_ANCHORS = (
    "deviation",
    "deviation investigation",
    "investigation",
    "root cause",
    "impact assessment",
    "corrective action",
    "preventive action",
    "capa",
    "deviation report",
)

_EQUIPMENT_QUAL_QUERY_ANCHORS = (
    "equipment qualification",
    "installation qualification",
    "operational qualification",
    "performance qualification",
    "iq",
    "oq",
    "pq",
    "qualification protocol",
    "qualification report",
    "acceptance criteria",
)

_VALIDATION_QUERY_ANCHORS = (
    "process validation",
    "stage 1",
    "stage 2",
    "stage 3",
    "lifecycle",
    "validation master plan",
    "vmp",
    "continued process verification",
    "cleaning validation",
    "revalidation",
    "ppq",
    "cpv",
)

_REQUALIFICATION_QUERY_ANCHORS = (
    "requalification",
    "periodic qualification",
    "periodic review",
    "qualification review",
    "revalidation",
    "iq",
    "oq",
    "pq",
)

_DATA_INTEGRITY_QUERY_TRIGGERS = (
    "data integrity",
    "alcoa",
    "audit trail",
    "metadata",
    "record retention",
    "true copy",
)

_VALIDATION_QUERY_TRIGGERS = (
    "process validation",
    "validation stages",
    "lifecycle approach",
    "vmp",
    "validation master plan",
    "cleaning validation",
    "revalidation",
)

_REQUALIFICATION_QUERY_TRIGGERS = (
    "requalification",
    "periodic qualification",
)

_VALIDATION_DOMAIN_QUERY_SUBSTRINGS = (
    "validation",
    "vmp",
    "revalidation",
    "cleaning validation",
    "continued process verification",
    "lifecycle",
)

_DOMAIN_DETECTION_PRIORITY = (
    "deviations",
    "equipment_qualification",
    "data_integrity",
    "computerized_systems",
    "apr",
    "validation",
)

_DEVIATION_OOS_QUERY_SUBSTRINGS = (
    "oos",
    "out-of-specification",
    "out of specification",
    "out of spec",
    "laboratory investigation",
)

_APR_PRIMARY_QUERY_SUBSTRINGS = (
    "apr",
    "pqr",
    "product quality review",
    "annual product review",
)

_RELAXED_DOMAIN_CHUNK_FILTER_KEYS = {
    "data_integrity",
    "computerized_systems",
    "apr",
    "validation",
}

_ALLOWED_DOCS_RELAXED_RETRY_KEYS = {
    "computerized_systems",
    "data_integrity",
}

_PHASE_C_DEBUG_IDS = {"ev037", "ev042"}


def _fallback_debug(
    *,
    case_id: str | None,
    branch: str,
    retrieved_chunks: int,
    used_sentences: int,
    citations_count: int,
    flags: dict[str, object],
) -> None:
    if case_id not in _PHASE_C_DEBUG_IDS:
        return
    print(
        f"[FALLBACK] case={case_id} branch={branch} retrieved_chunks={int(retrieved_chunks)} "
        f"used_sentences={int(used_sentences)} citations={int(citations_count)} flags={flags}"
    )


def _intent(question: str) -> str:
    q = (question or "").strip().lower()
    if q.startswith("what is") or q.startswith("define") or "constitutes" in q:
        return "definition"
    if any(k in q for k in ("how should", "steps", "investigat", "handle", "document")):
        return "procedure"
    if any(k in q for k in ("required", "requirements", "expectations", "must")):
        return "requirements"
    return "unknown"


def _tokens(text: str) -> set[str]:
    stop = {"what", "how", "when", "where", "which", "should", "this", "that", "with", "from", "into"}
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in stop}


def _query_doc_hints(question: str) -> set[str]:
    q = (question or "").lower()
    hints: set[str] = set()
    for n in re.findall(r"\bq\s*[- ]?(\d{1,2})(?:\s*\(r\d+\))?\b", q):
        hints.add(f"q{n}")
    for n in re.findall(r"\bpart\s*(\d{1,3})\b", q):
        hints.add(f"part{n}")
    return hints


def _fallback_pdf_match(question: str, top_pdf: str) -> bool:
    pdf_l = (top_pdf or "").lower()
    if not pdf_l:
        return False
    keywords = {t for t in _tokens(question) if len(t) >= 4}
    keywords |= _query_doc_hints(question)
    if not keywords:
        return False
    pdf_norm = re.sub(r"[^a-z0-9]+", "", pdf_l)
    for kw in keywords:
        k = re.sub(r"[^a-z0-9]+", "", kw.lower())
        if k and k in pdf_norm:
            return True
    return False


def _is_broad_requirement_query(query_norm: str) -> bool:
    q = normalize_text(query_norm or "")
    if not q:
        return False
    if not re.search(r"\bgmp\b", q):
        return False
    if not any(re.search(rf"\b{re.escape(term)}\b", q) for term in _BROAD_REQUIREMENT_INCLUDE_TERMS):
        return False
    if any(phrase in q for phrase in _BROAD_REQUIREMENT_EXCLUDE_PHRASES):
        return False
    return True


def _phase_a_query_variants(question: str) -> list[str]:
    q_norm = normalize_text(question)
    variants = expand_query(q_norm)
    seen = set(variants)

    if "continuous processing" in q_norm:
        targeted = q_norm
        if "continuous manufacturing" not in targeted:
            targeted = f"{targeted} continuous manufacturing"
        if not re.search(r"\bich\s*q\s*13\b", targeted):
            targeted = f"{targeted} ich q13"
        targeted = normalize_text(targeted)
        if targeted and targeted not in seen:
            variants.append(targeted)
            seen.add(targeted)

    if _is_broad_requirement_query(q_norm):
        expanded = normalize_text(f"{q_norm} {' '.join(_BROAD_REQUIREMENT_ANCHORS)}")
        if expanded and expanded not in seen:
            variants.append(expanded)
            seen.add(expanded)
    return variants


def _is_apr_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    q_pad = f" {q} "
    if " apr " in q_pad or " pqr " in q_pad:
        return True
    if "annual product review" in q or "product quality review" in q:
        return True
    if "annual" in q and "review" in q:
        return True
    if "product" in q and "review" in q:
        return True
    return False


def _apr_anchor_terms(question: str) -> list[str]:
    if not _is_apr_query(question):
        return []
    return [a for a in _APR_QUERY_ANCHORS]


def _normalize_substring_text(text: str) -> str:
    s = "".join(ch if ch.isalnum() else " " for ch in (text or "").lower())
    return " ".join(s.split())


def _is_deviation_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    q_pad = f" {q} "
    has_deviation = ("deviation" in q) or ("nonconformance" in q) or ("non conformance" in q)
    if has_deviation:
        return True
    if "root cause" in q:
        return True
    if "investigation" in q and has_deviation:
        return True
    if "capa" in q and ("deviation" in q or "investigation" in q):
        return True
    # Keep exact-token behavior for single acronyms.
    if " capa " in q_pad and (" deviation " in q_pad or " investigation " in q_pad):
        return True
    return False


def _deviation_anchor_terms(question: str) -> list[str]:
    if not _is_deviation_query(question):
        return []
    return [a for a in _DEVIATION_QUERY_ANCHORS]


def _is_data_integrity_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    for trigger in _DATA_INTEGRITY_QUERY_TRIGGERS:
        if trigger in q:
            return True
    return False


def _data_integrity_anchor_terms(question: str) -> list[str]:
    if not _is_data_integrity_query(question):
        return []
    return [a for a in _DATA_INTEGRITY_QUERY_ANCHORS]


def _is_equipment_qualification_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    q_pad = f" {q} "
    if " iq " in q_pad or " oq " in q_pad or " pq " in q_pad:
        return True
    if "installation qualification" in q:
        return True
    if "operational qualification" in q:
        return True
    if "performance qualification" in q:
        return True
    if "equipment qualification" in q:
        return True
    if "qualification protocol" in q:
        return True
    if "qualification report" in q:
        return True
    return False


def _equipment_qualification_anchor_terms(question: str) -> list[str]:
    if not _is_equipment_qualification_query(question):
        return []
    return [a for a in _EQUIPMENT_QUAL_QUERY_ANCHORS]


def _is_validation_anchor_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    for trigger in _VALIDATION_QUERY_TRIGGERS:
        if trigger in q:
            return True
    return False


def _validation_anchor_terms(question: str) -> list[str]:
    if not _is_validation_anchor_query(question):
        return []
    return [a for a in _VALIDATION_QUERY_ANCHORS]


def _is_requalification_query(question: str) -> bool:
    q_raw = (question or "")
    if not q_raw:
        return False
    q = _normalize_substring_text(q_raw)
    for trigger in _REQUALIFICATION_QUERY_TRIGGERS:
        if trigger in q:
            return True
    return False


def _requalification_anchor_terms(question: str) -> list[str]:
    if not _is_requalification_query(question):
        return []
    return [a for a in _REQUALIFICATION_QUERY_ANCHORS]


def _targeted_anchor_terms(question: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for src in (
        _apr_anchor_terms(question),
        _data_integrity_anchor_terms(question),
        _deviation_anchor_terms(question),
        _equipment_qualification_anchor_terms(question),
        _validation_anchor_terms(question),
        _requalification_anchor_terms(question),
    ):
        for t in src:
            ts = str(t).strip().lower()
            if not ts or ts in seen:
                continue
            seen.add(ts)
            out.append(ts)
    return out


def _expanded_query_with_anchors(question: str, anchor_terms: list[str]) -> str:
    q = normalize_text(question or "")
    if not q or not anchor_terms:
        return ""
    return normalize_text(f"{q} {' '.join(anchor_terms)}")


def _phase_a_chunk_preview(chunks: list[dict]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for c in chunks:
        base_score = float(c.get("_base_score", c.get("_score", 0.0)))
        final_score = float(c.get("_final_score", c.get("_score", 0.0)))
        out.append(
            {
                "pdf": str(c.get("file") or ""),
                "authority": str(c.get("authority") or c.get("source") or "OTHER"),
                "page": int(c.get("page") or 0),
                "chunk_id": str(c.get("chunk_id") or ""),
                "score": float(c.get("_score", 0.0)),
                "base_score": base_score,
                "req_signal_hits": int(c.get("_req_signal_hits", 0)),
                "req_boost": float(c.get("_req_boost", 0.0)),
                "final_score": final_score,
                "quote_preview": _clip_snippet(str(c.get("text") or ""), max_chars=220),
            }
        )
    return out


def _chunk_key(chunk: dict) -> tuple[str, int, str]:
    return (
        str(chunk.get("file") or ""),
        int(chunk.get("page") or 0),
        str(chunk.get("chunk_id") or ""),
    )


def _chunk_sort_key(chunk: dict) -> tuple[float, str, int, str]:
    return (
        -float(chunk.get("_final_score", chunk.get("_score", 0.0))),
        str(chunk.get("file") or ""),
        int(chunk.get("page") or 0),
        str(chunk.get("chunk_id") or ""),
    )


def _chunk_authority(chunk: dict) -> str:
    authority = str(chunk.get("authority") or chunk.get("source") or "OTHER").strip().upper()
    return authority or "OTHER"


def _chunk_base_score(chunk: dict) -> float:
    return float(chunk.get("_base_score", chunk.get("_score", 0.0)))


def _chunk_req_hits(chunk: dict) -> int:
    try:
        return int(chunk.get("_req_signal_hits", 0))
    except Exception:
        return 0


def _apply_domain_chunk_post_filter(
    chunks: list[dict],
    *,
    domain_key: str | None,
) -> tuple[list[dict], dict[str, object]]:
    before_count = int(len(chunks))
    stats: dict[str, object] = {
        "chunks_before": before_count,
        "chunks_after_domain_chunk_filter": before_count,
        "chunks_after_level1_strict": before_count,
        "chunks_after_level2_relaxed": before_count,
        "domain_chunk_filter_used": False,
        "domain_chunk_filter_level_used": 0,
        "fallback_used": False,
    }
    if not domain_key or domain_key not in DOMAIN_LEXICONS:
        return list(chunks), stats

    cfg = DOMAIN_LEXICONS.get(domain_key, {})
    strong_tokens = list(cfg.get("strong_tokens", []))
    secondary_tokens = list(cfg.get("secondary_tokens", []))
    negative_tokens = list(cfg.get("negative_tokens", []))

    def _chunk_passes_rule(chunk_text: str, *, min_secondary: int) -> bool:
        t = _match_normalized_text(chunk_text)
        if not t:
            return False
        if _unique_token_hits(t, negative_tokens):
            return False
        strong_hits = _unique_token_hits(t, strong_tokens)
        secondary_hits = _unique_token_hits(t, secondary_tokens)
        return (len(strong_hits) >= 1) or (len(secondary_hits) >= int(min_secondary))

    strict_filtered = [
        c for c in chunks if _chunk_passes_rule(str(c.get("text") or ""), min_secondary=2)
    ]
    strict_count = int(len(strict_filtered))
    stats["chunks_after_level1_strict"] = strict_count
    if strict_count >= int(MIN_CONTEXT_CHUNKS):
        stats["chunks_after_domain_chunk_filter"] = strict_count
        stats["domain_chunk_filter_used"] = True
        stats["domain_chunk_filter_level_used"] = 1
        return strict_filtered, stats

    relaxed_count = strict_count
    relaxed_filtered = strict_filtered
    if domain_key in _RELAXED_DOMAIN_CHUNK_FILTER_KEYS:
        relaxed_filtered = [
            c for c in chunks if _chunk_passes_rule(str(c.get("text") or ""), min_secondary=1)
        ]
        relaxed_count = int(len(relaxed_filtered))
    stats["chunks_after_level2_relaxed"] = relaxed_count
    if relaxed_count >= int(MIN_CONTEXT_CHUNKS):
        stats["chunks_after_domain_chunk_filter"] = relaxed_count
        stats["domain_chunk_filter_used"] = True
        stats["domain_chunk_filter_level_used"] = 2
        return relaxed_filtered, stats

    stats["chunks_after_domain_chunk_filter"] = before_count
    stats["fallback_used"] = True
    return list(chunks), stats


def _balance_phase_a_chunks(chunks: list[dict], *, target_k: int) -> tuple[list[dict], dict[str, object]]:
    ordered = sorted(chunks, key=_chunk_sort_key)
    if not ordered:
        return [], {
            "input_count": 0,
            "output_count": 0,
            "authorities_seen": [],
            "authorities_kept": [],
            "soft_floor_score": 0.0,
        }

    top_score = _chunk_base_score(ordered[0])
    floor_score = max(float(MIN_SIMILARITY), top_score - float(PHASE_A_BALANCE_SOFT_FLOOR_DELTA))

    by_authority: dict[str, list[dict]] = {}
    for c in ordered:
        by_authority.setdefault(_chunk_authority(c), []).append(c)

    authorities_seen = sorted(by_authority.keys())
    authority_order = sorted(
        by_authority.keys(),
        key=lambda a: (-_chunk_base_score(by_authority[a][0]), a),
    )
    eligible_authorities = [
        a for a in authority_order if _chunk_base_score(by_authority[a][0]) >= floor_score
    ]
    if len(eligible_authorities) < 2:
        out = ordered[: max(1, int(target_k))]
        return out, {
            "input_count": len(ordered),
            "output_count": len(out),
            "authorities_seen": authorities_seen,
            "authorities_kept": sorted({_chunk_authority(c) for c in out}),
            "soft_floor_score": float(floor_score),
        }

    selected: list[dict] = []
    selected_keys: set[tuple[str, int, str]] = set()

    # Round-robin per-authority quotas to preserve diversity first.
    for slot in range(max(1, int(PHASE_A_BALANCE_PER_AUTHORITY))):
        for authority in eligible_authorities:
            bucket = by_authority.get(authority, [])
            if slot >= len(bucket):
                continue
            c = bucket[slot]
            k = _chunk_key(c)
            if k in selected_keys:
                continue
            selected.append(c)
            selected_keys.add(k)
            if len(selected) >= max(1, int(target_k)):
                break
        if len(selected) >= max(1, int(target_k)):
            break

    # Soft floor: refill remaining slots by score from all authorities.
    if len(selected) < max(1, int(target_k)):
        def _refill_cmp(a: dict, b: dict) -> int:
            ba = _chunk_base_score(a)
            bb = _chunk_base_score(b)
            if abs(ba - bb) <= 0.02:
                ha = _chunk_req_hits(a)
                hb = _chunk_req_hits(b)
                if ha > hb:
                    return -1
                if ha < hb:
                    return 1
            fa = float(a.get("_final_score", a.get("_score", 0.0)))
            fb = float(b.get("_final_score", b.get("_score", 0.0)))
            if fa > fb:
                return -1
            if fa < fb:
                return 1
            if ba > bb:
                return -1
            if ba < bb:
                return 1
            ka = _chunk_key(a)
            kb = _chunk_key(b)
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        refill_order = sorted(ordered, key=cmp_to_key(_refill_cmp))
        for c in refill_order:
            k = _chunk_key(c)
            if k in selected_keys:
                continue
            selected.append(c)
            selected_keys.add(k)
            if len(selected) >= max(1, int(target_k)):
                break

    out = sorted(selected, key=_chunk_sort_key)
    return out, {
        "input_count": len(ordered),
        "output_count": len(out),
        "authorities_seen": authorities_seen,
        "authorities_kept": sorted({_chunk_authority(c) for c in out}),
        "soft_floor_score": float(floor_score),
    }


def _collect_retrieved_chunks_for_queries(
    queries: list[str],
    *,
    scope: str,
    data_dir: Path,
    chunk_chars: int,
    min_chunk_chars: int,
    min_similarity: float,
    top_k: int,
    allowed_docs: list[str] | None = None,
    anchor_terms: list[str] | None = None,
    empty_retrieval_events: list[dict[str, object]] | None = None,
) -> list[dict]:
    merged: dict[tuple[str, int, str], dict] = {}
    for q in queries:
        hits = search_chunks(
            q,
            data_dir=data_dir,
            scope=scope,
            top_k=top_k,
            max_context_chunks=MAX_CONTEXT_CHUNKS,
            min_similarity=min_similarity,
            use_mmr=USE_MMR,
            mmr_lambda=MMR_LAMBDA,
            allowed_docs=allowed_docs,
            anchor_terms=anchor_terms,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )
        if empty_retrieval_events is not None:
            empty_retrieval_events.extend(search_module.pop_empty_retrieval_events())
        for c in hits:
            key = (str(c.get("file") or ""), int(c.get("page") or 0), str(c.get("chunk_id") or ""))
            prev = merged.get(key)
            if prev is None or float(c.get("_score", 0.0)) > float(prev.get("_score", 0.0)):
                merged[key] = c
    ordered = sorted(
        merged.values(),
        key=lambda x: (
            -float(x.get("_score", 0.0)),
            str(x.get("file") or ""),
            int(x.get("page") or 0),
            str(x.get("chunk_id") or ""),
        ),
    )
    return ordered[:PHASE_A_BALANCE_POOL_CHUNKS]


def _retrieve_chunks_for_queries(
    queries: list[str],
    *,
    scope: str,
    data_dir: Path,
    chunk_chars: int,
    min_chunk_chars: int,
    min_similarity: float,
    domain_key: str | None = None,
    allowed_docs: list[str] | None = None,
    anchor_terms: list[str] | None = None,
    empty_retrieval_events: list[dict[str, object]] | None = None,
    enable_allowed_docs_relaxed_retry: bool = False,
) -> tuple[list[dict], dict[str, object]]:
    phase_a_top_k = int(RETRIEVAL_TOP_K) + 4
    attempt1_chunks = _collect_retrieved_chunks_for_queries(
        queries,
        scope=scope,
        data_dir=data_dir,
        chunk_chars=chunk_chars,
        min_chunk_chars=min_chunk_chars,
        min_similarity=min_similarity,
        top_k=phase_a_top_k,
        allowed_docs=allowed_docs,
        anchor_terms=anchor_terms,
        empty_retrieval_events=empty_retrieval_events,
    )
    pool_size_attempt1 = int(len(attempt1_chunks))
    pool_size_attempt2 = 0
    allowed_docs_relaxed_retry_used = False
    top_chunks = list(attempt1_chunks)
    if (
        enable_allowed_docs_relaxed_retry
        and domain_key in _ALLOWED_DOCS_RELAXED_RETRY_KEYS
        and allowed_docs is not None
        and pool_size_attempt1 < 3
    ):
        allowed_docs_relaxed_retry_used = True
        attempt2_chunks = _collect_retrieved_chunks_for_queries(
            queries,
            scope=scope,
            data_dir=data_dir,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
            min_similarity=min_similarity,
            top_k=phase_a_top_k,
            allowed_docs=None,
            anchor_terms=anchor_terms,
            empty_retrieval_events=empty_retrieval_events,
        )
        pool_size_attempt2 = int(len(attempt2_chunks))
        merged_chunks: dict[tuple[str, int, str], dict] = {}
        for c in top_chunks + attempt2_chunks:
            key = (str(c.get("file") or ""), int(c.get("page") or 0), str(c.get("chunk_id") or ""))
            prev = merged_chunks.get(key)
            if prev is None or float(c.get("_score", 0.0)) > float(prev.get("_score", 0.0)):
                merged_chunks[key] = c
        top_chunks = sorted(
            merged_chunks.values(),
            key=lambda x: (
                -float(x.get("_score", 0.0)),
                str(x.get("file") or ""),
                int(x.get("page") or 0),
                str(x.get("chunk_id") or ""),
            ),
        )[:PHASE_A_BALANCE_POOL_CHUNKS]

    filtered_chunks, filter_stats = _apply_domain_chunk_post_filter(
        top_chunks,
        domain_key=domain_key,
    )
    filter_stats["top_k_used"] = int(phase_a_top_k)
    filter_stats["pool_size_returned"] = int(len(top_chunks))
    filter_stats["pool_size_attempt1"] = int(pool_size_attempt1)
    filter_stats["pool_size_attempt2"] = int(pool_size_attempt2)
    filter_stats["allowed_docs_relaxed_retry_used"] = bool(allowed_docs_relaxed_retry_used)
    return filtered_chunks, filter_stats


def _clean_text(text: str) -> str:
    s = (
        (text or "")
        .replace("\uf0b7", " ")
        .replace("\u2022", " ")
        .replace("\u00a0", " ")
    )
    s = re.sub(r"(?<=\D)\b\d{2,4}\b(?=\D)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_noisy_quote(text: str) -> bool:
    s = _clean_text(text).lower()
    if not s:
        return True
    if re.search(r"\b(output / result|risk assessment initiate|risk communication risk control)\b", s):
        return True
    if len(s) < 35:
        return True
    words = s.split()
    if len(words) >= 14 and s.count(".") == 0:
        cap_like = sum(1 for w in words if w[:1].isalpha() and w[:1].upper() == w[:1] and len(w) > 3)
        if cap_like >= 7:
            return True
    return False


def _clip_snippet(text: str, max_chars: int = 420) -> str:
    s = _clean_text(text)
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    last_space = cut.rfind(" ")
    if last_space > 120:
        cut = cut[:last_space]
    return cut + " ..."


def format_facts(facts: list[Fact]) -> str:
    lines = ["FACTS:"]
    for f in facts:
        lines.append(
            f'- "{f.quote}" (pdf="{f.pdf}", page={f.page}, chunk_id="{f.chunk_id}", score={f.score:.4f})'
        )
    return "\n".join(lines)


def format_relevant_facts(facts: list[Fact]) -> str:
    lines = ["RELEVANT FACTS:"]
    for f in facts:
        lines.append(f'- "{f.quote}" (pdf="{f.pdf}", page={f.page}, chunk_id="{f.chunk_id}")')
    return "\n".join(lines)


def _phase_a_retrieval(
    question: str,
    *,
    scope: str,
    data_dir: Path,
    chunk_chars: int,
    min_chunk_chars: int,
) -> tuple[list[Fact], bool, dict[str, object]]:
    fallback_used = False
    balance_trace: dict[str, object] = {
        "input_count": 0,
        "output_count": 0,
        "authorities_seen": [],
        "authorities_kept": [],
        "soft_floor_score": 0.0,
    }
    domain_key = _domain_key_for_question(question)
    queries = _phase_a_query_variants(question)
    anchor_terms = _targeted_anchor_terms(question)
    expanded_query = _expanded_query_with_anchors(question, anchor_terms)
    retrieval_queries = list(queries)
    if expanded_query and expanded_query not in retrieval_queries:
        retrieval_queries.append(expanded_query)

    candidate_docs = route_docs(retrieval_queries, top_docs=5)
    allowed_docs = candidate_docs if candidate_docs else None
    search_module.pop_empty_retrieval_events()
    empty_retrieval_events: list[dict[str, object]] = []
    empty_result_scope_retry_attempted = False
    empty_result_scope_retry_used = False
    attempts: list[dict[str, object]] = []
    chunks, chunk_filter_stats = _retrieve_chunks_for_queries(
        retrieval_queries,
        data_dir=data_dir,
        scope=scope,
        min_similarity=MIN_SIMILARITY,
        domain_key=domain_key,
        chunk_chars=chunk_chars,
        min_chunk_chars=min_chunk_chars,
        allowed_docs=allowed_docs,
        anchor_terms=anchor_terms,
        empty_retrieval_events=empty_retrieval_events,
        enable_allowed_docs_relaxed_retry=True,
    )
    attempts.append(
        {
            "min_similarity": float(MIN_SIMILARITY),
            "scope": str(scope or "MIXED"),
            "allowed_docs_restricted": bool(allowed_docs),
            "chunks_before": int(chunk_filter_stats.get("chunks_before", 0)),
            "top_k_used": int(chunk_filter_stats.get("top_k_used", 0)),
            "pool_size_returned": int(chunk_filter_stats.get("pool_size_returned", 0)),
            "pool_size_attempt1": int(chunk_filter_stats.get("pool_size_attempt1", 0)),
            "pool_size_attempt2": int(chunk_filter_stats.get("pool_size_attempt2", 0)),
            "allowed_docs_relaxed_retry_used": bool(
                chunk_filter_stats.get("allowed_docs_relaxed_retry_used", False)
            ),
            "chunks_after_domain_chunk_filter": int(
                chunk_filter_stats.get("chunks_after_domain_chunk_filter", 0)
            ),
            "chunks_after_level1_strict": int(
                chunk_filter_stats.get("chunks_after_level1_strict", 0)
            ),
            "chunks_after_level2_relaxed": int(
                chunk_filter_stats.get("chunks_after_level2_relaxed", 0)
            ),
            "domain_chunk_filter_used": bool(
                chunk_filter_stats.get("domain_chunk_filter_used", False)
            ),
            "domain_chunk_filter_level_used": int(
                chunk_filter_stats.get("domain_chunk_filter_level_used", 0)
            ),
            "fallback_used": bool(chunk_filter_stats.get("fallback_used", False)),
            "top_results": _phase_a_chunk_preview(chunks),
        }
    )
    if not chunks:
        empty_result_scope_retry_attempted = True
        mixed_scope_chunks, mixed_chunk_filter_stats = _retrieve_chunks_for_queries(
            retrieval_queries,
            data_dir=data_dir,
            scope="MIXED",
            min_similarity=MIN_SIMILARITY,
            domain_key=domain_key,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
            allowed_docs=None,
            anchor_terms=anchor_terms,
            empty_retrieval_events=empty_retrieval_events,
        )
        attempts.append(
            {
                "min_similarity": float(MIN_SIMILARITY),
                "scope": "MIXED",
                "allowed_docs_restricted": False,
                "retry_reason": "empty_primary_relaxed_scope",
                "chunks_before": int(mixed_chunk_filter_stats.get("chunks_before", 0)),
                "top_k_used": int(mixed_chunk_filter_stats.get("top_k_used", 0)),
                "pool_size_returned": int(mixed_chunk_filter_stats.get("pool_size_returned", 0)),
                "pool_size_attempt1": int(mixed_chunk_filter_stats.get("pool_size_attempt1", 0)),
                "pool_size_attempt2": int(mixed_chunk_filter_stats.get("pool_size_attempt2", 0)),
                "allowed_docs_relaxed_retry_used": bool(
                    mixed_chunk_filter_stats.get("allowed_docs_relaxed_retry_used", False)
                ),
                "chunks_after_domain_chunk_filter": int(
                    mixed_chunk_filter_stats.get("chunks_after_domain_chunk_filter", 0)
                ),
                "chunks_after_level1_strict": int(
                    mixed_chunk_filter_stats.get("chunks_after_level1_strict", 0)
                ),
                "chunks_after_level2_relaxed": int(
                    mixed_chunk_filter_stats.get("chunks_after_level2_relaxed", 0)
                ),
                "domain_chunk_filter_used": bool(
                    mixed_chunk_filter_stats.get("domain_chunk_filter_used", False)
                ),
                "domain_chunk_filter_level_used": int(
                    mixed_chunk_filter_stats.get("domain_chunk_filter_level_used", 0)
                ),
                "fallback_used": bool(mixed_chunk_filter_stats.get("fallback_used", False)),
                "top_results": _phase_a_chunk_preview(mixed_scope_chunks),
            }
        )
        if mixed_scope_chunks:
            chunks = mixed_scope_chunks
            empty_result_scope_retry_used = True
    if not chunks and FALLBACK_ONLY_IF_EMPTY:
        fallback_chunks, fallback_chunk_filter_stats = _retrieve_chunks_for_queries(
            retrieval_queries,
            data_dir=data_dir,
            scope=scope,
            min_similarity=FALLBACK_MIN_SIMILARITY,
            domain_key=domain_key,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
            allowed_docs=allowed_docs,
            anchor_terms=anchor_terms,
            empty_retrieval_events=empty_retrieval_events,
        )
        attempts.append(
            {
                "min_similarity": float(FALLBACK_MIN_SIMILARITY),
                "scope": str(scope or "MIXED"),
                "allowed_docs_restricted": bool(allowed_docs),
                "chunks_before": int(fallback_chunk_filter_stats.get("chunks_before", 0)),
                "top_k_used": int(fallback_chunk_filter_stats.get("top_k_used", 0)),
                "pool_size_returned": int(fallback_chunk_filter_stats.get("pool_size_returned", 0)),
                "pool_size_attempt1": int(fallback_chunk_filter_stats.get("pool_size_attempt1", 0)),
                "pool_size_attempt2": int(fallback_chunk_filter_stats.get("pool_size_attempt2", 0)),
                "allowed_docs_relaxed_retry_used": bool(
                    fallback_chunk_filter_stats.get("allowed_docs_relaxed_retry_used", False)
                ),
                "chunks_after_domain_chunk_filter": int(
                    fallback_chunk_filter_stats.get("chunks_after_domain_chunk_filter", 0)
                ),
                "chunks_after_level1_strict": int(
                    fallback_chunk_filter_stats.get("chunks_after_level1_strict", 0)
                ),
                "chunks_after_level2_relaxed": int(
                    fallback_chunk_filter_stats.get("chunks_after_level2_relaxed", 0)
                ),
                "domain_chunk_filter_used": bool(
                    fallback_chunk_filter_stats.get("domain_chunk_filter_used", False)
                ),
                "domain_chunk_filter_level_used": int(
                    fallback_chunk_filter_stats.get("domain_chunk_filter_level_used", 0)
                ),
                "fallback_used": bool(fallback_chunk_filter_stats.get("fallback_used", False)),
                "top_results": _phase_a_chunk_preview(fallback_chunks),
            }
        )
        if fallback_chunks:
            top1 = fallback_chunks[0]
            top1_score = float(top1.get("_score", 0.0))
            top1_pdf = str(top1.get("file") or "")
            if top1_score >= FALLBACK_MIN_SIMILARITY and _fallback_pdf_match(question, top1_pdf):
                chunks = fallback_chunks
                fallback_used = True
    chunks, balance_trace = _balance_phase_a_chunks(chunks, target_k=MAX_CONTEXT_CHUNKS)
    out: list[Fact] = []
    for c in chunks:
        out.append(
            Fact(
                quote=str(c.get("text") or ""),
                pdf=str(c.get("file") or ""),
                page=int(c.get("page") or 0),
                chunk_id=str(c.get("chunk_id") or ""),
                score=float(c.get("_score", 0.0)),
            )
        )
    trace = {
        "queries_used": queries,
        "anchor_terms_used": anchor_terms,
        "expanded_query_used": expanded_query,
        "candidate_docs": candidate_docs,
        "thresholds_attempted": [float(a["min_similarity"]) for a in attempts],
        "attempts": attempts,
        "fallback_used": bool(fallback_used),
        "empty_result_scope_retry_attempted": bool(empty_result_scope_retry_attempted),
        "empty_result_scope_retry_used": bool(empty_result_scope_retry_used),
        "empty_retrieval_events": empty_retrieval_events,
        "balancing": balance_trace,
    }
    return out, fallback_used, trace


def _similarity_key(text: str) -> str:
    s = _clean_text(text).lower()
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(s.split())


def _is_relevant_fact(question: str, fact: Fact) -> bool:
    q_tokens = _tokens(question)
    f_tokens = _tokens(fact.quote)
    if not f_tokens:
        return False
    overlap = len(q_tokens & f_tokens)
    if overlap >= 2:
        return True
    if overlap >= 1 and fact.score >= max(MIN_SIMILARITY, 0.28):
        return True
    ql = (question or "").lower()
    fl = fact.quote.lower()
    if "risk management" in ql and ("risk management" in fl or "quality risk management" in fl):
        return True
    if "data integrity" in ql and ("data integrity" in fl or "alcoa" in fl):
        return True
    return False


def _requires_external_live_data(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if "market share" in q:
        return True
    if "average cost" in q:
        return True
    if "failure rate" in q and ("current" in q or re.search(r"\b20\d{2}\b", q)):
        return True
    if "who is" in q and "head of" in q:
        return True
    if "how many" in q and ("warning letter" in q or "warning letters" in q) and "last month" in q:
        return True
    return False


def _phase_b_filter(question: str, facts: list[Fact]) -> list[Fact]:
    ordered = sorted(facts, key=lambda x: (-x.score, x.pdf, x.page, x.chunk_id))
    out: list[Fact] = []
    seen_keys: set[str] = set()
    for f in ordered:
        if _is_noisy_quote(f.quote):
            continue
        if not _is_relevant_fact(question, f):
            continue
        k = _similarity_key(f.quote)
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)
        out.append(f)
        if len(out) >= MAX_CONTEXT_CHUNKS:
            break
    filtered = _enforce_phase_b_requirement_filter(question=question, facts=out, source_facts=ordered)
    return filtered[:MAX_CONTEXT_CHUNKS]


def _phase_b_empty_reason_summary(
    *,
    question: str,
    phase_a_facts: list[Fact],
    phase_a_trace: dict[str, object],
) -> dict[str, object]:
    attempts = list((phase_a_trace or {}).get("attempts", []) or [])
    fallback_used = bool((phase_a_trace or {}).get("fallback_used", False))
    active_attempt = dict(attempts[-1] if (attempts and fallback_used) else (attempts[0] if attempts else {}))
    top_rows = list(active_attempt.get("top_results", []) or [])
    row_index: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in top_rows:
        if not isinstance(row, dict):
            continue
        k = (
            str(row.get("pdf") or ""),
            int(row.get("page") or 0),
            str(row.get("chunk_id") or ""),
        )
        row_index[k] = row

    ordered = sorted(phase_a_facts, key=lambda x: (-x.score, x.pdf, x.page, x.chunk_id))
    requirement_pass = [f for f in ordered if _phase_b_has_requirement_language(f.quote)]
    validation_gate = _phase_b_validation_gate_active(question)
    if validation_gate:
        topic_pass = [f for f in requirement_pass if _phase_b_topic_coherent(f.quote, question=question)]
    else:
        topic_pass = list(requirement_pass)

    top_candidates: list[dict[str, object]] = []
    for f in ordered[:5]:
        k = (f.pdf, int(f.page), f.chunk_id)
        meta = row_index.get(k, {})
        top_candidates.append(
            {
                "pdf": f.pdf,
                "page": int(f.page),
                "chunk_id": f.chunk_id,
                "authority": str(meta.get("authority") or "OTHER"),
                "base_score": float(meta.get("base_score", f.score)),
                "req_signal_hits": int(meta.get("req_signal_hits", 0)),
                "score": float(f.score),
            }
        )

    return {
        "reason": "phase_b_filters_removed_all_candidates",
        "phase_a_candidate_count": len(ordered),
        "phase_a_pool_count": len(top_rows),
        "requirement_bearing_count": len(requirement_pass),
        "validation_topic_gate_active": bool(validation_gate),
        "topic_coherence_count": len(topic_pass),
        "top_candidates": top_candidates,
    }


def _sentence_from_fact(question: str, fact: Fact) -> str | None:
    ql = (question or "").lower()
    best: tuple[float, str] | None = None
    for part in _SENT_SPLIT_RE.split(_clean_text(fact.quote)):
        s = part.strip()
        if len(s) < 35:
            continue
        if _is_noisy_quote(s):
            continue
        s = re.sub(r"^\d+\s+", "", s).strip()
        if s and s[0].islower():
            s = s[:1].upper() + s[1:]
        sl = s.lower()
        score = 0.0
        if any(x in sl for x in (" is ", " refers to ", " consists of ", " is defined as ", " means ")):
            score += 0.45
        if "risk management" in ql and "risk management" in sl:
            score += 0.30
        if "data integrity" in ql and "data integrity" in sl:
            score += 0.30
        if sl.startswith("since "):
            score -= 0.15
        if sl.startswith("an example"):
            score -= 0.12
        score += min(0.25, 0.03 * len(_tokens(ql) & _tokens(sl)))
        if best is None or score > best[0]:
            best = (score, s)
    if best is None:
        return None
    return best[1]


def _has_requirement_language(text: str) -> bool:
    s = (text or "").lower()
    if not s:
        return False
    for t in _REQ_TERMS:
        if re.search(rf"\b{re.escape(t)}\w*\b", s):
            return True
    return False


def _looks_like_heading_text(text: str) -> bool:
    s = " ".join((text or "").split()).strip()
    if not s:
        return True
    sl = s.lower()
    if sl.endswith(":"):
        return True
    if any(sl.startswith(p) for p in _PHASE_C_HEADING_PATTERNS):
        return True
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        if len(letters) >= 6 and upper_ratio >= 0.85:
            return True
    return False


def _split_complete_sentences(text: str) -> list[str]:
    s = _clean_text(text)
    if not s:
        return []
    # Split on sentence terminators while keeping complete sentence boundaries.
    parts = re.split(r"(?<=[\.;:])\s+", s)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p[-1] not in ".;:":
            continue
        if _looks_like_heading_text(p):
            continue
        alpha_tokens = re.findall(r"[a-zA-Z]{2,}", p)
        if len(alpha_tokens) < 4:
            continue
        out.append(p)
    return out


def _normalize_known_acronyms(text: str) -> str:
    s = text or ""
    for ac in ("FDA", "GMP", "CGMP", "ICH", "API", "OOS", "EMA", "WHO"):
        s = re.sub(rf"\b{ac.lower()}\b", ac, s, flags=re.IGNORECASE)
    s = re.sub(r"\bpic\s*/\s*s\b", "PIC/S", s, flags=re.IGNORECASE)
    return s


def _trim_sentence_words(sentence: str, max_words: int = 30) -> str:
    s = " ".join((sentence or "").split()).strip()
    if not s:
        return ""
    words = s.split()
    if len(words) > max_words:
        words = words[:max_words]
        while words and re.fullmatch(r"(a|an|the|and|or|to|of|for|with|in|on|at|by)", words[-1].strip(".,;:").lower()):
            words.pop()
        if len(words) < 4:
            return ""
        s = " ".join(words)
        if s[-1] not in ".;:":
            s += "."
    return s


def _normalize_sentence_casing(sentence: str) -> str:
    s = _normalize_known_acronyms(sentence)
    if not s:
        return ""
    # Capitalize first alphabetic character.
    for i, ch in enumerate(s):
        if ch.isalpha():
            if ch.islower():
                s = s[:i] + ch.upper() + s[i + 1 :]
            break
    return s


def _looks_truncated_ending(sentence: str) -> bool:
    s = (sentence or "").strip().lower()
    if not s:
        return True
    if re.search(r"\b(with|and|or|to|of|for|in|on|at|by|a|an|the)\.$", s):
        return True
    if s.endswith("with a.") or s.endswith("and ."):
        return True
    return False


def _extract_clean_requirement_sentences(quote: str, *, max_items: int = 2) -> list[str]:
    out, _counts, _reasons = _extract_clean_requirement_sentences_with_trace(
        quote,
        max_items=max_items,
    )
    return out


def _extract_clean_requirement_sentences_with_trace(
    quote: str,
    *,
    max_items: int = 2,
) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    candidates = _split_complete_sentences(quote)
    counts = {
        "n_after_split": int(len(candidates)),
        "n_after_sentence_cleaning": 0,
        "n_after_dedupe": 0,
    }
    reasons: dict[str, list[str]] = {
        "sentence_cleaning": [],
        "dedupe": [],
    }
    if not candidates:
        return [], counts, reasons

    ordered = sorted(
        candidates,
        key=lambda s: (
            0 if _has_requirement_language(s) else 1,
            len(s.split()),
        ),
    )

    cleaned: list[str] = []
    for s in ordered:
        trimmed = _trim_sentence_words(s, max_words=30)
        if not trimmed:
            reasons["sentence_cleaning"].append("empty after clean")
            continue
        normalized = _normalize_sentence_casing(trimmed)
        if not normalized:
            reasons["sentence_cleaning"].append("empty after casing normalize")
            continue
        if _looks_like_heading_text(normalized):
            reasons["sentence_cleaning"].append("failed heading filter")
            continue
        if _looks_truncated_ending(normalized):
            reasons["sentence_cleaning"].append("failed truncated ending filter")
            continue
        if normalized[-1] not in ".;:":
            reasons["sentence_cleaning"].append("failed terminal punctuation filter")
            continue
        cleaned.append(normalized)
    counts["n_after_sentence_cleaning"] = int(len(cleaned))

    deduped: list[str] = []
    seen: set[str] = set()
    for normalized in cleaned:
        k = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
        if not k:
            reasons["dedupe"].append("empty dedupe key")
            continue
        if k in seen:
            reasons["dedupe"].append("duplicate sentence")
            continue
        seen.add(k)
        deduped.append(normalized)
    counts["n_after_dedupe"] = int(len(deduped))

    return deduped[:max_items], counts, reasons


def _match_normalized_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _unique_token_hits(text: str, tokens: list[str] | tuple[str, ...]) -> set[str]:
    t = _match_normalized_text(text)
    hits: set[str] = set()
    if not t:
        return hits
    for tok in tokens:
        s = _match_normalized_text(str(tok))
        if s and s in t:
            hits.add(s)
    return hits


def _contains_any_substring(text: str, substrings: tuple[str, ...]) -> bool:
    t = _match_normalized_text(text)
    if not t:
        return False
    for s in substrings:
        if _match_normalized_text(s) in t:
            return True
    return False


def _has_oos_deviation_trigger(question_norm: str) -> bool:
    if _contains_any_substring(question_norm, _DEVIATION_OOS_QUERY_SUBSTRINGS):
        return True
    return ("phase i" in question_norm) and ("phase ii" in question_norm)


def _apr_trigger_allowed(question_norm: str) -> bool:
    if _has_oos_deviation_trigger(question_norm):
        return False
    return _contains_any_substring(question_norm, _APR_PRIMARY_QUERY_SUBSTRINGS)


def _domain_score_for_question(question_norm: str, domain_key: str) -> int:
    cfg = DOMAIN_LEXICONS.get(domain_key, {})
    strong_hits = _unique_token_hits(question_norm, list(cfg.get("strong_tokens", [])))
    secondary_hits = _unique_token_hits(question_norm, list(cfg.get("secondary_tokens", [])))
    return (3 * len(strong_hits)) + len(secondary_hits)


def _domain_key_for_question(question: str) -> str | None:
    q = _match_normalized_text(normalize_text(question or ""))
    if not q:
        return None

    # Deterministic OOS/deviation trigger must override APR and others.
    if _has_oos_deviation_trigger(q):
        return "deviations"

    best_key: str | None = None
    best_score = 0
    for key in _DOMAIN_DETECTION_PRIORITY:
        if key not in DOMAIN_LEXICONS:
            continue
        if key == "apr" and not _apr_trigger_allowed(q):
            continue
        score = _domain_score_for_question(q, key)
        if key == "validation" and _contains_any_substring(q, _VALIDATION_DOMAIN_QUERY_SUBSTRINGS):
            score = max(score, 1)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score > 0:
        return best_key
    if _apr_trigger_allowed(q):
        return "apr"
    if _contains_any_substring(q, _VALIDATION_DOMAIN_QUERY_SUBSTRINGS):
        return "validation"
    return None


def _sentence_passes_domain_lexicon(sentence_text: str, domain_key: str) -> bool:
    cfg = DOMAIN_LEXICONS.get(domain_key)
    if not isinstance(cfg, dict):
        return True

    t = _match_normalized_text(sentence_text)
    if not t:
        return False

    neg_hits = _unique_token_hits(t, list(cfg.get("negative_tokens", [])))
    if neg_hits:
        return False

    strong_hits = _unique_token_hits(t, list(cfg.get("strong_tokens", [])))
    secondary_hits = _unique_token_hits(t, list(cfg.get("secondary_tokens", [])))

    min_strong = int(cfg.get("min_strong", 1))
    min_secondary = int(cfg.get("min_secondary", 2))
    return len(strong_hits) >= min_strong or len(secondary_hits) >= min_secondary


def _domain_sentence_rejection_reason(sentence_text: str, domain_key: str) -> str:
    cfg = DOMAIN_LEXICONS.get(domain_key)
    if not isinstance(cfg, dict):
        return "domain not configured"

    t = _match_normalized_text(sentence_text)
    if not t:
        return "failed domain lexicon: empty normalized sentence"

    neg_hits = _unique_token_hits(t, list(cfg.get("negative_tokens", [])))
    if neg_hits:
        shown = ", ".join(sorted(neg_hits)[:3])
        return f"failed negative tokens: {shown}"

    strong_hits = _unique_token_hits(t, list(cfg.get("strong_tokens", [])))
    secondary_hits = _unique_token_hits(t, list(cfg.get("secondary_tokens", [])))
    min_strong = int(cfg.get("min_strong", 1))
    min_secondary = int(cfg.get("min_secondary", 2))
    if len(strong_hits) >= min_strong or len(secondary_hits) >= min_secondary:
        return ""
    return (
        f"failed domain lexicon: strong={len(strong_hits)} secondary={len(secondary_hits)} "
        f"required_strong={min_strong} required_secondary={min_secondary}"
    )


def _strip_requirement_prefix(sentence: str) -> str:
    s = (sentence or "").strip()
    if s.lower().startswith("requirement:"):
        return s[len("Requirement:") :].strip()
    return s


def _phase_c_sentence_candidates(
    *,
    question: str,
    selected_facts: list[Fact],
    case_id: str | None = None,
) -> tuple[list[tuple[str, Citation, float]], dict[str, object]]:
    debug = bool(case_id in _PHASE_C_DEBUG_IDS)
    candidates_all: list[tuple[str, Citation, float]] = []
    candidates_req: list[tuple[str, Citation, float]] = []
    n_after_split = 0
    n_after_sentence_cleaning = 0
    n_after_dedupe = 0
    reasons_sentence_cleaning: list[str] = []
    reasons_dedupe: list[str] = []
    reasons_requirement: list[str] = []
    reasons_domain: list[str] = []
    per_fact_limit = 2 if len(selected_facts) > 1 else 3
    for f in selected_facts:
        cit = Citation(doc_id=f.pdf, page=f.page, chunk_id=f.chunk_id)
        extracted, stage_counts, stage_reasons = _extract_clean_requirement_sentences_with_trace(
            f.quote,
            max_items=per_fact_limit,
        )
        n_after_split += int(stage_counts.get("n_after_split", 0))
        n_after_sentence_cleaning += int(stage_counts.get("n_after_sentence_cleaning", 0))
        n_after_dedupe += int(stage_counts.get("n_after_dedupe", 0))
        reasons_sentence_cleaning.extend(stage_reasons.get("sentence_cleaning", []))
        reasons_dedupe.extend(stage_reasons.get("dedupe", []))
        for s in extracted:
            candidates_all.append((f"Requirement: {s}" if _has_requirement_language(s) else s, cit, float(f.score)))
            if not _has_requirement_language(s):
                reasons_requirement.append(
                    "failed requirement-bearing: no requirement indicator in sentence"
                )
                continue
            candidates_req.append((f"Requirement: {s}", cit, float(f.score)))
            if len(candidates_all) >= DETERMINISTIC_PHASE_C_MAX_FACTS:
                break
        if len(candidates_all) >= DETERMINISTIC_PHASE_C_MAX_FACTS:
            break

    domain_key = _domain_key_for_question(question)
    domain_candidates: list[tuple[str, Citation, float]] = []
    if domain_key in DOMAIN_LEXICONS:
        for sent, cit, sc in candidates_req:
            plain = _strip_requirement_prefix(sent)
            if _sentence_passes_domain_lexicon(plain, domain_key):
                domain_candidates.append((sent, cit, sc))
            else:
                reasons_domain.append(_domain_sentence_rejection_reason(plain, domain_key))

    use_domain_filter = bool(domain_key in DOMAIN_LEXICONS and len(domain_candidates) >= MIN_DOMAIN_SENTENCES)
    if domain_key in DOMAIN_LEXICONS:
        chosen = domain_candidates if use_domain_filter else candidates_req
    else:
        chosen = candidates_all
    n_final_used = int(len(chosen[:DETERMINISTIC_PHASE_C_MAX_FACTS]))

    if debug:
        n_after_requirement_gate = int(len(candidates_req))
        n_after_domain_gate = int(len(domain_candidates))
        print(
            f"[PHASEC_STAGES] case={case_id} split={n_after_split} "
            f"clean={n_after_sentence_cleaning} dedupe={n_after_dedupe} "
            f"req={n_after_requirement_gate} domain={n_after_domain_gate} "
            f"used={n_final_used}"
        )
        stage_names = [
            "split",
            "sentence_cleaning",
            "dedupe",
            "requirement_gate",
            "domain_gate",
            "final_used",
        ]
        stage_values = [
            int(n_after_split),
            int(n_after_sentence_cleaning),
            int(n_after_dedupe),
            int(n_after_requirement_gate),
            int(n_after_domain_gate),
            int(n_final_used),
        ]
        collapse_stage = ""
        for idx in range(1, len(stage_values)):
            if stage_values[idx - 1] > 0 and stage_values[idx] == 0:
                collapse_stage = stage_names[idx]
                break
        if collapse_stage:
            print(f"[PHASEC_COLLAPSE] case={case_id} stage={collapse_stage}")
            reason_map = {
                "sentence_cleaning": reasons_sentence_cleaning,
                "dedupe": reasons_dedupe,
                "requirement_gate": reasons_requirement,
                "domain_gate": reasons_domain,
                "final_used": [],
            }
            stage_reasons = reason_map.get(collapse_stage, [])
            for ridx, reason in enumerate(stage_reasons[:10], start=1):
                print(f"[PHASEC_REASON] case={case_id} stage={collapse_stage} #{ridx}: {reason}")
        elif stage_values[0] > 0 and int(n_after_split) == 0:
            print(f"[PHASEC_COLLAPSE] case={case_id} stage=bookkeeping_bug")

    stats = {
        "domain_key": domain_key,
        "N_total_sentences": int(n_after_dedupe),
        "N_requirement_bearing": int(len(candidates_req)),
        "N_domain_eligible": int(len(domain_candidates)),
        "domain_filter_used": bool(use_domain_filter),
        "sentences_extracted_total": int(n_after_split),
        "sentences_after_cleaning": int(n_after_sentence_cleaning),
        "sentences_after_dedupe": int(n_after_dedupe),
        "sentences_after_requirement_gate": int(len(candidates_req)),
        "sentences_after_domain_sentence_gate": int(len(domain_candidates)),
        "sentences_final_used": int(n_final_used),
    }
    _LOGGER.debug("phase_c_domain_filter_stats=%s", stats)
    return chosen[:DETERMINISTIC_PHASE_C_MAX_FACTS], stats


def _select_phase_c_facts(relevant: list[Fact], *, case_id: str | None = None) -> list[Fact]:
    if not relevant:
        return []
    debug = bool(case_id in _PHASE_C_DEBUG_IDS)
    if debug:
        print(f"[PHASEC] case={case_id} n_chunks={len(relevant)}")
    ordered = sorted(
        relevant,
        key=lambda f: (
            0 if _has_requirement_language(f.quote) else 1,
            -float(f.score),
            f.pdf,
            f.page,
            f.chunk_id,
        ),
    )
    selected: list[Fact] = []
    per_doc: dict[str, int] = {}
    for f in ordered:
        if debug:
            raw = f.quote
            raw_type = type(raw).__name__
            raw_len = len(raw) if isinstance(raw, str) else -1
            raw_preview = repr(raw[:180]) if isinstance(raw, str) else repr(raw)[:180]
            cleaned = _clean_text(raw) if isinstance(raw, str) else ""
            clean_len = len(cleaned)
            clean_preview = repr(cleaned[:180])
            print(
                f"[CHUNK] file={f.pdf} page={int(f.page)} id={f.chunk_id} "
                f"raw_type={raw_type} raw_len={raw_len} clean_len={clean_len} "
                f"raw_preview={raw_preview} clean_preview={clean_preview}"
            )
            sents_dbg = _split_complete_sentences(cleaned)
            first_dbg = repr(sents_dbg[0][:120]) if sents_dbg else "NONE"
            print(f"[SPLIT] n_sents={len(sents_dbg)} first={first_dbg}")
        if _is_noisy_quote(f.quote):
            continue
        if not _extract_clean_requirement_sentences(f.quote, max_items=1):
            continue
        doc_count = per_doc.get(f.pdf, 0)
        if doc_count >= 2:
            continue
        selected.append(f)
        per_doc[f.pdf] = doc_count + 1
        if len(selected) >= DETERMINISTIC_PHASE_C_MAX_FACTS:
            break
    return selected


def _deterministic_sentences_from_fact(fact: Fact, max_items: int = 2) -> list[str]:
    extracted = _extract_clean_requirement_sentences(fact.quote, max_items=max_items)
    if not extracted:
        return []
    out: list[str] = []
    for s in extracted:
        if _has_requirement_language(s):
            out.append(f"Requirement: {s}")
        else:
            out.append(s)
    return out


def _deterministic_confidence_label(
    *,
    num_sentences: int,
    distinct_docs: int,
    avg_similarity: float,
) -> str:
    if num_sentences >= 3 and distinct_docs >= 2 and avg_similarity >= 0.25:
        return "High"
    if num_sentences >= 2 and distinct_docs >= 1 and avg_similarity >= 0.22:
        return "Medium"
    return "Low"


def _confidence_label(relevant: list[Fact]) -> str:
    if not relevant:
        return "Low"
    avg = sum(max(0.0, f.score) for f in relevant) / max(1, len(relevant))
    if len(relevant) >= 4 and avg >= 0.45:
        return "High"
    if len(relevant) >= 2 and avg >= 0.30:
        return "Medium"
    return "Low"


def _phase_c_synthesis(
    question: str,
    relevant: list[Fact],
    *,
    case_id: str | None = None,
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []
    used_line_keys: set[tuple[str, int, str, str]] = set()
    selected = _select_phase_c_facts(relevant, case_id=case_id)
    candidates, _stats = _phase_c_sentence_candidates(
        question=question,
        selected_facts=selected,
        case_id=case_id,
    )

    if len(candidates) < 2:
        _fallback_debug(
            case_id=case_id,
            branch="phase_c_renderer_precheck_candidates",
            retrieved_chunks=len(relevant),
            used_sentences=0,
            citations_count=0,
            flags={
                "no_sentences_before_render": True,
                "candidates_lt_2": True,
                "fallback_before_render": False,
                "fallback_inside_renderer": True,
                "fallback_after_citation_validation": False,
            },
        )
        return "Not found in provided PDFs", []

    candidates_by_doc: dict[str, int] = {}
    for _, cit, _ in candidates:
        candidates_by_doc[cit.doc_id] = candidates_by_doc.get(cit.doc_id, 0) + 1
    possible_max = sum(min(DETERMINISTIC_MAX_SENTENCES_PER_DOC, n) for n in candidates_by_doc.values())
    target_n = min(DETERMINISTIC_PHASE_C_MAX_FACTS, len(candidates), possible_max)
    if target_n < 2:
        _fallback_debug(
            case_id=case_id,
            branch="phase_c_renderer_target_n",
            retrieved_chunks=len(relevant),
            used_sentences=0,
            citations_count=0,
            flags={
                "possible_max_lt_2": True,
                "target_n_lt_2": True,
                "fallback_before_render": False,
                "fallback_inside_renderer": True,
                "fallback_after_citation_validation": False,
            },
        )
        return "Not found in provided PDFs", []

    doc_counts: dict[str, int] = {}
    selected_scores: list[float] = []
    doc_pool = {cit.doc_id for _, cit, _ in candidates}
    diversity_target = DETERMINISTIC_MIN_DISTINCT_DOCS if len(doc_pool) >= DETERMINISTIC_MIN_DISTINCT_DOCS else len(doc_pool)

    # Pass 1: ensure diversity target when available (one sentence per doc).
    for sent, cit, sc in candidates:
        if len(lines) >= target_n or len({c.doc_id for c in citations}) >= diversity_target:
            break
        if doc_counts.get(cit.doc_id, 0) >= 1:
            continue
        line_key = (cit.doc_id, int(cit.page), cit.chunk_id, sent.lower())
        if line_key in used_line_keys:
            continue
        n = len(lines) + 1
        lines.append(f"{n}) {sent} ({cit.doc_id}, p{cit.page}, {cit.chunk_id})")
        citations.append(cit)
        selected_scores.append(float(sc))
        used_line_keys.add(line_key)
        doc_counts[cit.doc_id] = doc_counts.get(cit.doc_id, 0) + 1

    # Pass 2: fill remaining with per-doc cap.
    for sent, cit, sc in candidates:
        if len(lines) >= target_n:
            break
        if doc_counts.get(cit.doc_id, 0) >= DETERMINISTIC_MAX_SENTENCES_PER_DOC:
            continue
        line_key = (cit.doc_id, int(cit.page), cit.chunk_id, sent.lower())
        if line_key in used_line_keys:
            continue
        n = len(lines) + 1
        lines.append(f"{n}) {sent} ({cit.doc_id}, p{cit.page}, {cit.chunk_id})")
        citations.append(cit)
        selected_scores.append(float(sc))
        used_line_keys.add(line_key)
        doc_counts[cit.doc_id] = doc_counts.get(cit.doc_id, 0) + 1

    if len(lines) < 2:
        _fallback_debug(
            case_id=case_id,
            branch="phase_c_renderer_post_selection",
            retrieved_chunks=len(relevant),
            used_sentences=len(lines),
            citations_count=len(citations),
            flags={
                "lines_lt_2_after_selection": True,
                "fallback_before_render": False,
                "fallback_inside_renderer": True,
                "fallback_after_citation_validation": False,
            },
        )
        return "Not found in provided PDFs", []

    distinct_docs = len({c.doc_id for c in citations})
    avg_similarity = (sum(selected_scores) / max(1, len(selected_scores))) if selected_scores else 0.0
    confidence = _deterministic_confidence_label(
        num_sentences=len(lines),
        distinct_docs=distinct_docs,
        avg_similarity=avg_similarity,
    )
    text = "ANSWER:\n" + "\n".join(lines) + f"\nCONFIDENCE: {confidence}"
    return text, citations


def _dedupe_citations_stable(
    citations: list[Citation],
    *,
    max_items: int,
) -> list[Citation]:
    out: list[Citation] = []
    seen: set[tuple[str, int, str]] = set()
    for c in citations:
        key = (str(c.doc_id), int(c.page), str(c.chunk_id))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= int(max_items):
            break
    return out


def _phase_c_debug_sentences(text: str, citations: list[Citation]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not (text or "").startswith("ANSWER:"):
        return out
    body_lines = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s == "ANSWER:" or s.startswith("CONFIDENCE:"):
            continue
        if re.match(r"^\d+\)\s+", s):
            body_lines.append(s)
    for i, cit in enumerate(citations):
        sentence = ""
        if i < len(body_lines):
            sentence = re.sub(r"^\d+\)\s*", "", body_lines[i]).strip()
            sentence = re.sub(r"\s+\([^)]+\)\s*$", "", sentence).strip()
        out.append(
            {
                "sentence": sentence,
                "pdf": cit.doc_id,
                "page": int(cit.page),
                "chunk_id": cit.chunk_id,
            }
        )
    return out


def _use_llm_phases() -> bool:
    raw = (os.getenv("USE_LLM_PHASES", "1") or "").strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}
    return enabled and llm_phase_available()


def answer(
    question: str,
    *,
    case_id: str | None = None,
    scope: str = "MIXED",
    top_k: int = 5,
    data_dir: Path = Path("data"),
    chunk_chars: int = 1000,
    min_chunk_chars: int = 220,
) -> AnswerResult:
    del top_k

    intent = _intent(question)
    default_domain_stats = {
        "domain_key": _domain_key_for_question(question),
        "N_total_sentences": 0,
        "N_requirement_bearing": 0,
        "N_domain_eligible": 0,
        "domain_filter_used": False,
        "sentences_extracted_total": 0,
        "sentences_after_cleaning": 0,
        "sentences_after_dedupe": 0,
        "sentences_after_requirement_gate": 0,
        "sentences_after_domain_sentence_gate": 0,
        "sentences_final_used": 0,
    }

    facts, phase_a_fallback_used, phase_a_trace = _phase_a_retrieval(
        question,
        scope=scope,
        data_dir=data_dir,
        chunk_chars=chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )
    debug_trace: dict[str, object] = {
        "phase_a": phase_a_trace,
        "phase_b": {"relevant_facts_kept": []},
        "phase_c": {
            "answer_sentences": [],
            "citations": [],
            "domain_filter_stats": default_domain_stats,
            "zero_sentence_bug": False,
        },
    }
    phase_a_attempts = list(phase_a_trace.get("attempts", []) or [])
    if phase_a_attempts and bool(phase_a_trace.get("fallback_used", False)):
        phase_a_active_attempt = dict(phase_a_attempts[-1] or {})
    elif phase_a_attempts:
        phase_a_active_attempt = dict(phase_a_attempts[0] or {})
    else:
        phase_a_active_attempt = {}
    phase_a_pool_nonempty = bool(
        int(phase_a_active_attempt.get("pool_size_returned", 0) or 0) > 0
        and int(phase_a_active_attempt.get("chunks_before", 0) or 0) > 0
    )
    if not facts:
        _fallback_debug(
            case_id=case_id,
            branch="phase_a_empty",
            retrieved_chunks=0,
            used_sentences=0,
            citations_count=0,
            flags={
                "empty_pool": True,
                "no_passages": True,
                "no_sentences": True,
                "fallback_before_render": True,
                "fallback_inside_renderer": False,
                "fallback_after_citation_validation": False,
            },
        )
        return AnswerResult(
            text="Not found in provided PDFs",
            intent=intent,
            scope=scope,
            phase_a_fallback_used=phase_a_fallback_used,
            debug_trace=debug_trace,
            citations=[],
            facts=[],
            relevant_facts=[],
        )

    if _requires_external_live_data(question):
        _fallback_debug(
            case_id=case_id,
            branch="external_live_data_block",
            retrieved_chunks=len(facts),
            used_sentences=0,
            citations_count=0,
            flags={
                "scope_blocked": True,
                "requires_external_live_data": True,
                "fallback_before_render": True,
                "fallback_inside_renderer": False,
                "fallback_after_citation_validation": False,
            },
        )
        return AnswerResult(
            text="Not found in provided PDFs",
            intent=intent,
            scope=scope,
            phase_a_fallback_used=phase_a_fallback_used,
            debug_trace=debug_trace,
            citations=[],
            facts=facts,
            relevant_facts=[],
        )

    if _use_llm_phases():
        runner = LLMPhaseRunner()
        relevant = runner.phase_b_filter(question, facts)
    else:
        relevant = _phase_b_filter(question, facts)
    phase_c_input_facts = list(relevant)
    phase_c_precomputed = False
    text = "Not found in provided PDFs"
    citations: list[Citation] = []
    phase_b_trace: dict[str, object] = {
        "relevant_facts_kept": [
            {
                "pdf": f.pdf,
                "page": int(f.page),
                "chunk_id": f.chunk_id,
            }
            for f in relevant
        ]
    }
    debug_trace["phase_b"] = phase_b_trace
    if not relevant:
        phase_b_trace["empty_reason_summary"] = _phase_b_empty_reason_summary(
            question=question,
            phase_a_facts=facts,
            phase_a_trace=phase_a_trace,
        )
        dbg_stats: dict[str, object] = {}
        if case_id in _PHASE_C_DEBUG_IDS:
            selected_dbg = _select_phase_c_facts(facts, case_id=case_id)
            _dbg_chosen, dbg_stats = _phase_c_sentence_candidates(
                question=question,
                selected_facts=selected_dbg,
                case_id=case_id,
            )
            phase_c_dbg = dict(debug_trace.get("phase_c", {}) or {})
            phase_c_dbg["domain_filter_stats"] = dbg_stats
            debug_trace["phase_c"] = phase_c_dbg
        check_stats = dbg_stats if dbg_stats else default_domain_stats
        if phase_a_pool_nonempty and int(check_stats.get("N_total_sentences", 0) or 0) == 0:
            phase_c_trace = dict(debug_trace.get("phase_c", {}) or {})
            phase_c_trace["zero_sentence_bug"] = True
            phase_c_trace["zero_sentence_bug_counts"] = {
                "sentences_extracted_total": int(
                    check_stats.get("sentences_extracted_total", 0) or 0
                ),
                "sentences_after_cleaning": int(
                    check_stats.get("sentences_after_cleaning", 0) or 0
                ),
                "sentences_after_dedupe": int(
                    check_stats.get("sentences_after_dedupe", 0) or 0
                ),
                "sentences_after_requirement_gate": int(
                    check_stats.get("sentences_after_requirement_gate", 0) or 0
                ),
                "sentences_after_domain_sentence_gate": int(
                    check_stats.get("sentences_after_domain_sentence_gate", 0) or 0
                ),
                "sentences_final_used": int(check_stats.get("sentences_final_used", 0) or 0),
            }
            debug_trace["phase_c"] = phase_c_trace
        if _use_llm_phases():
            runner = LLMPhaseRunner()
            phase_b_text, phase_b_citations = runner.phase_c_synthesis(question, facts)
        else:
            phase_b_text, phase_b_citations = _phase_c_synthesis(question, facts, case_id=case_id)
        phase_b_citations = _dedupe_citations_stable(
            phase_b_citations,
            max_items=DETERMINISTIC_PHASE_C_MAX_FACTS,
        )
        phase_b_not_found = (phase_b_text or "").strip() == "Not found in provided PDFs"
        if (not phase_b_not_found) and bool(phase_b_citations):
            phase_b_trace["phase_b_relevant_empty_phase_c_from_facts_used"] = True
            phase_c_input_facts = list(facts)
            text = phase_b_text
            citations = phase_b_citations
            phase_c_precomputed = True
        else:
            _fallback_debug(
                case_id=case_id,
                branch="phase_b_empty_pre_render",
                retrieved_chunks=len(facts),
                used_sentences=int(check_stats.get("sentences_final_used", 0) or 0),
                citations_count=len(phase_b_citations),
                flags={
                    "no_passages": True,
                    "phase_b_relevant_empty": True,
                    "phase_a_pool_nonempty": bool(phase_a_pool_nonempty),
                    "phase_c_from_facts_not_found": phase_b_not_found,
                    "phase_c_from_facts_citations_empty": len(phase_b_citations) == 0,
                    "validator_failed": False,
                    "fallback_before_render": True,
                    "fallback_inside_renderer": False,
                    "fallback_after_citation_validation": False,
                },
            )
            return AnswerResult(
                text="Not found in provided PDFs",
                intent=intent,
                scope=scope,
                phase_a_fallback_used=phase_a_fallback_used,
                debug_trace=debug_trace,
                citations=[],
                facts=facts,
                relevant_facts=[],
            )

    if not phase_c_precomputed:
        phase_c_input_facts = list(relevant)
        if _use_llm_phases():
            runner = LLMPhaseRunner()
            text, citations = runner.phase_c_synthesis(question, phase_c_input_facts)
        else:
            text, citations = _phase_c_synthesis(question, phase_c_input_facts, case_id=case_id)
    citations = _dedupe_citations_stable(
        citations,
        max_items=DETERMINISTIC_PHASE_C_MAX_FACTS,
    )
    rendered_used_sentences = len(_phase_c_debug_sentences(text, citations))
    if (text or "").strip() != "Not found in provided PDFs" and not citations:
        _fallback_debug(
            case_id=case_id,
            branch="phase_c_uncited_after_render",
            retrieved_chunks=len(phase_c_input_facts),
            used_sentences=rendered_used_sentences,
            citations_count=0,
            flags={
                "uncited_answer": True,
                "fallback_before_render": False,
                "fallback_inside_renderer": False,
                "fallback_after_citation_validation": True,
            },
        )
        text = "Not found in provided PDFs"
    if (text or "").strip() == "Not found in provided PDFs":
        _fallback_debug(
            case_id=case_id,
            branch="phase_c_not_found_after_render",
            retrieved_chunks=len(phase_c_input_facts),
            used_sentences=rendered_used_sentences,
            citations_count=len(citations),
            flags={
                "renderer_returned_not_found": True,
                "validator_failed": False,
                "fallback_before_render": False,
                "fallback_inside_renderer": True,
                "fallback_after_citation_validation": False,
            },
        )
    _, phase_c_domain_stats = _phase_c_sentence_candidates(
        question=question,
        selected_facts=_select_phase_c_facts(phase_c_input_facts, case_id=case_id),
        case_id=None,
    )
    attempts = list(phase_a_trace.get("attempts", []) or [])
    if attempts and bool(phase_a_trace.get("fallback_used", False)):
        active_attempt = dict(attempts[-1] or {})
    elif attempts:
        active_attempt = dict(attempts[0] or {})
    else:
        active_attempt = {}
    pool_size_returned = int(active_attempt.get("pool_size_returned", 0) or 0)
    chunks_before = int(active_attempt.get("chunks_before", 0) or 0)
    n_total_sentences = int(phase_c_domain_stats.get("N_total_sentences", 0) or 0)
    zero_sentence_bug = bool(
        pool_size_returned > 0 and chunks_before > 0 and n_total_sentences == 0
    )

    phase_c_trace: dict[str, object] = {
        "answer_sentences": _phase_c_debug_sentences(text, citations),
        "citations": [
            {
                "pdf": c.doc_id,
                "page": int(c.page),
                "chunk_id": c.chunk_id,
            }
            for c in citations
        ],
        "domain_filter_stats": phase_c_domain_stats,
        "zero_sentence_bug": zero_sentence_bug,
    }
    if zero_sentence_bug:
        phase_c_trace["zero_sentence_bug_counts"] = {
            "sentences_extracted_total": int(
                phase_c_domain_stats.get("sentences_extracted_total", 0) or 0
            ),
            "sentences_after_cleaning": int(
                phase_c_domain_stats.get("sentences_after_cleaning", 0) or 0
            ),
            "sentences_after_requirement_gate": int(
                phase_c_domain_stats.get("sentences_after_requirement_gate", 0) or 0
            ),
            "sentences_after_domain_sentence_gate": int(
                phase_c_domain_stats.get("sentences_after_domain_sentence_gate", 0) or 0
            ),
        }
    debug_trace["phase_c"] = phase_c_trace

    return AnswerResult(
        text=text,
        intent=intent,
        scope=scope,
        phase_a_fallback_used=phase_a_fallback_used,
        debug_trace=debug_trace,
        citations=citations,
        facts=facts,
        relevant_facts=phase_c_input_facts,
    )
