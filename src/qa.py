from __future__ import annotations

import os
import re
from functools import cmp_to_key
from pathlib import Path

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
from .search import search_chunks

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

RETRIEVAL_TOP_K = 8
MAX_CONTEXT_CHUNKS = 6
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


def _retrieve_chunks_for_queries(
    queries: list[str],
    *,
    scope: str,
    data_dir: Path,
    chunk_chars: int,
    min_chunk_chars: int,
    min_similarity: float,
    allowed_docs: list[str] | None = None,
) -> list[dict]:
    merged: dict[tuple[str, int, str], dict] = {}
    for q in queries:
        for c in search_chunks(
            q,
            data_dir=data_dir,
            scope=scope,
            top_k=RETRIEVAL_TOP_K,
            max_context_chunks=MAX_CONTEXT_CHUNKS,
            min_similarity=min_similarity,
            use_mmr=USE_MMR,
            mmr_lambda=MMR_LAMBDA,
            allowed_docs=allowed_docs,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
        ):
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
    queries = _phase_a_query_variants(question)
    candidate_docs = route_docs(queries, top_docs=5)
    allowed_docs = candidate_docs if candidate_docs else None
    attempts: list[dict[str, object]] = []
    chunks = _retrieve_chunks_for_queries(
        queries,
        data_dir=data_dir,
        scope=scope,
        min_similarity=MIN_SIMILARITY,
        chunk_chars=chunk_chars,
        min_chunk_chars=min_chunk_chars,
        allowed_docs=allowed_docs,
    )
    attempts.append(
        {
            "min_similarity": float(MIN_SIMILARITY),
            "top_results": _phase_a_chunk_preview(chunks),
        }
    )
    if not chunks and FALLBACK_ONLY_IF_EMPTY:
        fallback_chunks = _retrieve_chunks_for_queries(
            queries,
            data_dir=data_dir,
            scope=scope,
            min_similarity=FALLBACK_MIN_SIMILARITY,
            chunk_chars=chunk_chars,
            min_chunk_chars=min_chunk_chars,
            allowed_docs=allowed_docs,
        )
        attempts.append(
            {
                "min_similarity": float(FALLBACK_MIN_SIMILARITY),
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
        "candidate_docs": candidate_docs,
        "thresholds_attempted": [float(a["min_similarity"]) for a in attempts],
        "attempts": attempts,
        "fallback_used": bool(fallback_used),
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
    candidates = _split_complete_sentences(quote)
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda s: (
            0 if _has_requirement_language(s) else 1,
            len(s.split()),
        ),
    )

    out: list[str] = []
    seen: set[str] = set()
    for s in ordered:
        trimmed = _trim_sentence_words(s, max_words=30)
        if not trimmed:
            continue
        normalized = _normalize_sentence_casing(trimmed)
        if not normalized:
            continue
        if _looks_like_heading_text(normalized):
            continue
        if _looks_truncated_ending(normalized):
            continue
        if normalized[-1] not in ".;:":
            continue
        k = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _select_phase_c_facts(relevant: list[Fact]) -> list[Fact]:
    if not relevant:
        return []
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


def _phase_c_synthesis(question: str, relevant: list[Fact]) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []
    used_line_keys: set[tuple[str, int, str, str]] = set()
    selected = _select_phase_c_facts(relevant)
    candidates: list[tuple[str, Citation, float]] = []
    per_fact_limit = 2 if len(selected) > 1 else 3
    for f in selected:
        cit = Citation(doc_id=f.pdf, page=f.page, chunk_id=f.chunk_id)
        for sent in _deterministic_sentences_from_fact(f, max_items=per_fact_limit):
            candidates.append((sent, cit, float(f.score)))
            if len(candidates) >= DETERMINISTIC_PHASE_C_MAX_FACTS:
                break
        if len(candidates) >= DETERMINISTIC_PHASE_C_MAX_FACTS:
            break

    if len(candidates) < 2:
        return "Not found in provided PDFs", []

    candidates_by_doc: dict[str, int] = {}
    for _, cit, _ in candidates:
        candidates_by_doc[cit.doc_id] = candidates_by_doc.get(cit.doc_id, 0) + 1
    possible_max = sum(min(DETERMINISTIC_MAX_SENTENCES_PER_DOC, n) for n in candidates_by_doc.values())
    target_n = min(DETERMINISTIC_PHASE_C_MAX_FACTS, len(candidates), possible_max)
    if target_n < 2:
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
    scope: str = "MIXED",
    top_k: int = 5,
    data_dir: Path = Path("data"),
    chunk_chars: int = 1000,
    min_chunk_chars: int = 220,
) -> AnswerResult:
    del top_k

    intent = _intent(question)

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
        "phase_c": {"answer_sentences": [], "citations": []},
    }
    if not facts:
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
        text, citations = runner.phase_c_synthesis(question, relevant)
    else:
        text, citations = _phase_c_synthesis(question, relevant)
    debug_trace["phase_c"] = {
        "answer_sentences": _phase_c_debug_sentences(text, citations),
        "citations": [
            {
                "pdf": c.doc_id,
                "page": int(c.page),
                "chunk_id": c.chunk_id,
            }
            for c in citations
        ],
    }

    return AnswerResult(
        text=text,
        intent=intent,
        scope=scope,
        phase_a_fallback_used=phase_a_fallback_used,
        debug_trace=debug_trace,
        citations=citations,
        facts=facts,
        relevant_facts=relevant,
    )
