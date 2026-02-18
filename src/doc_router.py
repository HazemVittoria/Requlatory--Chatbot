from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import authority_from_folder
from .query_normalization import normalize_text
from . import search as search_module
from .topic_taxonomy import TOPICS

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_GMP_VALIDATION_DOC_HINTS = (
    "annex15",
    "validation",
    "qualification",
    "processvalidation",
    "211",
)


def _build_relpath_map(data_dir: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    if not data_dir.exists():
        return out
    for folder in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        authority = authority_from_folder(folder.name)
        for pdf in sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower()):
            key = (authority, pdf.name)
            if key not in out:
                out[key] = str(pdf.relative_to(data_dir)).replace("\\", "/")
    return out


def _doc_profiles(data_dir: Path) -> list[dict[str, str]]:
    search_module._ensure_index(data_dir=data_dir, chunk_chars=1000, min_chunk_chars=220)
    corpus = getattr(search_module, "_CORPUS", None) or []
    rel_map = _build_relpath_map(data_dir)

    first_chunk_by_key: dict[tuple[str, str], str] = {}
    for c in corpus:
        authority = str(c.get("authority") or "OTHER")
        file_name = str(c.get("file") or "")
        if not file_name:
            continue
        key = (authority, file_name)
        if key not in first_chunk_by_key:
            first_chunk_by_key[key] = str(c.get("text") or "")

    rows: list[dict[str, str]] = []
    for key in sorted(first_chunk_by_key, key=lambda k: ((rel_map.get(k) or k[1]).lower(), k[0])):
        authority, file_name = key
        rel = rel_map.get(key, file_name)
        chunk = first_chunk_by_key.get(key, "")
        rows.append(
            {
                "doc_id": rel,
                "authority": authority,
                "file_name": file_name,
                "profile_text": f"{Path(file_name).stem} {chunk[:500]}",
            }
        )
    return rows


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 3}


def _path_match(doc_id: str, path_rule: str) -> bool:
    d = (doc_id or "").replace("\\", "/").lower()
    p = (path_rule or "").replace("\\", "/").lower().strip()
    if not d or not p:
        return False
    if d.startswith(p):
        return True
    if p.endswith("/"):
        return d.startswith(p)
    return p in d


def _match_topics(queries: list[str]) -> list[str]:
    q_text = " ".join(normalize_text(q) for q in (queries or []) if normalize_text(q))
    if not q_text:
        return []
    scores: dict[str, int] = {}
    for topic_id, conf in TOPICS.items():
        keywords = [normalize_text(k) for k in conf.get("keywords", [])]
        score = 0
        for kw in keywords:
            if kw and kw in q_text:
                score += 1
        if score > 0:
            scores[topic_id] = score
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def match_topics(queries: list[str]) -> list[str]:
    return _match_topics(queries)


def _filename_overlap_signal(query_text: str, doc_id: str) -> float:
    q_tokens = _tokens(query_text)
    d_tokens = _tokens(doc_id)
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / max(1, len(q_tokens))


def route_docs(queries: list[str], *, top_docs: int = 5) -> list[str]:
    q_list = [normalize_text(q) for q in (queries or []) if normalize_text(q)]
    if not q_list:
        return []

    profiles = _doc_profiles(Path("data"))
    if not profiles:
        return []

    docs = [str(p["doc_id"]) for p in profiles]
    texts = [p["profile_text"] for p in profiles]

    query_text = " ".join(q_list)

    matched_topics = _match_topics(q_list)
    validation_topics = {"process_validation", "qualification_validation"}
    gmp_validation_routing = (
        bool(validation_topics & set(matched_topics))
        and bool(re.search(r"\bgmp\b", query_text))
    )
    routing_top_docs = max(int(top_docs), 8) if gmp_validation_routing else int(top_docs)
    topic_paths: list[str] = []
    for t in matched_topics:
        topic_paths.extend(TOPICS.get(t, {}).get("paths", []))

    tfidf_scores = np.zeros(len(profiles), dtype=float)
    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        x_docs = vect.fit_transform(texts)
        if x_docs.shape[0] > 0:
            q_mat = vect.transform(q_list)
            sim = cosine_similarity(q_mat, x_docs)
            if sim.size:
                tfidf_scores = np.max(sim, axis=0)
    except Exception:
        tfidf_scores = np.zeros(len(profiles), dtype=float)

    if not len(tfidf_scores):
        return []

    tfidf_order = np.argsort(tfidf_scores)[::-1]
    tfidf_top_n = max(1, routing_top_docs)
    tfidf_top_docs = {docs[int(ix)] for ix in tfidf_order[:tfidf_top_n] if float(tfidf_scores[int(ix)]) > 0.0}

    topic_docs: set[str] = set()
    if topic_paths:
        for d in docs:
            if any(_path_match(d, p) for p in topic_paths):
                topic_docs.add(d)

    if topic_docs:
        candidate_docs = topic_docs | tfidf_top_docs
    else:
        candidate_docs = tfidf_top_docs

    if not candidate_docs:
        candidate_docs = {docs[int(ix)] for ix in tfidf_order[:tfidf_top_n]}
    if not candidate_docs:
        return []

    candidate_idx = [i for i, d in enumerate(docs) if d in candidate_docs]
    scored: list[tuple[float, str]] = []
    for i in candidate_idx:
        d = docs[i]
        name_sig = _filename_overlap_signal(query_text, d)
        topic_hits = 0
        if topic_paths:
            topic_hits = sum(1 for p in topic_paths if _path_match(d, p))
        score = (0.80 * float(tfidf_scores[i])) + (0.20 * float(name_sig)) + (0.15 * float(topic_hits))
        if gmp_validation_routing:
            d_norm = re.sub(r"[^a-z0-9]+", "", d.lower())
            hint_hits = sum(1 for h in _GMP_VALIDATION_DOC_HINTS if h in d_norm)
            score += 0.12 * float(hint_hits)
        scored.append((score, d))

    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    out: list[str] = []
    for score, d in scored:
        if len(out) >= max(1, routing_top_docs):
            break
        if score <= 0.0 and len(out) > 0:
            break
        out.append(d)
    return out
