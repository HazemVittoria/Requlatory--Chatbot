from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Callable

from .llm_client import DeterminismSettings, DeterministicLLMClient
from .query_normalization import normalize_text
from .qa_types import Citation, Fact


def llm_phase_available() -> bool:
    try:
        import openai  # noqa: F401
    except Exception:
        return False
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _extract_json_object(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty LLM output")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    obj = json.loads(s[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


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

_BOILERPLATE_PATTERNS = (
    "introduction",
    "purpose",
    "recommendations",
    "scope of this guidance",
    "this guidance represents",
    "table of contents",
    "guidance for industry",
)

_GENERIC_GMP_VALIDATION_TOPIC_KEYWORDS = (
    "validation",
    "qualification",
    "protocol",
    "plan",
    "acceptance criteria",
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

_VALIDATION_PRIMARY_TOKENS = (
    "validation",
    "validate",
    "validated",
    "process validation",
)

_VALIDATION_SECONDARY_TOKENS = (
    "qualification",
    "qualified",
    "iq",
    "oq",
    "pq",
    "acceptance criteria",
    "critical process parameter",
    "critical quality attribute",
    "continued process verification",
    "ongoing verification",
    "continued verification",
)

_VALIDATION_NEGATIVE_PATTERNS = (
    "email subject line",
    "co-operation between manufacturers and sponsors",
    "investigational medicinal products",
    "shall be deemed to be adulterated",
)

_TOPIC_STOP = {
    "what",
    "does",
    "how",
    "for",
    "the",
    "and",
    "with",
    "under",
    "according",
    "about",
    "require",
    "requires",
    "required",
    "requirement",
    "gmp",
}

_PHASE_C_TITLE_PATTERNS = (
    "introduction",
    "purpose",
    "scope",
    "recommendations",
    "guidance for industry",
    "table of contents",
)


def _has_requirement_language(quote: str) -> bool:
    q = " ".join((quote or "").lower().split())
    if not q:
        return False
    for t in _REQ_TERMS:
        if re.search(rf"\b{re.escape(t)}\w*\b", q):
            return True
    return False


def _is_boilerplate_quote(quote: str) -> bool:
    q = " ".join((quote or "").lower().split())
    if not q:
        return True
    for p in _BOILERPLATE_PATTERNS:
        if q.startswith(p) or p in q:
            return True
    if len(q.split()) <= 12 and ("guidance" in q or "guideline" in q) and not _has_requirement_language(q):
        return True
    return False


def _topic_tokens(question: str) -> set[str]:
    q = normalize_text(question or "")
    toks = set(re.findall(r"[a-z0-9]{3,}", q))
    return {t for t in toks if t not in _TOPIC_STOP}


def _contains_topic_keywords(quote: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    q = " ".join((quote or "").lower().split())
    return any(re.search(rf"\b{re.escape(t)}\b", q) for t in tokens)


def _is_generic_gmp_validation_query(question: str) -> bool:
    q = normalize_text(question or "")
    if not q:
        return False
    has_gmp = bool(re.search(r"\bgmp\b", q))
    has_validation = bool(re.search(r"\b(validation|qualification)\b", q))
    return has_gmp and has_validation


def _is_broad_requirement_intent(question: str) -> bool:
    q = normalize_text(question or "")
    if not q:
        return False
    if not re.search(r"\bgmp\b", q):
        return False
    if not any(re.search(rf"\b{re.escape(term)}\b", q) for term in _BROAD_REQUIREMENT_INCLUDE_TERMS):
        return False
    if any(p in q for p in _BROAD_REQUIREMENT_EXCLUDE_PHRASES):
        return False
    return True


def _is_validation_topic_gate_active(question: str) -> bool:
    q = normalize_text(question or "")
    if not q:
        return False
    return _is_broad_requirement_intent(q) and ("validation" in q or "validate" in q)


def _bypass_validation_negative_filter(question: str) -> bool:
    q = normalize_text(question or "")
    if not q:
        return False
    if re.search(r"\bimp\b", q):
        return True
    if "investigational medicinal products" in q:
        return True
    if "adulteration" in q or "adulterated" in q:
        return True
    return False


def _validation_topic_coherent(quote: str, *, question: str) -> bool:
    q = " ".join((quote or "").lower().split())
    if not q:
        return False

    if not _bypass_validation_negative_filter(question):
        for pat in _VALIDATION_NEGATIVE_PATTERNS:
            if pat in q:
                return False

    if any(tok in q for tok in _VALIDATION_PRIMARY_TOKENS):
        return True

    secondary_hits = sum(1 for tok in _VALIDATION_SECONDARY_TOKENS if tok in q)
    return secondary_hits >= 2


def _matches_generic_gmp_validation_topic(quote: str) -> bool:
    q = " ".join((quote or "").lower().split())
    if not q:
        return False
    for kw in _GENERIC_GMP_VALIDATION_TOPIC_KEYWORDS:
        if " " in kw:
            if kw in q:
                return True
            continue
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return True
    return False


def _enforce_phase_b_requirement_filter(
    question: str,
    facts: list[Fact],
    *,
    source_facts: list[Fact] | None = None,
) -> list[Fact]:
    validation_gate = _is_validation_topic_gate_active(question)

    # Keep requirement-bearing, non-boilerplate facts.
    filtered = [f for f in facts if not _is_boilerplate_quote(f.quote) and _has_requirement_language(f.quote)]
    if validation_gate:
        filtered = [f for f in filtered if _validation_topic_coherent(f.quote, question=question)]

    # For broad GMP validation/qualification questions, ensure at least two sources
    # when Phase A already contains requirement-bearing facts across documents.
    if filtered and validation_gate:
        docs = {f.pdf for f in filtered}
        if len(docs) == 1:
            selected_ids = {(f.pdf, int(f.pdf_index), f.chunk_id) for f in filtered}
            pool = list(source_facts or facts)
            candidates = [
                f
                for f in pool
                if (f.pdf, int(f.pdf_index), f.chunk_id) not in selected_ids
                and f.pdf not in docs
                and not _is_boilerplate_quote(f.quote)
                and _has_requirement_language(f.quote)
                and _validation_topic_coherent(f.quote, question=question)
            ]
            if candidates:
                extra = max(candidates, key=lambda f: float(f.score))
                filtered = [*filtered, extra]

    if filtered:
        return filtered

    if validation_gate:
        pool = list(source_facts or facts)
        pool = [
            f
            for f in pool
            if not _is_boilerplate_quote(f.quote)
            and _has_requirement_language(f.quote)
            and _validation_topic_coherent(f.quote, question=question)
        ]
        if not pool:
            return []
        best = max(pool, key=lambda f: float(f.score))
        return [best]

    # Minimal-utility fallback: keep one strongest topic-matching non-boilerplate fact.
    topic = _topic_tokens(question)
    pool = list(source_facts or facts)
    pool = [f for f in pool if not _is_boilerplate_quote(f.quote)]
    if topic:
        pool = [f for f in pool if _contains_topic_keywords(f.quote, topic)]
    if not pool:
        return []
    best = max(pool, key=lambda f: float(f.score))
    return [best]


def _compact_text(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _looks_like_heading(sentence: str) -> bool:
    s = " ".join((sentence or "").split()).strip()
    if not s:
        return True
    if s.endswith(":"):
        return True
    if len(s.split()) <= 3:
        return True
    sl = s.lower()
    if any(sl.startswith(p) for p in _PHASE_C_TITLE_PATTERNS):
        return True
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
        if len(letters) >= 6 and upper_ratio >= 0.85:
            return True
    return False


def _is_quote_dump(sentence: str, source_quote: str, threshold: float = 0.80) -> bool:
    a = _compact_text(sentence)
    b = _compact_text(source_quote)
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio > float(threshold)


def _question_keywords(question: str) -> set[str]:
    q = normalize_text(question or "")
    toks = set(re.findall(r"[a-z0-9]{3,}", q))
    return {t for t in toks if t not in _TOPIC_STOP}


def _citation_label(doc_id: str, doc_title: str) -> str:
    title = str(doc_title or "").strip()
    if title:
        return title
    return str(doc_id or "").strip()


class LLMPhaseRunner:
    def __init__(
        self,
        model: str | None = None,
        request_fn: Callable[..., Any] | None = None,
    ):
        self.model = (model or os.getenv("LLM_MODEL", "gpt-4.1-mini")).strip()
        self._request_fn = request_fn or self._build_openai_request_fn()
        self._primary = DeterministicLLMClient(
            self._request_fn,
            settings=DeterminismSettings(),
            allow_param_overrides=False,
        )
        self._fallback = DeterministicLLMClient(
            self._request_fn,
            settings=DeterminismSettings(force_fallback=True),
            allow_param_overrides=False,
        )

    @staticmethod
    def _build_openai_request_fn() -> Callable[..., Any]:
        from openai import OpenAI

        client = OpenAI()

        def _request(**kwargs: Any) -> str:
            resp = client.chat.completions.create(**kwargs)
            try:
                return str(resp.choices[0].message.content or "")
            except Exception:
                return ""

        return _request

    def _call_json(self, *, messages: list[dict[str, str]]) -> dict[str, Any]:
        common = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        first_err: Exception | None = None
        for client in (self._primary, self._fallback):
            try:
                raw = client.request(**common)
                return _extract_json_object(str(raw))
            except Exception as e:
                first_err = first_err or e
                continue
        if first_err is not None:
            raise first_err
        raise RuntimeError("LLM call failed without explicit error")

    def phase_b_filter(self, question: str, facts: list[Fact]) -> list[Fact]:
        if not facts:
            return []

        fact_payload = [
            {
                "quote": f.quote,
                "pdf": f.pdf,
                "page": int(f.page),
                "chunk_id": f.chunk_id,
                "score": float(f.score),
            }
            for f in facts
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are PHASE B fact-filtering. Keep only facts directly useful to answer the question. "
                    "Deduplicate overlap. Keep requirement-bearing facts (shall/must/should/required/etc). "
                    "Drop boilerplate such as introduction/purpose/scope/title lines. "
                    "Return ONLY JSON in schema: "
                    '{"relevant_facts":[{"quote":"...","pdf":"...","page":1,"chunk_id":"..."}]}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "facts": fact_payload}, ensure_ascii=True),
            },
        ]
        obj = self._call_json(messages=messages)
        return _validate_relevant_facts(obj=obj, source_facts=facts, question=question)

    def phase_c_synthesis(self, question: str, relevant_facts: list[Fact]) -> tuple[str, list[Citation]]:
        if not relevant_facts:
            return "Not found in provided PDFs", []

        rel_payload = [
            {
                "quote": f.quote,
                "pdf": f.pdf,
                "page": int(f.page),
                "chunk_id": f.chunk_id,
            }
            for f in relevant_facts
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are PHASE C answer synthesis. Use ONLY provided relevant facts. "
                    "Every sentence must cite one fact. Max 6 sentences. "
                    "Each sentence must be a concise paraphrase (not copied quote/title) and <= 30 words. "
                    "If a sentence cannot be cited or violates constraints, exclude it. "
                    "Return ONLY JSON in schema: "
                    '{"answer_sentences":[{"sentence":"...","pdf":"...","page":1,"chunk_id":"..."}],'
                    '"confidence":"High|Medium|Low"}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "relevant_facts": rel_payload}, ensure_ascii=True),
            },
        ]
        obj = self._call_json(messages=messages)
        return _validate_synthesis(obj=obj, relevant_facts=relevant_facts, question=question)


def _validate_relevant_facts(obj: dict[str, Any], source_facts: list[Fact], question: str = "") -> list[Fact]:
    rows = obj.get("relevant_facts", [])
    if not isinstance(rows, list):
        raise ValueError("phase_b invalid schema: relevant_facts must be a list")

    source_index = {(f.pdf, int(f.pdf_index), f.chunk_id): f for f in source_facts}
    out: list[Fact] = []
    seen: set[tuple[str, int, str]] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        pdf = str(row.get("pdf") or "")
        chunk_id = str(row.get("chunk_id") or "")
        try:
            page = int(row.get("page"))
        except Exception:
            continue
        pdf_index = max(0, page - 1)
        key = (pdf, pdf_index, chunk_id)
        if key in seen or key not in source_index:
            continue
        src = source_index[key]
        quote = str(row.get("quote") or src.quote).strip() or src.quote
        out.append(
            Fact(
                quote=quote,
                pdf=pdf,
                page=int(src.pdf_index) + 1,
                chunk_id=chunk_id,
                score=float(src.score),
                doc_title=src.doc_title,
                pdf_index=int(src.pdf_index),
                page_label=src.page_label,
            )
        )
        seen.add(key)
    return _enforce_phase_b_requirement_filter(question=question, facts=out, source_facts=source_facts)


def _normalize_confidence(v: str) -> str:
    s = (v or "").strip().lower()
    if s == "high":
        return "High"
    if s == "medium":
        return "Medium"
    if s == "low":
        return "Low"
    return "Low"


def _validate_synthesis(
    obj: dict[str, Any],
    relevant_facts: list[Fact],
    question: str = "",
) -> tuple[str, list[Citation]]:
    rows = obj.get("answer_sentences", [])
    if not isinstance(rows, list):
        raise ValueError("phase_c invalid schema: answer_sentences must be a list")

    rel_index = {(f.pdf, int(f.pdf_index), f.chunk_id): f for f in relevant_facts}
    lines: list[str] = []
    cits: list[Citation] = []
    used: set[tuple[str, int, str]] = set()

    for row in rows:
        if len(lines) >= 6:
            break
        if not isinstance(row, dict):
            continue
        pdf = str(row.get("pdf") or "")
        chunk_id = str(row.get("chunk_id") or "")
        try:
            page = int(row.get("page"))
        except Exception:
            continue
        sentence = " ".join(str(row.get("sentence") or "").split()).strip()
        if not sentence:
            continue
        if len(sentence.split()) > 30:
            continue
        if _looks_like_heading(sentence):
            continue
        pdf_index = max(0, page - 1)
        key = (pdf, pdf_index, chunk_id)
        if key in used or key not in rel_index:
            continue
        src = rel_index[key]
        src_quote = src.quote
        if _is_quote_dump(sentence, src_quote):
            continue
        lines.append(
            f"{len(lines)+1}) {sentence} ({_citation_label(pdf, src.doc_title)} | {src.display_page} | {chunk_id})"
        )
        cits.append(
            Citation(
                doc_id=pdf,
                page=int(src.pdf_index) + 1,
                chunk_id=chunk_id,
                doc_title=src.doc_title,
                pdf_index=int(src.pdf_index),
                page_label=src.page_label,
            )
        )
        used.add(key)

    if not lines:
        return "Not found in provided PDFs", []

    content = " ".join(
        re.sub(r"\s+\([^)]+\)\s*$", "", re.sub(r"^\d+\)\s*", "", ln).strip())
        for ln in lines
    ).lower()
    keys = _question_keywords(question)
    required_hits = 2 if len(keys) >= 2 else len(keys)
    if required_hits > 0:
        hits = sum(1 for k in keys if re.search(rf"\b{re.escape(k)}\b", content))
        if hits < required_hits:
            return "Not found in provided PDFs", []

    conf = _normalize_confidence(str(obj.get("confidence") or "Low"))
    text = "ANSWER:\n" + "\n".join(lines) + f"\nCONFIDENCE: {conf}"
    return text, cits
