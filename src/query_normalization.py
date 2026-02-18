from __future__ import annotations

import re

SYNONYMS: dict[str, str] = {
    "continuous processing": "continuous manufacturing",
}

ACRONYMS: dict[str, str] = {
    "gmp": "good manufacturing practice",
}


def normalize_text(q: str) -> str:
    s = (q or "").lower()
    s = " ".join(s.split())
    # Collapse duplicated punctuation marks (e.g., "??" -> "?").
    s = re.sub(r"([?!.,;:])\1+", r"\1", s)
    return s.strip()


def expand_query(q_norm: str) -> list[str]:
    base = normalize_text(q_norm)
    out = [base]
    if not base:
        return out

    max_expansions = 4
    seen = {base}

    def _add(candidate: str) -> None:
        nonlocal out
        c = normalize_text(candidate)
        if not c or c in seen:
            return
        if (len(out) - 1) >= max_expansions:
            return
        out.append(c)
        seen.add(c)

    for src, dst in SYNONYMS.items():
        if src in base:
            _add(base.replace(src, dst))

    for token, expansion in ACRONYMS.items():
        if re.search(rf"\b{re.escape(token)}\b", base):
            _add(re.sub(rf"\b{re.escape(token)}\b", expansion, base))

    return out

