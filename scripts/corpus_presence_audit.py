from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import search as search_module


DEVIATION_TOKENS = [
    "deviation",
    "deviations",
    "nonconformance",
    "non-conformance",
    "investigation",
    "root cause",
    "impact assessment",
    "capa",
    "corrective action",
    "preventive action",
]

EQUIPMENT_QUAL_TOKENS = [
    "equipment qualification",
    "qualification protocol",
    "qualification report",
    "installation qualification",
    "operational qualification",
    "performance qualification",
    "iq",
    "oq",
    "pq",
    "acceptance criteria",
    "commissioning",
    "verification",
]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def main() -> int:
    search_module._ensure_index(data_dir=Path("data"), chunk_chars=1000, min_chunk_chars=220)
    corpus = getattr(search_module, "_CORPUS", None) or []

    print(f"total_chunk_count_scanned: {len(corpus)}")

    all_tokens = [
        ("deviations", t) for t in DEVIATION_TOKENS
    ] + [
        ("equipment_qualification", t) for t in EQUIPMENT_QUAL_TOKENS
    ]

    for group, token in all_tokens:
        tok = _norm(token)
        hits: list[tuple[str, int, str]] = []
        for c in corpus:
            text = _norm(str(c.get("text") or ""))
            if tok and tok in text:
                hits.append(
                    (
                        str(c.get("file") or ""),
                        int(c.get("page") or 0),
                        str(c.get("chunk_id") or ""),
                    )
                )
        print()
        print(f"{group} | token: {token}")
        print(f"chunk_count: {len(hits)}")
        print("examples_top5:")
        for file_name, page, chunk_id in hits[:5]:
            print(f"- {file_name} | {page} | {chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
