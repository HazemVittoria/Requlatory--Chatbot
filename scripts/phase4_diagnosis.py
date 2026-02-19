from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import median
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qa import answer

RESULTS_PATH = Path("eval/results_baseline.csv")
VERDICTS_PATH = Path("eval/verdicts_baseline.csv")

DOMAIN_ORDER = [
    "validation",
    "deviations",
    "computerized_systems",
    "data_integrity",
    "apr",
    "equipment_qualification",
]

FRAMEWORK_SUBSTRINGS = [
    "stages",
    "lifecycle",
    "what documentation",
    "what controls",
    "what is required in",
    "what is validation",
    "what is process validation",
    "what is computer system validation",
    "what is csv",
    "what is an out-of-specification",
    "what is oos",
]

MIN_DOMAIN_SENTENCES = 2
OUTPUTS_DIR = Path("eval/outputs")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _parse_bool(value: str) -> bool:
    s = (value or "").strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _parse_int(value: str) -> int:
    try:
        return int(float((value or "").strip()))
    except Exception:
        return 0


def _fmt_median(values: list[int]) -> str:
    if not values:
        return "0"
    m = median(values)
    if float(m).is_integer():
        return str(int(m))
    return str(float(m))


def _is_framework_question(question: str) -> bool:
    q = (question or "").lower()
    for needle in FRAMEWORK_SUBSTRINGS:
        if needle in q:
            return True
    return False


def _is_domain_key_missing(domain_key: str) -> bool:
    key = (domain_key or "").strip()
    return key == "" or key.lower() == "none"


def _short_cited_sources(cited_sources: str, max_items: int = 2) -> str:
    items = [x.strip() for x in (cited_sources or "").split(";") if x.strip()]
    if not items:
        return ""
    if len(items) <= max_items:
        return ";".join(items)
    return ";".join(items[:max_items]) + ";..."


def _extract_answer_sentences_from_output(case_id: str, max_items: int = 5) -> list[str]:
    path = OUTPUTS_DIR / f"{case_id}.txt"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    in_answer = False
    out: list[str] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue
        s_upper = s.upper()
        if s_upper == "ANSWER:":
            in_answer = True
            continue
        if in_answer and (s_upper.startswith("CONFIDENCE:") or s_upper == "CITATIONS:"):
            break
        if not in_answer:
            continue
        m = re.match(r"^\d+\)\s*(.+)$", s)
        if m:
            out.append(m.group(1).strip())
        elif s:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))  # type: ignore[arg-type]
        except Exception:
            return default


def _active_phase_a_attempt(phase_a: dict[str, object]) -> dict[str, object]:
    attempts = list(phase_a.get("attempts", []) or [])
    if not attempts:
        return {}
    if bool(phase_a.get("fallback_used", False)):
        return dict(attempts[-1] or {})
    return dict(attempts[0] or {})


def _chunk_filter_diag_for_question(question: str) -> dict[str, object]:
    defaults: dict[str, object] = {
        "domain_chunk_filter_level_used": 0,
        "top_k_used": 0,
        "pool_size_returned": 0,
        "allowed_docs_relaxed_retry_used": False,
        "chunks_before": 0,
        "chunks_after_level1": 0,
        "chunks_after_level2": 0,
        "fallback_used": False,
    }
    try:
        res = answer(question)
    except Exception:
        return defaults
    trace = dict(getattr(res, "debug_trace", {}) or {})
    phase_a = dict(trace.get("phase_a", {}) or {})
    active = _active_phase_a_attempt(phase_a)
    return {
        "domain_chunk_filter_level_used": _safe_int(
            active.get("domain_chunk_filter_level_used", 0), default=0
        ),
        "top_k_used": _safe_int(active.get("top_k_used", 0), default=0),
        "pool_size_returned": _safe_int(active.get("pool_size_returned", 0), default=0),
        "allowed_docs_relaxed_retry_used": bool(
            active.get("allowed_docs_relaxed_retry_used", False)
        ),
        "chunks_before": _safe_int(active.get("chunks_before", 0), default=0),
        "chunks_after_level1": _safe_int(active.get("chunks_after_level1_strict", 0), default=0),
        "chunks_after_level2": _safe_int(active.get("chunks_after_level2_relaxed", 0), default=0),
        "fallback_used": bool(active.get("fallback_used", False)),
    }


def main() -> int:
    results_rows = _read_csv_rows(RESULTS_PATH)
    verdict_rows = _read_csv_rows(VERDICTS_PATH)

    verdict_by_id: dict[str, dict[str, str]] = {}
    for row in verdict_rows:
        case_id = (row.get("id") or "").strip()
        if case_id:
            verdict_by_id[case_id] = row

    merged_rows: list[dict[str, str]] = []
    for row in results_rows:
        case_id = (row.get("id") or "").strip()
        if not case_id or case_id not in verdict_by_id:
            continue
        verdict_row = verdict_by_id[case_id]
        merged_rows.append(
            {
                **row,
                "verdict": (verdict_row.get("verdict") or "").strip(),
                "notes": (verdict_row.get("notes") or "").strip(),
            }
        )

    partial_rows = [
        r for r in merged_rows if (r.get("verdict") or "").strip().lower() == "partial"
    ]

    partial_count_by_domain = {d: 0 for d in DOMAIN_ORDER}
    for row in partial_rows:
        domain = (row.get("domain") or "").strip()
        if domain in partial_count_by_domain:
            partial_count_by_domain[domain] += 1

    print("=== A) Partial Count by Domain ===")
    for domain in DOMAIN_ORDER:
        print(f"{domain}: {partial_count_by_domain[domain]}")
    print(f"TOTAL_PARTIAL: {len(partial_rows)}")

    partial_domain_filter_used_true = 0
    partial_domain_filter_used_false = 0
    for row in partial_rows:
        used = _parse_bool(row.get("domain_filter_used") or "")
        if used:
            partial_domain_filter_used_true += 1
        else:
            partial_domain_filter_used_false += 1

    print("")
    print("=== B) Domain Filter Usage (Partials Only) ===")
    print(f"partial_total: {len(partial_rows)}")
    print(f"domain_filter_used=True: {partial_domain_filter_used_true}")
    print(f"domain_filter_used=False: {partial_domain_filter_used_false}")

    n_total_values = [_parse_int(r.get("N_total") or "") for r in partial_rows]
    n_req_values = [_parse_int(r.get("N_req") or "") for r in partial_rows]
    count_n_total_le_4 = sum(1 for n in n_total_values if n <= 4)
    count_citations_eq_1 = sum(
        1 for r in partial_rows if _parse_int(r.get("citations_count") or "") == 1
    )

    print("")
    print("=== C) Evidence Thickness (Partials Only) ===")
    print(f"median_N_total: {_fmt_median(n_total_values)}")
    print(f"median_N_req: {_fmt_median(n_req_values)}")
    print(f"count_N_total<=4: {count_n_total_le_4}")
    print(f"count_citations_count==1: {count_citations_eq_1}")

    partial_framework_true = 0
    partial_framework_false = 0
    for row in partial_rows:
        if _is_framework_question(row.get("question") or ""):
            partial_framework_true += 1
        else:
            partial_framework_false += 1

    print("")
    print("=== D) Framework Question Rate (Partials Only) ===")
    print(f"framework_true: {partial_framework_true}")
    print(f"framework_false: {partial_framework_false}")

    partial_diag_rows: list[dict[str, object]] = []
    for row in partial_rows:
        n_total = _parse_int(row.get("N_total") or "")
        n_req = _parse_int(row.get("N_req") or "")
        n_domain = _parse_int(row.get("N_domain") or "")
        domain_key = (row.get("domain_key") or "").strip()
        domain_filter_used = _parse_bool(row.get("domain_filter_used") or "")
        would_use_domain_filter = n_domain >= MIN_DOMAIN_SENTENCES
        empty_scope_retry_used = _parse_bool(row.get("empty_scope_retry_used") or "")
        cited_short = _short_cited_sources(row.get("cited_sources") or "", max_items=2)
        partial_diag_rows.append(
            {
                "id": (row.get("id") or "").strip(),
                "domain": (row.get("domain") or "").strip(),
                "question": (row.get("question") or "").strip(),
                "domain_key": domain_key,
                "N_total": n_total,
                "N_req": n_req,
                "N_domain": n_domain,
                "domain_filter_used": domain_filter_used,
                "would_use_domain_filter": would_use_domain_filter,
                "empty_scope_retry_used": empty_scope_retry_used,
                "cited_sources_short": cited_short,
            }
        )

    print("")
    print("=== Partial Domain Filter Diagnostics (Partials Only) ===")
    print(
        "id | domain | domain_key | N_total | N_req | N_domain | MIN_DOMAIN_SENTENCES | "
        "would_use_domain_filter | empty_scope_retry_used | cited_sources(first2)"
    )
    for row in sorted(partial_diag_rows, key=lambda x: str(x["id"])):
        print(
            f"{row['id']} | {row['domain']} | {row['domain_key']} | {row['N_total']} | "
            f"{row['N_req']} | {row['N_domain']} | {MIN_DOMAIN_SENTENCES} | "
            f"{row['would_use_domain_filter']} | {row['empty_scope_retry_used']} | "
            f"{row['cited_sources_short']}"
        )

    count_n_domain_eq_0 = sum(1 for row in partial_diag_rows if int(row["N_domain"]) == 0)
    count_n_domain_eq_1 = sum(1 for row in partial_diag_rows if int(row["N_domain"]) == 1)
    count_domain_key_missing = sum(
        1 for row in partial_diag_rows if _is_domain_key_missing(str(row["domain_key"]))
    )
    count_n_domain_ge_2_but_not_used = sum(
        1
        for row in partial_diag_rows
        if int(row["N_domain"]) >= MIN_DOMAIN_SENTENCES and not bool(row["domain_filter_used"])
    )
    count_empty_scope_retry_used = sum(
        1 for row in partial_diag_rows if bool(row["empty_scope_retry_used"])
    )

    print("")
    print("=== E) Domain Filter Non-Use Breakdown (Partials Only) ===")
    print(f"count_N_domain==0: {count_n_domain_eq_0}")
    print(f"count_N_domain==1: {count_n_domain_eq_1}")
    print(f"count_domain_key_missing: {count_domain_key_missing}")
    print(f"count_N_domain>=2_but_not_used: {count_n_domain_ge_2_but_not_used}")
    print(f"count_empty_scope_retry_used: {count_empty_scope_retry_used}")

    worst10 = sorted(
        partial_diag_rows,
        key=lambda row: (
            int(row["N_domain"]),
            int(row["N_total"]),
            str(row["id"]),
        ),
    )[:10]
    for row in worst10:
        diag = _chunk_filter_diag_for_question(str(row.get("question") or ""))
        row["domain_chunk_filter_level_used"] = int(diag.get("domain_chunk_filter_level_used", 0))
        row["top_k_used"] = int(diag.get("top_k_used", 0))
        row["pool_size_returned"] = int(diag.get("pool_size_returned", 0))
        row["allowed_docs_relaxed_retry_used"] = bool(
            diag.get("allowed_docs_relaxed_retry_used", False)
        )
        row["chunks_before"] = int(diag.get("chunks_before", 0))
        row["chunks_after_level1"] = int(diag.get("chunks_after_level1", 0))
        row["chunks_after_level2"] = int(diag.get("chunks_after_level2", 0))
        row["chunk_filter_fallback_used"] = bool(diag.get("fallback_used", False))

    print("")
    print("Worst 10 partials by N_domain then N_total:")
    print(
        "id | domain | N_domain | N_total | N_req | domain_key | would_use_domain_filter | "
        "domain_chunk_filter_level_used | top_k_used | pool_size_returned | allowed_docs_relaxed_retry_used | chunks_before | chunks_after_level1 | "
        "chunks_after_level2 | fallback_used"
    )
    for row in worst10:
        print(
            f"{row['id']} | {row['domain']} | {row['N_domain']} | {row['N_total']} | "
            f"{row['N_req']} | {row['domain_key']} | {row['would_use_domain_filter']} | "
            f"{row['domain_chunk_filter_level_used']} | {row['top_k_used']} | {row['pool_size_returned']} | "
            f"{row['allowed_docs_relaxed_retry_used']} | {row['chunks_before']} | "
            f"{row['chunks_after_level1']} | {row['chunks_after_level2']} | "
            f"{row['chunk_filter_fallback_used']}"
        )

    evidence_rows = [row for row in worst10 if int(row["N_domain"]) == 0]
    print("")
    print("=== Evidence for N_domain=0 Partials (Worst-10) ===")
    print("")
    for row in evidence_rows:
        case_id = str(row["id"])
        cited_sources = str(row["cited_sources_short"])
        print(
            f"{case_id} | {row['domain']} | {row['domain_key']} | cited_sources: {cited_sources}"
        )
        evidence_sentences = _extract_answer_sentences_from_output(case_id, max_items=5)
        if evidence_sentences:
            for sent in evidence_sentences:
                print(f"- {sent}")
        else:
            print("- [no answer sentences found in eval/outputs]")
        print("")

    print("")
    print("=== END DIAGNOSIS ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
