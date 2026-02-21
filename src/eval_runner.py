from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from .index_integrity import validate_index_integrity
from .qa import answer

NOT_FOUND_TEXT = "Not found in provided PDFs"
FALLBACK_USED_RATE_ANSWERABLE_MAX = 0.15
BASELINE_ALLOWED_VERDICTS = {
    "unknown",
    "strong_correct",
    "partial",
    "incorrect",
    "correct_refusal",
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    scope: str
    expected_not_found: bool
    must_include: list[str]
    must_not_include: list[str]
    min_citations: int
    phase_a_require_authorities_when_eligible: list[str]


@dataclass(frozen=True)
class BaselineEvalCase:
    id: str
    question: str
    domain: str
    scope: str
    expected_outcome: str
    must_include: list[str]
    must_not_include: list[str]
    min_citations: int
    max_citations: int


def load_golden(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                id=str(row["id"]),
                question=row["question"],
                scope=row.get("scope", "MIXED"),
                expected_not_found=bool(row.get("expected_not_found", False)),
                must_include=list(row.get("must_include", [])),
                must_not_include=list(row.get("must_not_include", [])),
                min_citations=int(row.get("min_citations", 1)),
                phase_a_require_authorities_when_eligible=list(
                    row.get("phase_a_require_authorities_when_eligible", [])
                ),
            )
        )
    return cases


def load_baseline_eval_set(path: Path) -> list[BaselineEvalCase]:
    cases: list[BaselineEvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            BaselineEvalCase(
                id=str(row["id"]),
                question=str(row["question"]),
                domain=str(row.get("domain", "")),
                scope=str(row.get("scope", "MIXED")),
                expected_outcome=str(row.get("expected_outcome", "unknown")),
                must_include=list(row.get("must_include", [])),
                must_not_include=list(row.get("must_not_include", [])),
                min_citations=int(row.get("min_citations", 1)),
                max_citations=int(row.get("max_citations", 6)),
            )
        )
    return cases


def _is_not_found_answer(text: str) -> bool:
    return (text or "").strip() == NOT_FOUND_TEXT


def _safe_case_id(case_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id))
    return s or "unknown_case"


def _norm_authority(v: str) -> str:
    s = (v or "").strip().upper()
    return s or "OTHER"


def _phase_a_active_attempt(phase_a: dict[str, Any]) -> dict[str, Any]:
    attempts = list(phase_a.get("attempts", []) or [])
    if not attempts:
        return {}
    if bool(phase_a.get("fallback_used", False)):
        return dict(attempts[-1] or {})
    return dict(attempts[0] or {})


def _eligible_authorities_above_soft_floor(phase_a: dict[str, Any]) -> set[str]:
    balancing = dict(phase_a.get("balancing", {}) or {})
    if not balancing:
        return set()
    try:
        floor_score = float(balancing.get("soft_floor_score"))
    except Exception:
        return set()
    if floor_score <= 0.0:
        return set()

    active = _phase_a_active_attempt(phase_a)
    rows = list(active.get("top_results", []) or [])
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            score = float(row.get("base_score", row.get("score", 0.0)))
        except Exception:
            continue
        if score < floor_score:
            continue
        out.add(_norm_authority(str(row.get("authority") or "")))
    return out


def _kept_phase_a_authorities(phase_a: dict[str, Any]) -> set[str]:
    balancing = dict(phase_a.get("balancing", {}) or {})
    kept = list(balancing.get("authorities_kept", []) or [])
    return {_norm_authority(str(v)) for v in kept if str(v).strip()}


def _write_failure_artifact(
    *,
    run_dir: Path,
    case_id: str,
    question: str,
    expected_not_found: bool,
    outcome: str,
    res: Any,
    fail_reasons: list[str],
) -> None:
    trace = dict(getattr(res, "debug_trace", {}) or {})
    phase_a = dict(trace.get("phase_a", {}) or {})
    phase_b = dict(trace.get("phase_b", {}) or {})
    phase_c = dict(trace.get("phase_c", {}) or {})

    if "relevant_facts_kept" not in phase_b:
        phase_b["relevant_facts_kept"] = [
            {
                "pdf": f.pdf,
                "page": int(f.page),
                "chunk_id": f.chunk_id,
            }
            for f in (getattr(res, "relevant_facts", []) or [])
        ]

    if "answer_sentences" not in phase_c:
        phase_c["answer_sentences"] = []
    if "citations" not in phase_c:
        phase_c["citations"] = [
            {
                "pdf": c.doc_id,
                "citation": str(getattr(c, "doc_title", "") or c.doc_id),
                "page": int(c.page),
                "chunk_id": c.chunk_id,
            }
            for c in (getattr(res, "citations", []) or [])
        ]

    payload = {
        "question": question,
        "expected_not_found": bool(expected_not_found),
        "outcome": outcome,
        "phase_a": {
            "queries_used": phase_a.get("queries_used", []),
            "thresholds_attempted": phase_a.get("thresholds_attempted", []),
            "attempts": phase_a.get("attempts", []),
            "fallback_used": bool(getattr(res, "phase_a_fallback_used", False)),
            "balancing": phase_a.get("balancing", {}),
        },
        "phase_b": {
            "relevant_facts_kept": phase_b.get("relevant_facts_kept", []),
        },
        "phase_c": {
            "answer_sentences": phase_c.get("answer_sentences", []),
            "citations": phase_c.get("citations", []),
        },
        "validator": {
            "ok": False,
            "reasons": fail_reasons,
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{_safe_case_id(case_id)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _cited_sources(citations: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in citations:
        doc = str(getattr(c, "doc_id", "") or "").strip()
        page = int(getattr(c, "page", 0) or 0)
        if not doc or page <= 0:
            continue
        key = f"{doc}:{page}"
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def init_baseline_verdicts(
    results_path: Path = Path("eval/results_baseline.csv"),
    verdicts_path: Path = Path("eval/verdicts_baseline.csv"),
) -> dict[str, Any]:
    rows = _read_csv_rows(results_path)
    existing: dict[str, dict[str, str]] = {}
    for r in _read_csv_rows(verdicts_path):
        case_id = str(r.get("id") or "").strip()
        if case_id:
            existing[case_id] = r

    verdict_rows: list[dict[str, str]] = []
    unknown_count = 0
    for r in rows:
        case_id = str(r.get("id") or "").strip()
        if not case_id:
            continue
        prev = existing.get(case_id, {})
        verdict = str(prev.get("verdict") or "unknown").strip().lower()
        notes = str(prev.get("notes") or "").strip()
        if verdict not in BASELINE_ALLOWED_VERDICTS:
            verdict = "unknown"
        if verdict == "unknown":
            unknown_count += 1
        verdict_rows.append({"id": case_id, "verdict": verdict, "notes": notes})

    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with verdicts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "verdict", "notes"])
        writer.writeheader()
        writer.writerows(verdict_rows)

    return {
        "results_path": str(results_path).replace("\\", "/"),
        "verdicts_path": str(verdicts_path).replace("\\", "/"),
        "count": len(verdict_rows),
        "unknown_count": unknown_count,
    }


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (100.0 * float(part)) / float(total)


def _fmt_pct(part: int, total: int) -> str:
    return f"{_pct(part, total):.1f}%"


def _load_verdict_map(verdicts_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in _read_csv_rows(verdicts_path):
        case_id = str(r.get("id") or "").strip()
        if not case_id:
            continue
        verdict = str(r.get("verdict") or "unknown").strip().lower()
        notes = str(r.get("notes") or "").strip()
        if verdict not in BASELINE_ALLOWED_VERDICTS:
            raise ValueError(
                f"Invalid verdict '{verdict}' for id '{case_id}'. "
                f"Allowed: {sorted(BASELINE_ALLOWED_VERDICTS)}"
            )
        out[case_id] = {"verdict": verdict, "notes": notes}
    return out


def generate_baseline_scorecard(
    results_path: Path = Path("eval/results_baseline.csv"),
    verdicts_path: Path = Path("eval/verdicts_baseline.csv"),
    output_path: Path = Path("eval/scorecard_baseline.md"),
) -> dict[str, Any]:
    results = _read_csv_rows(results_path)
    verdict_map = _load_verdict_map(verdicts_path)

    merged: list[dict[str, str]] = []
    for r in results:
        case_id = str(r.get("id") or "").strip()
        if not case_id:
            continue
        v = verdict_map.get(case_id, {"verdict": "unknown", "notes": ""})
        verdict = str(v.get("verdict") or "unknown").strip().lower()
        notes = str(v.get("notes") or "").strip()
        if verdict not in BASELINE_ALLOWED_VERDICTS:
            raise ValueError(
                f"Invalid verdict '{verdict}' for id '{case_id}'. "
                f"Allowed: {sorted(BASELINE_ALLOWED_VERDICTS)}"
            )
        merged.append({**r, "verdict": verdict, "notes": notes})

    total = len(merged)
    overall_counts = {
        "strong_correct": 0,
        "partial": 0,
        "incorrect": 0,
        "correct_refusal": 0,
        "unknown": 0,
    }
    hallucinations = 0
    by_domain: dict[str, dict[str, int]] = {}
    for r in merged:
        verdict = str(r.get("verdict") or "unknown").strip().lower()
        domain = str(r.get("domain") or "").strip()
        notes_l = str(r.get("notes") or "").lower()
        if "hallucination" in notes_l:
            hallucinations += 1
        if verdict not in overall_counts:
            verdict = "unknown"
        overall_counts[verdict] += 1
        if domain not in by_domain:
            by_domain[domain] = {
                "total": 0,
                "strong_correct": 0,
                "partial": 0,
                "incorrect": 0,
                "correct_refusal": 0,
                "unknown": 0,
            }
        by_domain[domain]["total"] += 1
        by_domain[domain][verdict] += 1

    strong = overall_counts["strong_correct"]
    partial = overall_counts["partial"]
    strong_partial = strong + partial

    worst_pool = [
        r
        for r in merged
        if str(r.get("verdict") or "").strip().lower() in {"incorrect", "partial"}
    ]
    pri = {"incorrect": 0, "partial": 1}
    worst_pool.sort(
        key=lambda x: (
            pri.get(str(x.get("verdict") or "").strip().lower(), 9),
            str(x.get("id") or ""),
        )
    )
    worst10 = worst_pool[:10]

    lines: list[str] = []
    lines.append("# Baseline Scorecard")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append(f"- total: {total}")
    lines.append(f"- strong_correct: {overall_counts['strong_correct']} ({_fmt_pct(overall_counts['strong_correct'], total)})")
    lines.append(f"- partial: {overall_counts['partial']} ({_fmt_pct(overall_counts['partial'], total)})")
    lines.append(f"- incorrect: {overall_counts['incorrect']} ({_fmt_pct(overall_counts['incorrect'], total)})")
    lines.append(f"- correct_refusal: {overall_counts['correct_refusal']} ({_fmt_pct(overall_counts['correct_refusal'], total)})")
    lines.append(f"- unknown: {overall_counts['unknown']} ({_fmt_pct(overall_counts['unknown'], total)})")
    lines.append(f"- strong+partial % (target >= 85%): {_fmt_pct(strong_partial, total)}")
    lines.append(f"- strong % (target >= 75%): {_fmt_pct(strong, total)}")
    lines.append(f"- hallucinations count (target = 0): {hallucinations}")
    lines.append("")
    lines.append("## By-domain table")
    lines.append("")
    lines.append("| domain | total | strong | partial | incorrect | correct_refusal | strong% | strong+partial% | refusal% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for domain in sorted(by_domain.keys()):
        d = by_domain[domain]
        d_total = int(d["total"])
        d_strong = int(d["strong_correct"])
        d_partial = int(d["partial"])
        d_incorrect = int(d["incorrect"])
        d_refusal = int(d["correct_refusal"])
        lines.append(
            f"| {domain} | {d_total} | {d_strong} | {d_partial} | {d_incorrect} | {d_refusal} | "
            f"{_fmt_pct(d_strong, d_total)} | {_fmt_pct(d_strong + d_partial, d_total)} | {_fmt_pct(d_refusal, d_total)} |"
        )
    lines.append("")
    lines.append("## Worst 10")
    lines.append("")
    lines.append("| id | domain | question | cited_sources | domain_filter_used | empty_scope_retry_used |")
    lines.append("|---|---|---|---|---|---|")
    for r in worst10:
        lines.append(
            f"| {str(r.get('id') or '')} | {str(r.get('domain') or '')} | {str(r.get('question') or '')} | "
            f"{str(r.get('cited_sources') or '')} | {str(r.get('domain_filter_used') or '')} | {str(r.get('empty_scope_retry_used') or '')} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "results_path": str(results_path).replace("\\", "/"),
        "verdicts_path": str(verdicts_path).replace("\\", "/"),
        "scorecard_path": str(output_path).replace("\\", "/"),
        "count": total,
        "unknown_count": overall_counts["unknown"],
        "hallucinations": hallucinations,
    }


def run_baseline_eval(
    eval_set_path: Path = Path("eval/eval_set.jsonl"),
    results_path: Path = Path("eval/results_baseline.csv"),
    outputs_dir: Path = Path("eval/outputs"),
    *,
    save_outputs: bool = True,
) -> dict[str, Any]:
    validate_index_integrity(print_report=True)
    cases = load_baseline_eval_set(eval_set_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if save_outputs:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "domain",
        "question",
        "refusal",
        "verdict",
        "citations_count",
        "cited_sources",
        "domain_key",
        "domain_filter_used",
        "N_total",
        "N_req",
        "N_domain",
        "empty_scope_retry_used",
        "notes",
    ]

    row_count = 0
    refusal_count = 0
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cases:
            res = answer(c.question, case_id=c.id, scope=c.scope)
            text = (res.text or "").strip()
            refusal = _is_not_found_answer(text)
            if refusal:
                refusal_count += 1
            citations = list(getattr(res, "citations", []) or [])
            cited_sources = _cited_sources(citations)

            trace = dict(getattr(res, "debug_trace", {}) or {})
            phase_a = dict(trace.get("phase_a", {}) or {})
            phase_c = dict(trace.get("phase_c", {}) or {})
            domain_stats = dict(phase_c.get("domain_filter_stats", {}) or {})

            writer.writerow(
                {
                    "id": c.id,
                    "domain": c.domain,
                    "question": c.question,
                    "refusal": bool(refusal),
                    "verdict": "unknown",
                    "citations_count": int(len(citations)),
                    "cited_sources": ";".join(cited_sources),
                    "domain_key": domain_stats.get("domain_key"),
                    "domain_filter_used": bool(domain_stats.get("domain_filter_used", False)),
                    "N_total": int(domain_stats.get("N_total_sentences", 0)),
                    "N_req": int(domain_stats.get("N_requirement_bearing", 0)),
                    "N_domain": int(domain_stats.get("N_domain_eligible", 0)),
                    "empty_scope_retry_used": bool(phase_a.get("empty_result_scope_retry_used", False)),
                    "notes": "",
                }
            )
            row_count += 1

            if save_outputs:
                out_name = f"{_safe_case_id(c.id)}.txt"
                out_path = outputs_dir / out_name
                out_lines = [text, "", "Citations:"]
                if citations:
                    for cit in citations:
                        doc_label = str(getattr(cit, "doc_title", "") or getattr(cit, "doc_id", ""))
                        out_lines.append(
                            f"- {doc_label} | p{int(getattr(cit, 'page', 0) or 0)} | {getattr(cit, 'chunk_id', '')}"
                        )
                else:
                    out_lines.append("- none")
                out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    return {
        "eval_set_path": str(eval_set_path).replace("\\", "/"),
        "results_path": str(results_path).replace("\\", "/"),
        "outputs_dir": str(outputs_dir).replace("\\", "/") if save_outputs else "",
        "count": row_count,
        "refusal_count": refusal_count,
    }


def run_eval(golden_path: Path = Path("golden_set.jsonl")) -> dict[str, Any]:
    index_report = validate_index_integrity(print_report=True)
    cases = load_golden(golden_path)
    rows: list[dict[str, Any]] = []
    passed = 0
    failure_count = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_run_dir = Path("eval_artifacts") / run_id

    correct_refusal = 0
    incorrect_refusal = 0
    hallucination = 0
    phase_a_fallback_used_count = 0
    answerable_count = 0
    answerable_fallback_used_count = 0

    for c in cases:
        res = answer(c.question, case_id=c.id, scope=c.scope)
        text = (res.text or "").strip()
        text_l = text.lower()
        citations = res.citations or []
        is_not_found = _is_not_found_answer(text)
        phase_a_fallback_used = bool(getattr(res, "phase_a_fallback_used", False))
        if phase_a_fallback_used:
            phase_a_fallback_used_count += 1
        if not c.expected_not_found:
            answerable_count += 1
            if phase_a_fallback_used:
                answerable_fallback_used_count += 1

        fail_reasons: list[str] = []
        outcome = "pass"

        if c.expected_not_found:
            # Refusal-class case.
            if is_not_found:
                correct_refusal += 1
            else:
                outcome = "hallucination"
                hallucination += 1
                fail_reasons.append("hallucination_expected_refusal")
                if citations:
                    fail_reasons.append("hallucination_with_citations")
                else:
                    fail_reasons.append("hallucination_without_citations")
        else:
            # Answer-class case.
            if is_not_found:
                outcome = "incorrect_refusal"
                incorrect_refusal += 1
                fail_reasons.append("incorrect_refusal")
            else:
                for term in c.must_include:
                    if term.lower() not in text_l:
                        fail_reasons.append(f"missing:{term}")
                for term in c.must_not_include:
                    if term.lower() in text_l:
                        fail_reasons.append(f"forbidden:{term}")
                if len(citations) < c.min_citations:
                    fail_reasons.append("few_citations")

                required_auths = {
                    _norm_authority(a) for a in c.phase_a_require_authorities_when_eligible if str(a).strip()
                }
                if required_auths:
                    trace = dict(getattr(res, "debug_trace", {}) or {})
                    phase_a_trace = dict(trace.get("phase_a", {}) or {})
                    eligible_auths = _eligible_authorities_above_soft_floor(phase_a_trace)
                    if required_auths.issubset(eligible_auths):
                        kept_auths = _kept_phase_a_authorities(phase_a_trace)
                        missing = sorted(required_auths - kept_auths)
                        if missing:
                            fail_reasons.append("phase_a_balance_missing:" + ",".join(missing))

        ok = len(fail_reasons) == 0
        if ok:
            passed += 1
        else:
            failure_count += 1
            _write_failure_artifact(
                run_dir=artifacts_run_dir,
                case_id=c.id,
                question=c.question,
                expected_not_found=c.expected_not_found,
                outcome=outcome,
                res=res,
                fail_reasons=fail_reasons,
            )

        rows.append(
            {
                "id": c.id,
                "question": c.question,
                "expected_not_found": c.expected_not_found,
                "ok": ok,
                "outcome": outcome if not ok else ("correct_refusal" if c.expected_not_found else "answer_ok"),
                "fail_reasons": fail_reasons,
                "phase_a_fallback_used": phase_a_fallback_used,
                "answer": text,
                "citations": [f"{x.doc_id}|p{x.page}|{x.chunk_id}" for x in citations],
                "citations_display": [
                    f"{str(getattr(x, 'doc_title', '') or x.doc_id)}|p{x.page}|{x.chunk_id}" for x in citations
                ],
            }
        )

    total = len(cases)
    phase_a_fallback_used_rate = (phase_a_fallback_used_count / total) if total else 0.0
    fallback_used_rate_answerable = (
        (answerable_fallback_used_count / answerable_count) if answerable_count else 0.0
    )
    fallback_rate_threshold_exceeded = fallback_used_rate_answerable > FALLBACK_USED_RATE_ANSWERABLE_MAX
    alert_message = ""
    if fallback_rate_threshold_exceeded:
        alert_message = (
            "Fallback retrieval triggered too often; check embeddings/index changes "
            "or similarity calibration."
        )

    out = {
        "index_integrity": {
            "total_documents": index_report.total_documents,
            "chunks_per_document": index_report.chunks_per_document,
            "total_chunks": index_report.total_chunks,
            "embedding_model_version": index_report.embedding_model_version,
            "index_hash": index_report.index_hash,
        },
        "count": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "metrics": {
            "correct_refusal": correct_refusal,
            "incorrect_refusal": incorrect_refusal,
            "hallucination": hallucination,
            "phase_a_fallback_used_count": phase_a_fallback_used_count,
            "phase_a_fallback_used_rate": phase_a_fallback_used_rate,
            "fallback_used_rate_answerable": fallback_used_rate_answerable,
        },
        "alerts": {
            "fallback_used_rate_answerable_max": FALLBACK_USED_RATE_ANSWERABLE_MAX,
            "fallback_used_rate_answerable_exceeded": fallback_rate_threshold_exceeded,
            "message": alert_message,
        },
        "ci_fail": fallback_rate_threshold_exceeded,
        "results": rows,
    }
    if failure_count > 0:
        out["failure_artifacts"] = {
            "run_id": run_id,
            "path": str(artifacts_run_dir).replace("\\", "/"),
            "count": failure_count,
        }
    return out


if __name__ == "__main__":
    out = run_eval()
    print(json.dumps(out, ensure_ascii=True, indent=2))
    raise SystemExit(1 if out.get("ci_fail") else 0)
