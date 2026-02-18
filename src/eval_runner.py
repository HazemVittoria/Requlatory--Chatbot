from __future__ import annotations

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
        res = answer(c.question, scope=c.scope)
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
