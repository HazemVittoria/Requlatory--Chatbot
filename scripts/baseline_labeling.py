from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_runner import generate_baseline_scorecard, init_baseline_verdicts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/baseline_labeling.py")
    p.add_argument(
        "command",
        choices=["init-verdicts", "scorecard", "all"],
        help="init-verdicts: scaffold verdicts CSV; scorecard: build markdown; all: do both.",
    )
    p.add_argument("--results", default="eval/results_baseline.csv")
    p.add_argument("--verdicts", default="eval/verdicts_baseline.csv")
    p.add_argument("--scorecard", default="eval/scorecard_baseline.md")
    args = p.parse_args(argv)

    results = Path(args.results)
    verdicts = Path(args.verdicts)
    scorecard = Path(args.scorecard)

    if args.command in {"init-verdicts", "all"}:
        out = init_baseline_verdicts(results_path=results, verdicts_path=verdicts)
        print(json.dumps({"step": "init-verdicts", **out}, ensure_ascii=True))

    if args.command in {"scorecard", "all"}:
        out = generate_baseline_scorecard(
            results_path=results,
            verdicts_path=verdicts,
            output_path=scorecard,
        )
        print(json.dumps({"step": "scorecard", **out}, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
