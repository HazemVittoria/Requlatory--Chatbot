from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .index_integrity import validate_index_integrity
from .qa import answer
from .search import rebuild_index


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.cli")
    p.add_argument("question", help="Question to answer.")
    p.add_argument("--scope", default="MIXED", help="Scope filter: MIXED|FDA|EMA|ICH|SOPS")
    p.add_argument("--topk", type=int, default=5, help="Top chunks for retrieval.")
    p.add_argument("--chunk-chars", type=int, default=1000, help="Chunk size in characters.")
    p.add_argument("--min-chunk-chars", type=int, default=220, help="Minimum chunk size in characters.")
    p.add_argument("--data-dir", default="data", help="Folder containing authority subfolders with PDFs.")
    p.add_argument("--rebuild-index", action="store_true", help="Force full index rebuild.")
    p.add_argument("--json", action="store_true", help="Print JSON output.")
    p.add_argument("--show-citations", action="store_true", help="Also print citation list after answer text.")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.rebuild_index:
        rebuild_index(
            data_dir=data_dir,
            chunk_chars=max(300, int(args.chunk_chars)),
            min_chunk_chars=max(80, int(args.min_chunk_chars)),
            force=True,
        )
    validate_index_integrity(
        data_dir=data_dir,
        chunk_chars=max(300, int(args.chunk_chars)),
        min_chunk_chars=max(80, int(args.min_chunk_chars)),
        print_report=True,
    )

    res = answer(
        args.question,
        scope=args.scope,
        top_k=max(1, int(args.topk)),
        data_dir=data_dir,
        chunk_chars=max(300, int(args.chunk_chars)),
        min_chunk_chars=max(80, int(args.min_chunk_chars)),
    )

    if args.json:
        print(json.dumps(_jsonable(res), ensure_ascii=False, indent=2))
        return 0

    print((res.text or "").rstrip())
    if args.show_citations and res.citations:
        print("\nCitations:")
        for c in res.citations:
            print(f"- {c.doc_id} | page {c.page} | {c.chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
