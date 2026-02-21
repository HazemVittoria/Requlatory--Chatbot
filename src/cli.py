from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .ingestion import inspect_pdf_page_labels
from .index_integrity import validate_index_integrity
from .qa import answer
from .qa_types import display_page_value
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
    p.add_argument("question", nargs="?", default="", help="Question to answer.")
    p.add_argument("--scope", default="MIXED", help="Scope filter: MIXED|FDA|EMA|ICH|SOPS")
    p.add_argument("--topk", type=int, default=5, help="Top chunks for retrieval.")
    p.add_argument("--chunk-chars", type=int, default=1000, help="Chunk size in characters.")
    p.add_argument("--min-chunk-chars", type=int, default=220, help="Minimum chunk size in characters.")
    p.add_argument("--data-dir", default="data", help="Folder containing authority subfolders with PDFs.")
    p.add_argument("--rebuild-index", action="store_true", help="Force full index rebuild.")
    p.add_argument("--json", action="store_true", help="Print JSON output.")
    p.add_argument("--show-citations", action="store_true", help="Also print citation list after answer text.")
    p.add_argument(
        "--inspect-page-labels",
        default="",
        help="Print page-index/page-label mapping for a PDF path and exit.",
    )
    p.add_argument("--inspect-limit", type=int, default=10, help="Row limit for --inspect-page-labels output.")
    args = p.parse_args(argv)

    if args.inspect_page_labels:
        rows = inspect_pdf_page_labels(
            Path(args.inspect_page_labels),
            limit=max(1, int(args.inspect_limit)),
        )
        for i, viewer, label in rows:
            print(f"{i}\t{viewer}\t{label or ''}")
        return 0

    if not str(args.question or "").strip():
        raise SystemExit("question is required unless --inspect-page-labels is used")

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
    if res.citations:
        print("\n[CITDBG]")
        for c in res.citations:
            display_page = display_page_value(
                pdf_index=int(getattr(c, "pdf_index", 0) or 0),
                page_label=str(getattr(c, "page_label", "") or "").strip() or None,
            )
            print(
                "file={file} chunk_id={chunk_id} pdf_index={pdf_index} page_label={page_label} "
                "page={page} display_page={display_page}".format(
                    file=str(getattr(c, "doc_id", "") or ""),
                    chunk_id=str(getattr(c, "chunk_id", "") or ""),
                    pdf_index=int(getattr(c, "pdf_index", 0) or 0),
                    page_label=str(getattr(c, "page_label", "") or "").strip() or "",
                    page=int(getattr(c, "page", 0) or 0),
                    display_page=display_page,
                )
            )
    if args.show_citations and res.citations:
        print("\nCitations:")
        for c in res.citations:
            doc_label = str(getattr(c, "doc_title", "") or c.doc_id)
            display_page = display_page_value(
                pdf_index=int(getattr(c, "pdf_index", 0) or 0),
                page_label=str(getattr(c, "page_label", "") or "").strip() or None,
            )
            print(f"- {doc_label} | {display_page} | {c.chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
