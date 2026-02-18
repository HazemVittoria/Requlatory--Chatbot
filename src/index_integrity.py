from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .ingestion import authority_from_folder
from .search import _CACHE_VERSION, rebuild_index
from . import search as search_module

EMBEDDING_MODEL_VERSION = "tfidf_word_1_2+char_wb_3_5"


@dataclass(frozen=True)
class IndexIntegrityReport:
    total_documents: int
    chunks_per_document: dict[str, int]
    total_chunks: int
    embedding_model_version: str
    index_hash: str


def _expected_documents(data_dir: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if not data_dir.exists():
        return out
    for folder in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        authority = authority_from_folder(folder.name)
        for pdf_path in sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower()):
            rel = str(pdf_path.relative_to(data_dir)).replace("\\", "/")
            out.append((rel, authority, pdf_path.name))
    return out


def _build_index_report(data_dir: Path, chunk_chars: int, min_chunk_chars: int) -> IndexIntegrityReport:
    rebuild_index(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars, force=False)
    corpus = getattr(search_module, "_CORPUS", None)
    if not isinstance(corpus, list):
        raise RuntimeError("Index corpus is not loaded.")

    expected_docs = _expected_documents(data_dir)
    counts_by_key: dict[tuple[str, str], int] = {}
    for c in corpus:
        authority = str(c.get("authority") or "OTHER")
        file_name = str(c.get("file") or "")
        key = (authority, file_name)
        counts_by_key[key] = counts_by_key.get(key, 0) + 1

    chunks_per_document: dict[str, int] = {}
    for rel, authority, file_name in expected_docs:
        chunks_per_document[rel] = int(counts_by_key.get((authority, file_name), 0))

    total_documents = len(chunks_per_document)
    total_chunks = sum(chunks_per_document.values())

    hash_lines = [f"{name}:{chunks_per_document[name]}" for name in sorted(chunks_per_document)]
    digest = hashlib.sha256("\n".join(hash_lines).encode("utf-8")).hexdigest()
    embedding_model_version = f"{EMBEDDING_MODEL_VERSION}|{_CACHE_VERSION}"

    return IndexIntegrityReport(
        total_documents=total_documents,
        chunks_per_document=chunks_per_document,
        total_chunks=total_chunks,
        embedding_model_version=embedding_model_version,
        index_hash=digest,
    )


def validate_index_integrity(
    *,
    data_dir: Path = Path("data"),
    chunk_chars: int = 1000,
    min_chunk_chars: int = 220,
    print_report: bool = True,
) -> IndexIntegrityReport:
    report = _build_index_report(data_dir=data_dir, chunk_chars=chunk_chars, min_chunk_chars=min_chunk_chars)

    zero_chunk_docs = [doc for doc, count in report.chunks_per_document.items() if int(count) == 0]
    if report.total_documents == 0:
        raise RuntimeError("Index integrity failure: no expected PDF files found in data directory.")
    if zero_chunk_docs:
        msg = (
            "Index integrity failure: one or more expected PDFs have zero chunks: "
            + ", ".join(zero_chunk_docs)
        )
        raise RuntimeError(msg)

    if print_report:
        payload = {
            "total_documents": report.total_documents,
            "chunks_per_document": report.chunks_per_document,
            "total_chunks": report.total_chunks,
            "embedding_model_version": report.embedding_model_version,
            "index_hash": report.index_hash,
        }
        print("[index_integrity] " + json.dumps(payload, ensure_ascii=True), file=sys.stderr)

    return report

