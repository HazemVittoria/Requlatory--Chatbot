from __future__ import annotations

import logging
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object .*")

_WS_RE = re.compile(r"\s+")
_PAGE_RE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", flags=re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", flags=re.IGNORECASE)
_URL_RE = re.compile(r"^\s*(?:https?://|www\.)", flags=re.IGNORECASE)
_SECTION_PREFIX_RE = re.compile(r"^\s*(?:annex|chapter|section|part|appendix|module)\b", flags=re.IGNORECASE)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,4}|[ivxlcdm]{1,8})[.)]?\s+[A-Z].{2,}$")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-(),/&]{4,}$")
_HYPHENATED_END_RE = re.compile(r"[A-Za-z]{2,}-$")
_TITLE_DROP_RE = re.compile(r"^\s*(?:untitled|document\d*|none|null|na|n/a)\s*$", flags=re.IGNORECASE)

_GENERIC_TITLE_LINES = {
    "table of contents",
    "contents",
    "introduction",
    "purpose",
    "scope",
}


AUTHORITY_BY_FOLDER = {
    "fda": "FDA",
    "ema": "EMA",
    "eu_gmp": "EU_GMP",
    "ich": "ICH",
    "pic_s": "PIC_S",
    "who": "WHO",
    "sop": "SOP",
    "sops": "SOP",
    "other": "OTHER",
    "others": "OTHER",
}


def authority_from_folder(folder_name: str) -> str:
    return AUTHORITY_BY_FOLDER.get((folder_name or "").strip().lower(), "OTHER")


def _normalize_line(line: str) -> str:
    s = (
        (line or "")
        .replace("\u00a0", " ")
        .replace("\uf0b7", " ")
        .replace("\u2022", " ")
        .replace("", " ")
    )
    s = re.sub(r"[\t\r]", " ", s)
    return _WS_RE.sub(" ", s).strip()


def _normalize_for_frequency(line: str) -> str:
    s = _normalize_line(line).lower()
    if not s:
        return ""
    s = _PAGE_RE.sub(" ", s)
    s = re.sub(r"\b\d+\b", "#", s)
    return _WS_RE.sub(" ", s).strip()


def _is_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 180:
        return False
    if _SECTION_PREFIX_RE.match(s):
        return True
    if _NUMBERED_HEADING_RE.match(s):
        return True
    if _ALL_CAPS_HEADING_RE.match(s):
        letters = sum(1 for ch in s if ch.isalpha())
        return letters >= 6
    return False


def _is_artifact(line: str) -> bool:
    s = _normalize_line(line)
    sl = s.lower()
    if not s:
        return True
    if _PAGE_NUM_RE.match(s):
        return True
    if _PAGE_RE.search(sl):
        return True
    if _URL_RE.match(s):
        return True
    if "table of contents" in sl:
        return True
    if set(s) <= {".", "-", "_"} and len(s) >= 6:
        return True
    if re.match(r"^[\W_]{4,}$", s):
        return True
    return False


def _merge_hyphenated_linebreaks(lines: list[str]) -> list[str]:
    if not lines:
        return []

    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = _normalize_line(lines[i])
        if not cur:
            i += 1
            continue
        if i + 1 < len(lines):
            nxt = _normalize_line(lines[i + 1])
            if nxt and _HYPHENATED_END_RE.search(cur) and nxt[0].islower() and not _is_heading(nxt):
                out.append(f"{cur[:-1]}{nxt}")
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_title_value(raw: str) -> str:
    s = _normalize_line(raw)
    s = s.strip(" -_:;,.")
    if not s:
        return ""
    if _TITLE_DROP_RE.match(s):
        return ""
    if len(s) < 6:
        return ""
    letters = sum(1 for ch in s if ch.isalpha())
    if letters < 4:
        return ""
    return s


def _metadata_title(reader: PdfReader) -> str:
    try:
        md = reader.metadata
    except Exception:
        md = None
    if md is None:
        return ""
    raw = ""
    try:
        raw = str(getattr(md, "title", "") or "")
    except Exception:
        raw = ""
    if not raw:
        try:
            raw = str(md.get("/Title") or "")
        except Exception:
            raw = ""
    return _clean_title_value(raw)


def _heading_like_score(line: str) -> int:
    s = _normalize_line(line)
    if not s:
        return -1
    sl = s.lower()
    if sl in _GENERIC_TITLE_LINES:
        return -1
    if _is_artifact(s):
        return -1
    score = 0
    if _is_heading(s):
        score += 3
    n_words = len(s.split())
    if 3 <= n_words <= 18:
        score += 2
    if len(s) <= 140:
        score += 1
    if re.search(r"\b(guidance|guideline|gmp|quality|validation|integrity|annex|cfr|ich)\b", sl):
        score += 1
    return score


def _title_from_first_page(page_records: list[dict[str, Any]]) -> str:
    if not page_records:
        return ""
    lines = list(page_records[0].get("lines", []))
    best_line = ""
    best_score = -1
    for raw in lines[:100]:
        line = _normalize_line(str(raw))
        score = _heading_like_score(line)
        if score > best_score:
            best_score = score
            best_line = line
    if best_score < 0:
        return ""
    return _clean_title_value(best_line)


def _citation_title(reader: PdfReader, page_records: list[dict[str, Any]], fallback_name: str) -> str:
    title = _metadata_title(reader)
    if title:
        return title
    title = _title_from_first_page(page_records)
    if title:
        return title
    stem = _clean_title_value(Path(fallback_name).stem.replace("_", " ").replace("-", " "))
    if stem:
        return stem
    return str(fallback_name)


def _build_blocks(lines: list[str]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current_section = "Unknown"
    para: list[str] = []

    def flush_para() -> None:
        if not para:
            return
        text = _WS_RE.sub(" ", " ".join(para)).strip()
        para.clear()
        if not text:
            return
        blocks.append({"section": current_section, "text": text})

    for raw in lines:
        line = _normalize_line(raw)
        if not line or _is_artifact(line):
            flush_para()
            continue
        if _is_heading(line):
            flush_para()
            current_section = line
            continue
        if para and para[-1].endswith((".", "!", "?")):
            flush_para()
        para.append(line)

    flush_para()
    return blocks


def _chunk_blocks(blocks: list[dict[str, str]], max_chars: int, min_chars: int) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_len = 0
    section = "Unknown"

    def flush() -> None:
        nonlocal current, current_len, section
        if not current:
            return
        text = "\n".join(current).strip()
        current = []
        current_len = 0
        if text:
            chunks.append({"section": section, "text": text})
        section = "Unknown"

    for block in blocks:
        btext = (block.get("text") or "").strip()
        bsection = (block.get("section") or "Unknown").strip() or "Unknown"
        if not btext:
            continue

        parts = [btext]
        if len(btext) > max_chars:
            parts = []
            cur = ""
            for s in _split_sentences(btext):
                if not cur:
                    cur = s
                elif len(cur) + 1 + len(s) <= max_chars:
                    cur += " " + s
                else:
                    parts.append(cur.strip())
                    cur = s
            if cur:
                parts.append(cur.strip())

        for part in parts:
            if not part:
                continue
            add_len = len(part) if not current else (1 + len(part))
            if current and current_len + add_len > max_chars:
                flush()
            if not current:
                section = bsection
            current.append(part)
            current_len += add_len

    flush()

    merged: list[dict[str, str]] = []
    for c in chunks:
        if merged and len(c["text"]) < min_chars:
            merged[-1]["text"] = f'{merged[-1]["text"]}\n{c["text"]}'.strip()
        else:
            merged.append(c)
    return merged


def ingest_pdf(
    pdf_path: Path,
    authority: str,
    repeat_line_doc_frequency: float = 0.60,
    chunk_chars: int = 1000,
    min_chunk_chars: int = 220,
) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    page_records: list[dict[str, Any]] = []
    line_df: Counter[str] = Counter()

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        lines = _merge_hyphenated_linebreaks(raw_text.splitlines())
        page_records.append({"page": page_idx, "lines": lines})
        seen = {_normalize_for_frequency(ln) for ln in lines}
        seen.discard("")
        line_df.update(seen)

    min_df = max(2, int(math.ceil(max(1, len(page_records)) * repeat_line_doc_frequency)))
    repeated = {line for line, freq in line_df.items() if freq >= min_df}
    citation_title = pdf_path.name

    out: list[dict[str, Any]] = []
    for rec in page_records:
        page = int(rec["page"])
        kept_lines = [
            ln
            for ln in rec["lines"]
            if _normalize_for_frequency(ln) not in repeated and not _is_artifact(ln)
        ]
        blocks = _build_blocks(kept_lines)
        if not blocks:
            continue

        chunks = _chunk_blocks(blocks, max_chars=chunk_chars, min_chars=min_chunk_chars)
        for idx, chunk in enumerate(chunks, start=1):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "authority": authority,
                    "source": authority,
                    "source_type": "INTERNAL_SOP" if authority == "SOP" else "REGULATORY",
                    "is_internal_sop": bool(authority == "SOP"),
                    "file": pdf_path.name,
                    "citation": citation_title,
                    "source_pdf_name": pdf_path.name,
                    "page": page,
                    "section": chunk.get("section") or "Unknown",
                    "chunk_id": f"p{page}_c{idx}",
                    "text": text,
                }
            )
    return out


def build_corpus(data_dir: Path, chunk_chars: int = 1000, min_chunk_chars: int = 220) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    if not data_dir.exists():
        return corpus

    for folder in sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        authority = authority_from_folder(folder.name)
        for pdf_path in sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower()):
            corpus.extend(
                ingest_pdf(
                    pdf_path=pdf_path,
                    authority=authority,
                    chunk_chars=chunk_chars,
                    min_chunk_chars=min_chunk_chars,
                )
            )
    return corpus
