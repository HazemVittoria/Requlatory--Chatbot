from __future__ import annotations

from pathlib import Path

import pytest

from src import ingestion
from src import search as search_module
from src.qa_types import Citation, Fact


def test_display_page_prefers_label():
    c = Citation(doc_id="x.pdf", page=12, chunk_id="p11_c1", pdf_index=11, page_label="A-3")
    assert c.display_page == "A-3"

    f = Fact(quote="q", pdf="x.pdf", page=12, chunk_id="p11_c1", score=0.5, pdf_index=11)
    assert f.display_page == "12"


def test_search_chunk_migration_from_legacy_page():
    old = {"file": "x.pdf", "page": 7, "chunk_id": "p7_c1", "text": "abc"}
    migrated = search_module._migrate_chunk(old)
    assert migrated["pdf_index"] == 7
    assert migrated["page"] == 7
    assert migrated["page_label"] is None
    assert migrated["chunk_id"] == "p7_c1"


def test_search_chunk_migration_repairs_mismatched_chunk_id():
    old = {"file": "x.pdf", "pdf_index": 6, "page": 7, "chunk_id": "p7_c3", "text": "abc"}
    migrated = search_module._migrate_chunk(old)
    assert migrated["pdf_index"] == 6
    assert migrated["chunk_id"] == "p6_c3"


def test_ingest_pdf_includes_pdf_index_and_page_label(monkeypatch, tmp_path: Path):
    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str):
            self.pages = [_FakePage("Scope\nThe firm shall maintain procedures.")]
            self.page_labels = ["iv"]
            self.metadata = None

    monkeypatch.setattr(ingestion, "PdfReader", _FakeReader)
    pdf_path = tmp_path / "x.pdf"

    chunks = ingestion.ingest_pdf(
        pdf_path=pdf_path,
        authority="FDA",
        chunk_chars=1000,
        min_chunk_chars=1,
    )
    assert len(chunks) >= 1
    first = chunks[0]
    assert first["pdf_index"] == 0
    assert first["page"] == 1
    assert first["page_label"] == "iv"
    assert first["chunk_id"] == "p0_c1"
    assert int(str(first["chunk_id"]).split("_", 1)[0].lstrip("p")) == int(first["pdf_index"])

    rows = ingestion.inspect_pdf_page_labels(pdf_path, limit=10)
    assert rows == [(0, 1, "iv")]


def test_q9_risk_management_definition_uses_printed_page_not_physical_index(monkeypatch, tmp_path: Path):
    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str):
            self.pages = [
                _FakePage("Intro page"),
                _FakePage("Intro page 2"),
                _FakePage("Intro page 3"),
                _FakePage("Intro page 4"),
                _FakePage("Intro page 5"),
                _FakePage("Intro page 6"),
                _FakePage(
                    "ICH Q9 Guideline\n"
                    "Step 4\n"
                    "73\n"
                    "3\n"
                    "Quality risk management shall be applied across the product lifecycle."
                ),
            ]
            self.page_labels = [str(i + 1) for i in range(len(self.pages))]
            self.metadata = None

    monkeypatch.setattr(ingestion, "PdfReader", _FakeReader)
    pdf_path = tmp_path / "Q9.pdf"

    chunks = ingestion.ingest_pdf(
        pdf_path=pdf_path,
        authority="ICH",
        chunk_chars=1000,
        min_chunk_chars=1,
    )
    target = next(c for c in chunks if int(c["pdf_index"]) == 6)
    assert target["page"] == 7
    assert target["page_label"] == "3"
    assert str(target["chunk_id"]).startswith("p6_")
    assert int(str(target["chunk_id"]).split("_", 1)[0].lstrip("p")) == int(target["pdf_index"])

    cit = Citation(
        doc_id=str(target["file"]),
        page=int(target["page"]),
        chunk_id=str(target["chunk_id"]),
        pdf_index=int(target["pdf_index"]),
        page_label=str(target["page_label"] or ""),
    )
    assert cit.display_page == "3"
    assert cit.display_page != "7"

    rows = ingestion.inspect_pdf_page_labels(pdf_path, limit=10)
    assert rows[6] == (6, 7, "3")


def test_cross_page_smoothing_fixes_numeric_outlier(monkeypatch, tmp_path: Path):
    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str):
            self.pages = [
                _FakePage("ICH Q9 Guideline\n2\nAlpha"),
                _FakePage("ICH Q9 Guideline\n73\nBeta"),
                _FakePage("ICH Q9 Guideline\n4\nGamma"),
            ]
            self.page_labels = [str(i + 1) for i in range(len(self.pages))]
            self.metadata = None

    monkeypatch.setattr(ingestion, "PdfReader", _FakeReader)
    pdf_path = tmp_path / "Q9.pdf"

    rows = ingestion.inspect_pdf_page_labels(pdf_path, limit=10)
    assert rows[1] == (1, 2, "3")


def test_footer_candidate_beats_top_section_number(monkeypatch, tmp_path: Path):
    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, _path: str):
            self.pages = [
                _FakePage(
                    "\n".join(
                        [
                            "WHO Guideline on data integrity",
                            "3",
                            "Glossary",
                            "Definitions and terms",
                            "More text line",
                            "Even more text line",
                            "137",
                        ]
                    )
                )
            ]
            self.page_labels = ["1"]
            self.metadata = None

    monkeypatch.setattr(ingestion, "PdfReader", _FakeReader)
    pdf_path = tmp_path / "trs.pdf"
    rows = ingestion.inspect_pdf_page_labels(pdf_path, limit=10)
    assert rows == [(0, 1, "137")]


def test_who_trs_glossary_page_prefers_footer_137_not_header_3():
    pdf_path = Path("data") / "who" / "trs1033-annex4-guideline-on-data-integrity.pdf"
    if not pdf_path.exists():
        pytest.skip("WHO TRS fixture PDF not present in local data directory.")

    reader = ingestion.PdfReader(str(pdf_path))
    target_idx = -1
    for i, p in enumerate(reader.pages):
        raw = p.extract_text() or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if ("glossary" in raw.lower()) and ("137" in lines):
            target_idx = i
            break

    assert target_idx >= 0, "Could not find expected glossary page with printed page 137."

    rows = ingestion.inspect_pdf_page_labels(pdf_path, limit=target_idx + 1)
    _, _, label = rows[target_idx]
    assert label == "137"

    cit = Citation(
        doc_id=pdf_path.name,
        page=target_idx + 1,
        chunk_id=f"p{target_idx}_c1",
        pdf_index=target_idx,
        page_label=label,
    )
    assert cit.display_page == "137"
