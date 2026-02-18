from __future__ import annotations

from pathlib import Path

import pytest

from src import index_integrity as ii


def test_index_integrity_report_success(monkeypatch):
    monkeypatch.setattr(ii, "rebuild_index", lambda **kwargs: None)
    monkeypatch.setattr(
        ii,
        "_expected_documents",
        lambda data_dir: [
            ("ich/ICH_Q13_Step4_Guideline_2022_1116.pdf", "ICH", "ICH_Q13_Step4_Guideline_2022_1116.pdf"),
            ("fda/Data Integrity and Compliance_cGMP.pdf", "FDA", "Data Integrity and Compliance_cGMP.pdf"),
        ],
    )
    monkeypatch.setattr(
        ii.search_module,
        "_CORPUS",
        [
            {"authority": "ICH", "file": "ICH_Q13_Step4_Guideline_2022_1116.pdf"},
            {"authority": "ICH", "file": "ICH_Q13_Step4_Guideline_2022_1116.pdf"},
            {"authority": "FDA", "file": "Data Integrity and Compliance_cGMP.pdf"},
        ],
        raising=False,
    )

    report = ii.validate_index_integrity(data_dir=Path("data"), print_report=False)
    assert report.total_documents == 2
    assert report.total_chunks == 3
    assert report.chunks_per_document["ich/ICH_Q13_Step4_Guideline_2022_1116.pdf"] == 2
    assert report.chunks_per_document["fda/Data Integrity and Compliance_cGMP.pdf"] == 1
    assert len(report.index_hash) == 64


def test_index_integrity_fails_on_zero_chunk_pdf(monkeypatch):
    monkeypatch.setattr(ii, "rebuild_index", lambda **kwargs: None)
    monkeypatch.setattr(
        ii,
        "_expected_documents",
        lambda data_dir: [("ich/ICH_Q13_Step4_Guideline_2022_1116.pdf", "ICH", "ICH_Q13_Step4_Guideline_2022_1116.pdf")],
    )
    monkeypatch.setattr(ii.search_module, "_CORPUS", [], raising=False)

    with pytest.raises(RuntimeError, match="zero chunks"):
        ii.validate_index_integrity(data_dir=Path("data"), print_report=False)


def test_index_integrity_fails_if_no_expected_pdfs(monkeypatch):
    monkeypatch.setattr(ii, "rebuild_index", lambda **kwargs: None)
    monkeypatch.setattr(ii, "_expected_documents", lambda data_dir: [])
    monkeypatch.setattr(ii.search_module, "_CORPUS", [], raising=False)

    with pytest.raises(RuntimeError, match="no expected PDF files"):
        ii.validate_index_integrity(data_dir=Path("data"), print_report=False)

