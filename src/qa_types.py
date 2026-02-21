from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def display_page_value(
    *,
    pdf_index: int | None,
    page_label: str | None,
) -> str:
    label = str(page_label or "").strip()
    if label:
        return label
    try:
        idx = int(pdf_index) if pdf_index is not None else None
    except Exception:
        idx = None
    if idx is not None and idx >= 0:
        return str(idx + 1)
    return "?"


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int
    chunk_id: str
    doc_title: str = ""
    pdf_index: int | None = None
    page_label: str | None = None

    def __post_init__(self) -> None:
        page = int(self.page or 0)
        try:
            idx = int(self.pdf_index) if self.pdf_index is not None else None
        except Exception:
            idx = None
        if idx is None:
            idx = (page - 1) if page > 0 else 0
        if idx < 0:
            idx = 0
        if page <= 0:
            page = idx + 1
        label = str(self.page_label or "").strip() or None
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "pdf_index", idx)
        object.__setattr__(self, "page_label", label)

    @property
    def display_page(self) -> str:
        return display_page_value(
            pdf_index=self.pdf_index,
            page_label=self.page_label,
        )


@dataclass(frozen=True)
class Fact:
    quote: str
    pdf: str
    page: int
    chunk_id: str
    score: float
    doc_title: str = ""
    pdf_index: int | None = None
    page_label: str | None = None

    def __post_init__(self) -> None:
        page = int(self.page or 0)
        try:
            idx = int(self.pdf_index) if self.pdf_index is not None else None
        except Exception:
            idx = None
        if idx is None:
            idx = (page - 1) if page > 0 else 0
        if idx < 0:
            idx = 0
        if page <= 0:
            page = idx + 1
        label = str(self.page_label or "").strip() or None
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "pdf_index", idx)
        object.__setattr__(self, "page_label", label)

    @property
    def display_page(self) -> str:
        return display_page_value(
            pdf_index=self.pdf_index,
            page_label=self.page_label,
        )


@dataclass(frozen=True)
class AnswerResult:
    text: str
    intent: str
    scope: str
    phase_a_fallback_used: bool = False
    debug_trace: dict[str, Any] | None = None
    citations: list[Citation] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    relevant_facts: list[Fact] = field(default_factory=list)
