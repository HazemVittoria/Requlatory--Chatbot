from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int
    chunk_id: str
    doc_title: str = ""


@dataclass(frozen=True)
class Fact:
    quote: str
    pdf: str
    page: int
    chunk_id: str
    score: float
    doc_title: str = ""


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
