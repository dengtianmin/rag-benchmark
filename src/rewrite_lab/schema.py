from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RewriteCandidate:
    question_id: str
    original_question: str
    rewritten_query: str
    strategy_name: str
    question_type: str | None = None
    extracted_entities: list[str] = field(default_factory=list)
    extracted_attributes_or_relations: list[str] = field(default_factory=list)
    extracted_constraints: list[str] = field(default_factory=list)
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalSnapshot:
    query: str
    section_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    hit_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    rank_of_first_gold: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RewriteLabRecord:
    question_id: str
    strategy_name: str
    question_type: str | None
    original_question: str
    rewritten_query: str
    flags: list[str] = field(default_factory=list)
    bucket_tags: list[str] = field(default_factory=list)
    intrinsic_metrics: dict[str, Any] = field(default_factory=dict)
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    original_retrieval: RetrievalSnapshot | None = None
    rewritten_retrieval: RetrievalSnapshot | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class RewriteLabSummary:
    strategy_name: str
    sample_count: int
    metrics: dict[str, Any] = field(default_factory=dict)
    bucket_summary: dict[str, int] = field(default_factory=dict)
    by_question_type: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
