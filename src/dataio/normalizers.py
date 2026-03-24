from __future__ import annotations

from typing import Any

from core.schema import BenchmarkSample, EvidenceItem
from core.types import QuestionType, SourceScope


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def normalize_evidence_items(items: Any) -> list[EvidenceItem]:
    normalized: list[EvidenceItem] = []
    for item in items or []:
        if isinstance(item, EvidenceItem):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(EvidenceItem.from_raw(item))
        else:
            raise TypeError(f"Unsupported evidence item type: {type(item)!r}")
    return normalized


def normalize_benchmark_sample(record: dict[str, Any]) -> BenchmarkSample:
    """Normalize benchmark record from current dataset schema to internal protocol."""

    question_id = record.get("question_id") or record.get("qid")
    source_id = record.get("source_id") or record.get("doc_id") or ""
    section_id = record.get("section_id") or record.get("block_id") or ""

    payload = {
        "question_id": str(question_id or ""),
        "question": str(record.get("question", "")).strip(),
        "answer_short": str(record.get("answer_short", "")).strip(),
        "answer_long": str(record.get("answer_long", "")).strip(),
        "question_type": QuestionType(record.get("question_type", QuestionType.FACT.value)),
        "entities": _string_list(record.get("entities")),
        "relations": _string_list(record.get("relations")),
        "constraints": _string_list(record.get("constraints")),
        "evidence": normalize_evidence_items(record.get("evidence", [])),
        "source_scope": SourceScope(record.get("source_scope", SourceScope.SINGLE_SECTION.value)),
        "requires_text_compensation": bool(record.get("requires_text_compensation", False)),
        "source_id": str(source_id).strip(),
        "section_id": str(section_id).strip(),
        "support_score": record.get("support_score"),
        "completeness_score": record.get("completeness_score"),
        "raw_record": dict(record),
    }
    return BenchmarkSample(**payload)

