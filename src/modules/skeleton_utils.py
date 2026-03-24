from __future__ import annotations

from core.schema import BenchmarkSample, QuestionSkeletonLabel


def build_question_skeleton(sample: BenchmarkSample) -> QuestionSkeletonLabel:
    """Return the oracle skeleton directly from benchmark fields."""

    return sample.skeleton_label


def flatten_skeleton_to_query_parts(skeleton: QuestionSkeletonLabel) -> dict[str, list[str]]:
    """Convert skeleton fields into explicit retrieval-oriented query parts."""

    return {
        "entities": [item for item in skeleton.entities if item],
        "relations": [item for item in skeleton.relations if item],
        "constraints": [item for item in skeleton.constraints if item],
    }


def build_entity_relation_query(sample: BenchmarkSample) -> tuple[str, dict]:
    """
    Oracle rewrite that injects entities / relations / constraints into the query.
    This is the baseline form of chapter-4 skeleton-driven retrieval rewrite.
    """

    skeleton = build_question_skeleton(sample)
    parts = flatten_skeleton_to_query_parts(skeleton)
    segments = [sample.question]
    if parts["entities"]:
        segments.append("entities: " + ", ".join(parts["entities"]))
    if parts["relations"]:
        segments.append("relations: " + ", ".join(parts["relations"]))
    if parts["constraints"]:
        segments.append("constraints: " + ", ".join(parts["constraints"]))
    query = " | ".join(segments)
    details = {
        "question_type": skeleton.question_type.value if skeleton.question_type else None,
        "entities": parts["entities"],
        "relations": parts["relations"],
        "constraints": parts["constraints"],
        "strategy": "oracle_entity_relation",
    }
    return query, details
