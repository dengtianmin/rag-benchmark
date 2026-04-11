from __future__ import annotations

from evaluation.common import mean
from modules.skeleton_extractor import SkeletonExtractionResult


def _normalized_entities(skeleton: SkeletonExtractionResult) -> set[str]:
    return {item.normalized_name for item in skeleton.structured_entities if item.normalized_name}


def _normalized_relations(skeleton: SkeletonExtractionResult) -> set[str]:
    values = set()
    for item in skeleton.structured_relations:
        if item.relation_type and item.relation_type != "factoid":
            values.add(item.relation_type)
        elif item.normalized_name:
            values.add(item.normalized_name)
    return values


def _normalized_constraints(skeleton: SkeletonExtractionResult) -> set[str]:
    return {item.normalized_value for item in skeleton.structured_constraints if item.normalized_value}


def _precision(reference: set[str], candidate: set[str]) -> float:
    if not candidate:
        return 0.0
    return len(reference & candidate) / len(candidate)


def _recall(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def evaluate_rewrite_diagnostics(
    original_skeleton: SkeletonExtractionResult,
    rewritten_skeleton: SkeletonExtractionResult,
) -> dict[str, float | bool]:
    original_entities = _normalized_entities(original_skeleton)
    rewritten_entities = _normalized_entities(rewritten_skeleton)
    original_relations = _normalized_relations(original_skeleton)
    rewritten_relations = _normalized_relations(rewritten_skeleton)
    original_constraints = _normalized_constraints(original_skeleton)
    rewritten_constraints = _normalized_constraints(rewritten_skeleton)

    # This semantic drift label is a heuristic proxy for structure loss, not human-annotated gold.
    empty_rewrite_structure = not (rewritten_entities or rewritten_relations or rewritten_constraints)
    had_original_structure = bool(original_entities or original_relations or original_constraints)
    semantic_drift = bool(
        (original_entities and not (original_entities & rewritten_entities))
        or (original_relations and not (original_relations & rewritten_relations))
        or (original_constraints and not (original_constraints & rewritten_constraints))
        or (had_original_structure and empty_rewrite_structure)
    )

    return {
        "entity_retention_precision": _precision(original_entities, rewritten_entities),
        "entity_retention_recall": _recall(original_entities, rewritten_entities),
        "relation_retention_precision": _precision(original_relations, rewritten_relations),
        "relation_retention_recall": _recall(original_relations, rewritten_relations),
        "constraint_retention_precision": _precision(original_constraints, rewritten_constraints),
        "constraint_retention_recall": _recall(original_constraints, rewritten_constraints),
        "semantic_drift": semantic_drift,
    }


def aggregate_rewrite_diagnostics(metrics_list: list[dict[str, float | bool]]) -> dict[str, float]:
    if not metrics_list:
        return {
            "entity_retention_precision": 0.0,
            "entity_retention_recall": 0.0,
            "relation_retention_precision": 0.0,
            "relation_retention_recall": 0.0,
            "constraint_retention_precision": 0.0,
            "constraint_retention_recall": 0.0,
            "semantic_drift_rate": 0.0,
        }
    return {
        "entity_retention_precision": mean([float(item["entity_retention_precision"]) for item in metrics_list]),
        "entity_retention_recall": mean([float(item["entity_retention_recall"]) for item in metrics_list]),
        "relation_retention_precision": mean([float(item["relation_retention_precision"]) for item in metrics_list]),
        "relation_retention_recall": mean([float(item["relation_retention_recall"]) for item in metrics_list]),
        "constraint_retention_precision": mean([float(item["constraint_retention_precision"]) for item in metrics_list]),
        "constraint_retention_recall": mean([float(item["constraint_retention_recall"]) for item in metrics_list]),
        "semantic_drift_rate": mean([1.0 if bool(item["semantic_drift"]) else 0.0 for item in metrics_list]),
    }
