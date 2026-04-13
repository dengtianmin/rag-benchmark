from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.schema import QuestionSkeletonLabel


@dataclass(slots=True)
class SkeletonMetricCounts:
    predicted: int = 0
    gold: int = 0
    matched: int = 0


def normalize_skeleton_terms(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).strip().split()).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def label_to_sets(label: QuestionSkeletonLabel) -> dict[str, set[str]]:
    return {
        "entities": set(normalize_skeleton_terms(label.entities)),
        "relations": set(normalize_skeleton_terms(label.relations)),
        "constraints": set(normalize_skeleton_terms(label.constraints)),
    }


def compute_sample_metrics(prediction: QuestionSkeletonLabel, gold: QuestionSkeletonLabel) -> dict[str, object]:
    pred_sets = label_to_sets(prediction)
    gold_sets = label_to_sets(gold)

    result: dict[str, object] = {}
    exact_match = True
    for field in ("entities", "relations", "constraints"):
        pred_set = pred_sets[field]
        gold_set = gold_sets[field]
        matched = pred_set & gold_set
        precision = len(matched) / len(pred_set) if pred_set else 0.0
        recall = len(matched) / len(gold_set) if gold_set else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        result[field] = {
            "predicted": sorted(pred_set),
            "gold": sorted(gold_set),
            "matched": sorted(matched),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": len(pred_set),
            "gold_count": len(gold_set),
            "matched_count": len(matched),
        }
        if pred_set != gold_set:
            exact_match = False
    result["skeleton_exact_match"] = 1.0 if exact_match else 0.0
    return result


def aggregate_micro(sample_metric_rows: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        "entities": SkeletonMetricCounts(),
        "relations": SkeletonMetricCounts(),
        "constraints": SkeletonMetricCounts(),
    }
    exact_matches = 0.0
    total = len(sample_metric_rows)

    for row in sample_metric_rows:
        exact_matches += float(row["skeleton_exact_match"])
        for field in ("entities", "relations", "constraints"):
            payload = row[field]
            counts[field].predicted += int(payload["predicted_count"])
            counts[field].gold += int(payload["gold_count"])
            counts[field].matched += int(payload["matched_count"])

    summary: dict[str, object] = {
        "sample_count": total,
        "averaging": "micro",
        "skeleton_exact_match": exact_matches / total if total else 0.0,
    }
    for field, field_counts in counts.items():
        precision = field_counts.matched / field_counts.predicted if field_counts.predicted else 0.0
        recall = field_counts.matched / field_counts.gold if field_counts.gold else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        summary[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": field_counts.predicted,
            "gold_count": field_counts.gold,
            "matched_count": field_counts.matched,
        }
    return summary


def aggregate_by_question_type(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    return {question_type: aggregate_micro(items) for question_type, items in grouped.items()}
