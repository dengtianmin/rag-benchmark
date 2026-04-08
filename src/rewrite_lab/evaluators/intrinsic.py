from __future__ import annotations

from core.schema import BenchmarkSample
from pipelines.base import tokenize

from rewrite_lab.schema import RewriteCandidate
from rewrite_lab.utils import build_reference_fields, count_overlap, normalize_phrase, token_counter


def _rate(hit: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return hit / total


def evaluate_intrinsic_metrics(sample: BenchmarkSample, candidate: RewriteCandidate) -> dict[str, float | bool]:
    reference = build_reference_fields(sample)
    entity_hit, entity_total = count_overlap(reference["entities"], candidate.rewritten_query)
    relation_hit, relation_total = count_overlap(reference["attributes_or_relations"], candidate.rewritten_query)
    constraint_hit, constraint_total = count_overlap(reference["constraints"], candidate.rewritten_query)

    original_counter = token_counter(sample.question)
    rewritten_counter = token_counter(candidate.rewritten_query)
    original_vocab = set(original_counter)
    rewritten_vocab = set(rewritten_counter)
    normalized_original = normalize_phrase(sample.question)
    new_tokens = {token for token in rewritten_vocab if token not in original_vocab and token not in normalized_original}
    noise_rate = len(new_tokens) / max(len(rewritten_vocab), 1)

    parse_success = bool(candidate.rewritten_query.strip()) and "malformed" not in candidate.flags

    return {
        "entity_preservation_rate": _rate(entity_hit, entity_total),
        "attribute_or_relation_preservation_rate": _rate(relation_hit, relation_total),
        "constraint_preservation_rate": _rate(constraint_hit, constraint_total),
        "noise_rate": noise_rate,
        "compression_ratio": len(candidate.rewritten_query) / max(len(sample.question), 1),
        "rewrite_length": float(len(tokenize(candidate.rewritten_query))),
        "parse_success": parse_success,
        "parseable_rate": 1.0 if parse_success else 0.0,
    }


def aggregate_intrinsic_metrics(metric_rows: list[dict[str, float | bool]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = list(metric_rows[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in metric_rows]
        aggregated[key] = sum(values) / len(values)
    return aggregated
