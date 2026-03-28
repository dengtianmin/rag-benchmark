from __future__ import annotations

from core.schema import PipelineRunRecord
from evaluation.common import exact_match, mean, normalize_text, token_f1


def accuracy(prediction: str, gold_answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_gold = {normalize_text(item) for item in gold_answers if item.strip()}
    return 1.0 if normalized_prediction in normalized_gold else 0.0


def evaluate_answer_record(record: PipelineRunRecord) -> dict[str, float]:
    # Design choice: answer metrics are computed only from normalized answer fields.
    # We intentionally do not evaluate raw JSON output or supporting_evidence text,
    # so evidence rendering changes do not destabilize EM / Accuracy / Token-F1.
    prediction = record.answer.answer_text or record.answer.answer_short or record.answer.answer_long
    gold_answers = [record.sample.answer_short, record.sample.answer_long]
    primary_gold = record.sample.answer_short or record.sample.answer_long
    return {
        "em": exact_match(prediction, primary_gold),
        "token_f1": token_f1(prediction, primary_gold),
        "accuracy": accuracy(prediction, gold_answers),
    }


def aggregate_answer_metrics(records: list[PipelineRunRecord]) -> dict[str, float]:
    per_record = [evaluate_answer_record(record) for record in records]
    return {
        "em": mean([item["em"] for item in per_record]),
        "token_f1": mean([item["token_f1"] for item in per_record]),
        "accuracy": mean([item["accuracy"] for item in per_record]),
    }
