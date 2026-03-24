from __future__ import annotations

from dataclasses import dataclass

from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction
from benchmark_host.utils.text import exact_match, token_f1


@dataclass(slots=True)
class EvaluationRow:
    qid: str
    system_name: str
    exact_match: float
    token_f1: float
    evidence_hit: float
    retrieval_hit: float


class Evaluator:
    def evaluate(self, sample: BenchmarkSample, prediction: ExperimentPrediction) -> EvaluationRow:
        gold_sections = {item.section_id for item in sample.evidence}
        pred_sections = {item.section_id for item in prediction.evidence}
        retrieved_sections = {item.section_id for item in prediction.retrieved}
        evidence_hit = 1.0 if gold_sections and gold_sections & pred_sections else 0.0
        retrieval_hit = 1.0 if gold_sections and gold_sections & retrieved_sections else 0.0
        return EvaluationRow(
            qid=sample.qid,
            system_name=prediction.system_name,
            exact_match=exact_match(prediction.answer, sample.answer_short),
            token_f1=token_f1(prediction.answer, sample.answer_short),
            evidence_hit=evidence_hit,
            retrieval_hit=retrieval_hit,
        )
