from __future__ import annotations

from benchmark_host.baselines.base import BaselineSystem
from benchmark_host.components.generator import StubAnswerGenerator
from benchmark_host.components.retrievers import TraditionalRetriever
from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class TraditionalRAGBaseline(BaselineSystem):
    name = "traditional_rag"

    def __init__(self, retriever: TraditionalRetriever, generator: StubAnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        retrieved = self.retriever.run(sample).data["retrieved"]
        generation = self.generator.run(sample, retrieved).data
        return ExperimentPrediction(
            qid=sample.qid,
            system_name=self.name,
            question=sample.question,
            answer=generation["answer"],
            gold_answer=sample.answer_short,
            retrieved=retrieved,
            evidence=generation["evidence"],
            trace={"stage": "retrieve_then_read", **generation["notes"]},
        )
