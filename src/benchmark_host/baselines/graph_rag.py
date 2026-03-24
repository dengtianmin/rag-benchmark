from __future__ import annotations

from benchmark_host.baselines.base import BaselineSystem
from benchmark_host.components.generator import StubAnswerGenerator
from benchmark_host.components.retrievers import GraphRetriever, TraditionalRetriever
from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class GraphEnhancedRAGBaseline(BaselineSystem):
    name = "graph_enhanced_rag"

    def __init__(
        self,
        seed_retriever: TraditionalRetriever,
        graph_retriever: GraphRetriever,
        generator: StubAnswerGenerator,
    ) -> None:
        self.seed_retriever = seed_retriever
        self.graph_retriever = graph_retriever
        self.generator = generator

    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        seeds = self.seed_retriever.run(sample).data["retrieved"]
        retrieved = self.graph_retriever.run(sample, seeds=seeds).data["retrieved"]
        generation = self.generator.run(sample, retrieved).data
        return ExperimentPrediction(
            qid=sample.qid,
            system_name=self.name,
            question=sample.question,
            answer=generation["answer"],
            gold_answer=sample.answer_short,
            retrieved=retrieved,
            evidence=generation["evidence"],
            trace={"seed_count": len(seeds), "graph_expand": True, **generation["notes"]},
        )
