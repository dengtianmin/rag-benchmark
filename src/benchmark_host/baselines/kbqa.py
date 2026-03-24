from __future__ import annotations

from benchmark_host.baselines.base import BaselineSystem
from benchmark_host.components.generator import StubAnswerGenerator
from benchmark_host.components.retrievers import RelationDrivenRetriever
from benchmark_host.components.rewriters import SkeletonExtractionComponent
from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class KBQABaseline(BaselineSystem):
    name = "kbqa_baseline"

    def __init__(
        self,
        skeleton_extractor: SkeletonExtractionComponent,
        relation_retriever: RelationDrivenRetriever,
        generator: StubAnswerGenerator,
    ) -> None:
        self.skeleton_extractor = skeleton_extractor
        self.relation_retriever = relation_retriever
        self.generator = generator

    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        skeleton = self.skeleton_extractor.run(sample).data
        retrieved = self.relation_retriever.run(sample, query=skeleton["query"]).data["retrieved"]
        generation = self.generator.run(sample, retrieved).data
        return ExperimentPrediction(
            qid=sample.qid,
            system_name=self.name,
            question=sample.question,
            answer=generation["answer"],
            gold_answer=sample.answer_short,
            retrieved=retrieved,
            evidence=generation["evidence"],
            trace={
                "generate_then_retrieve": True,
                "logical_skeleton": skeleton["skeleton"],
                "todo": "Replace stub skeleton with executable logical form planner.",
                **generation["notes"],
            },
        )
