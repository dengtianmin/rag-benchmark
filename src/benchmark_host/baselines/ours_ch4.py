from __future__ import annotations

from benchmark_host.baselines.base import BaselineSystem
from benchmark_host.components.generator import StubAnswerGenerator, TextEvidenceCompensator
from benchmark_host.components.retrievers import RelationDrivenRetriever, TraditionalRetriever
from benchmark_host.components.rewriters import SkeletonExtractionComponent
from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class OursCh4Baseline(BaselineSystem):
    name = "ours_ch4"

    def __init__(
        self,
        skeleton_extractor: SkeletonExtractionComponent,
        relation_retriever: RelationDrivenRetriever,
        text_retriever: TraditionalRetriever,
        compensator: TextEvidenceCompensator,
        generator: StubAnswerGenerator,
    ) -> None:
        self.skeleton_extractor = skeleton_extractor
        self.relation_retriever = relation_retriever
        self.text_retriever = text_retriever
        self.compensator = compensator
        self.generator = generator

    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        skeleton = self.skeleton_extractor.run(sample).data
        relation_hits = self.relation_retriever.run(sample, query=skeleton["query"]).data["retrieved"]
        text_hits = self.text_retriever.run(sample, query=sample.question).data["retrieved"]
        compensated = self.compensator.run(sample, primary=relation_hits, fallback=text_hits).data
        generation = self.generator.run(sample, compensated["retrieved"]).data
        return ExperimentPrediction(
            qid=sample.qid,
            system_name=self.name,
            question=sample.question,
            answer=generation["answer"],
            gold_answer=sample.answer_short,
            retrieved=compensated["retrieved"],
            evidence=generation["evidence"],
            trace={
                "relation_driven_retrieval": True,
                "skeleton_extraction": skeleton["skeleton"],
                "text_evidence_compensation": compensated["activated"],
                "todo": "Connect real retrieval rewrite and evidence compensation policy.",
                **generation["notes"],
            },
        )
