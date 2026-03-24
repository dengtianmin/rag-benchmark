from __future__ import annotations

from benchmark_host.baselines.base import BaselineSystem
from benchmark_host.components.generator import StubAnswerGenerator
from benchmark_host.components.retrievers import TraditionalRetriever
from benchmark_host.components.rewriters import HeuristicRewriteComponent
from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class RewriteRAGBaseline(BaselineSystem):
    name = "rewrite_rag"

    def __init__(
        self,
        rewriter: HeuristicRewriteComponent,
        retriever: TraditionalRetriever,
        generator: StubAnswerGenerator,
    ) -> None:
        self.rewriter = rewriter
        self.retriever = retriever
        self.generator = generator

    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        rewrite = self.rewriter.run(sample).data
        retrieved = self.retriever.run(sample, query=rewrite["query"]).data["retrieved"]
        generation = self.generator.run(sample, retrieved).data
        return ExperimentPrediction(
            qid=sample.qid,
            system_name=self.name,
            question=sample.question,
            answer=generation["answer"],
            gold_answer=sample.answer_short,
            retrieved=retrieved,
            evidence=generation["evidence"],
            trace={"rewrite_query": rewrite["query"], "skeleton": rewrite["skeleton"], **generation["notes"]},
        )
