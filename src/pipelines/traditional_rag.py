from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, PipelineRunRecord
from pipelines.base import BasePipeline, MockGenerator, MockReranker, PublicIndex


@dataclass(slots=True)
class TraditionalRAGConfig:
    top_k: int = 5
    rerank: bool = True


class TraditionalRAGPipeline(BasePipeline):
    method_name = "traditional_rag"

    def __init__(
        self,
        index: PublicIndex,
        *,
        config: TraditionalRAGConfig | None = None,
        reranker: MockReranker | None = None,
        generator: MockGenerator | None = None,
    ) -> None:
        self.index = index
        self.config = config or TraditionalRAGConfig()
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        query = sample.question
        retrieved_documents = self.index.search(query, top_k=self.config.top_k)
        if self.config.rerank and retrieved_documents:
            retrieved_documents = self.reranker.rerank(query, retrieved_documents)
        retrieval_result = self.build_retrieval_result(sample.question_id, query, retrieved_documents)
        answer_result = self.generator.generate(sample, retrieved_documents)
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=None,
            trace={
                "pipeline": "question -> retriever top-k -> optional rerank -> build context -> generator answer",
                "top_k": self.config.top_k,
                "rerank_enabled": self.config.rerank,
                "retrieved_doc_ids": [doc.source_id for doc in retrieved_documents],
                "retrieved_section_ids": [doc.section_id for doc in retrieved_documents],
                "generator_metadata": answer_result.metadata,
            },
        )
