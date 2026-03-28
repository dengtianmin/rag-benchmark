from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.schema import BenchmarkSample, PipelineRunRecord
from pipelines.base import BasePipeline, MockGenerator, MockReranker, PublicIndex
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve


@dataclass(slots=True)
class TraditionalRAGConfig:
    top_k: int = 5
    rerank: bool = True
    rerank_top_n: int | None = None
    retrieval_mode: str = "lexical"
    trace_metadata: dict[str, Any] = field(default_factory=dict)


class TraditionalRAGPipeline(BasePipeline):
    method_name = "traditional_rag"

    def __init__(
        self,
        index: PublicIndex,
        *,
        config: TraditionalRAGConfig | None = None,
        retriever: SupportsRetrieve | None = None,
        reranker: Any | None = None,
        generator: MockGenerator | None = None,
    ) -> None:
        self.index = index
        self.config = config or TraditionalRAGConfig()
        self.retriever = retriever or LexicalTextRetriever(index)
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        query = sample.question
        retrieved_documents = self.retriever.retrieve(query, top_k=self.config.top_k)
        initial_dense_candidates = [document.section_id for document in retrieved_documents]
        dense_recall_scores = {
            document.section_id: float(document.metadata.get("dense_score", document.score))
            for document in retrieved_documents
            if document.metadata.get("retriever") in {"qdrant_dense", "hybrid_retriever"}
        }
        pre_rerank_documents = [document.model_copy() for document in retrieved_documents]
        if self.config.rerank and retrieved_documents:
            retrieved_documents = self.reranker.rerank(query, retrieved_documents, top_n=self.config.rerank_top_n)
        rerank_scores = {
            document.section_id: float(document.metadata.get("rerank_score"))
            for document in retrieved_documents
            if "rerank_score" in document.metadata
        }
        rerank_trace = self.build_rerank_trace(
            reranker=self.reranker if self.config.rerank else None,
            before_documents=pre_rerank_documents,
            after_documents=retrieved_documents,
            top_n=self.config.rerank_top_n,
        )
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
                "retrieval_mode": self.config.retrieval_mode,
                "use_rerank": self.config.rerank,
                **self.config.trace_metadata,
                "rerank_enabled": self.config.rerank,
                "initial_dense_candidates": initial_dense_candidates,
                "dense_recall_scores": dense_recall_scores,
                **rerank_trace,
                "rerank_scores": rerank_scores,
                "retrieved_doc_ids": [doc.source_id for doc in retrieved_documents],
                "retrieved_section_ids": [doc.section_id for doc in retrieved_documents],
                "final_reranked_candidates": [doc.section_id for doc in retrieved_documents],
                "final_kept_sections": [doc.section_id for doc in retrieved_documents],
                "generator_metadata": answer_result.metadata,
            },
        )
