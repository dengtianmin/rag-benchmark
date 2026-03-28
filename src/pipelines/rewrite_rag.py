from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.schema import BenchmarkSample, PipelineRunRecord, RetrievedDocument
from modules.query_rewriter import QueryRewriter, RewriteMode
from pipelines.base import BasePipeline, MockGenerator, MockReranker, PublicIndex
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve


@dataclass(slots=True)
class RewriteRAGConfig:
    top_k: int = 5
    rerank: bool = True
    rerank_top_n: int | None = None
    mode: RewriteMode = "naive"
    retrieval_mode: str = "lexical"
    trace_metadata: dict[str, Any] = field(default_factory=dict)


def compare_retrievals(
    sample: BenchmarkSample,
    original_documents: list[RetrievedDocument],
    rewritten_documents: list[RetrievedDocument],
    *,
    k: int,
) -> dict:
    gold_sections = {item.section_id for item in sample.evidence}
    original_sections = [item.section_id for item in original_documents[:k]]
    rewritten_sections = [item.section_id for item in rewritten_documents[:k]]
    original_hit = bool(gold_sections & set(original_sections))
    rewritten_hit = bool(gold_sections & set(rewritten_sections))
    original_coverage = len(gold_sections & set(original_sections))
    rewritten_coverage = len(gold_sections & set(rewritten_sections))
    return {
        "k": k,
        "gold_section_ids": sorted(gold_sections),
        "original_hit": original_hit,
        "rewritten_hit": rewritten_hit,
        "hit_improved": rewritten_hit and not original_hit,
        "original_gold_coverage": original_coverage,
        "rewritten_gold_coverage": rewritten_coverage,
        "coverage_improved": rewritten_coverage > original_coverage,
        "original_section_ids": original_sections,
        "rewritten_section_ids": rewritten_sections,
    }


def summarize_rewrite_improvements(records: list[PipelineRunRecord]) -> dict[str, float]:
    if not records:
        return {
            "hit_improvement_rate": 0.0,
            "coverage_improvement_rate": 0.0,
            "rewritten_hit_rate": 0.0,
            "original_hit_rate": 0.0,
        }
    comparisons = [record.trace["retrieval_comparison"] for record in records]
    total = len(comparisons)
    return {
        "hit_improvement_rate": sum(1 for item in comparisons if item["hit_improved"]) / total,
        "coverage_improvement_rate": sum(1 for item in comparisons if item["coverage_improved"]) / total,
        "rewritten_hit_rate": sum(1 for item in comparisons if item["rewritten_hit"]) / total,
        "original_hit_rate": sum(1 for item in comparisons if item["original_hit"]) / total,
    }


class RewriteRAGPipeline(BasePipeline):
    method_name = "rewrite_rag"

    def __init__(
        self,
        index: PublicIndex,
        *,
        config: RewriteRAGConfig | None = None,
        rewriter: QueryRewriter | None = None,
        retriever: SupportsRetrieve | None = None,
        reranker: Any | None = None,
        generator: MockGenerator | None = None,
    ) -> None:
        self.index = index
        self.config = config or RewriteRAGConfig()
        self.rewriter = rewriter or QueryRewriter()
        self.retriever = retriever or LexicalTextRetriever(index)
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        original_query = sample.question
        rewrite_output = self.rewriter.rewrite(original_query, sample=sample, mode=self.config.mode)
        original_documents = self.retriever.retrieve(original_query, top_k=self.config.top_k)
        rewritten_documents = self.retriever.retrieve(rewrite_output.rewritten_query, top_k=self.config.top_k)
        initial_dense_candidates = {
            "original": [document.section_id for document in original_documents],
            "rewritten": [document.section_id for document in rewritten_documents],
        }
        original_dense_recall_scores = {
            document.section_id: float(document.metadata.get("dense_score", document.score))
            for document in original_documents
            if document.metadata.get("retriever") in {"qdrant_dense", "hybrid_retriever"}
        }
        rewritten_dense_recall_scores = {
            document.section_id: float(document.metadata.get("dense_score", document.score))
            for document in rewritten_documents
            if document.metadata.get("retriever") in {"qdrant_dense", "hybrid_retriever"}
        }
        original_pre_rerank_documents = [document.model_copy() for document in original_documents]
        rewritten_pre_rerank_documents = [document.model_copy() for document in rewritten_documents]
        if self.config.rerank:
            if original_documents:
                original_documents = self.reranker.rerank(
                    original_query,
                    original_documents,
                    top_n=self.config.rerank_top_n,
                )
            if rewritten_documents:
                rewritten_documents = self.reranker.rerank(
                    rewrite_output.rewritten_query,
                    rewritten_documents,
                    top_n=self.config.rerank_top_n,
                )
        original_rerank_scores = {
            document.section_id: float(document.metadata.get("rerank_score"))
            for document in original_documents
            if "rerank_score" in document.metadata
        }
        rewritten_rerank_scores = {
            document.section_id: float(document.metadata.get("rerank_score"))
            for document in rewritten_documents
            if "rerank_score" in document.metadata
        }
        original_rerank_trace = self.build_rerank_trace(
            reranker=self.reranker if self.config.rerank else None,
            before_documents=original_pre_rerank_documents,
            after_documents=original_documents,
            top_n=self.config.rerank_top_n,
        )
        rewritten_rerank_trace = self.build_rerank_trace(
            reranker=self.reranker if self.config.rerank else None,
            before_documents=rewritten_pre_rerank_documents,
            after_documents=rewritten_documents,
            top_n=self.config.rerank_top_n,
        )
        retrieval_result = self.build_retrieval_result(
            sample.question_id, rewrite_output.rewritten_query, rewritten_documents
        )
        answer_result = self.generator.generate(sample, rewritten_documents)
        comparison = compare_retrievals(
            sample,
            original_documents,
            rewritten_documents,
            k=self.config.top_k,
        )
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=rewrite_output.rewritten_query,
            trace={
                "pipeline": "question -> rewrite -> retrieve -> rerank -> generate",
                "rewrite_mode": self.config.mode,
                "rewrite_details": rewrite_output.details,
                "top_k": self.config.top_k,
                "retrieval_mode": self.config.retrieval_mode,
                "use_rerank": self.config.rerank,
                **self.config.trace_metadata,
                "rerank_enabled": self.config.rerank,
                "original_query": original_query,
                "initial_dense_candidates": initial_dense_candidates,
                "original_dense_recall_scores": original_dense_recall_scores,
                "rewritten_dense_recall_scores": rewritten_dense_recall_scores,
                "rerank_backend": rewritten_rerank_trace["rerank_backend"],
                "rerank_url": rewritten_rerank_trace["rerank_url"],
                "rerank_top_n": rewritten_rerank_trace["rerank_top_n"],
                "original_rerank_input_count": original_rerank_trace["rerank_input_count"],
                "rewritten_rerank_input_count": rewritten_rerank_trace["rerank_input_count"],
                "original_pre_rerank_sections": original_rerank_trace["pre_rerank_sections"],
                "rewritten_pre_rerank_sections": rewritten_rerank_trace["pre_rerank_sections"],
                "original_post_rerank_sections": original_rerank_trace["post_rerank_sections"],
                "rewritten_post_rerank_sections": rewritten_rerank_trace["post_rerank_sections"],
                "original_rerank_score_list": original_rerank_trace["rerank_score_list"],
                "rewritten_rerank_score_list": rewritten_rerank_trace["rerank_score_list"],
                "original_rerank_order_changed": original_rerank_trace["rerank_order_changed"],
                "rewritten_rerank_order_changed": rewritten_rerank_trace["rerank_order_changed"],
                "rerank_fallback_used": bool(
                    original_rerank_trace["rerank_fallback_used"] or rewritten_rerank_trace["rerank_fallback_used"]
                ),
                "rerank_error": rewritten_rerank_trace["rerank_error"] or original_rerank_trace["rerank_error"],
                "original_rerank_scores": original_rerank_scores,
                "rewritten_rerank_scores": rewritten_rerank_scores,
                "retrieved_doc_ids": [doc.source_id for doc in rewritten_documents],
                "retrieved_section_ids": [doc.section_id for doc in rewritten_documents],
                "final_reranked_candidates": [doc.section_id for doc in rewritten_documents],
                "final_kept_sections": [doc.section_id for doc in rewritten_documents],
                "retrieval_comparison": comparison,
                "generator_metadata": answer_result.metadata,
            },
        )
