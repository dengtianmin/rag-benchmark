from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, PipelineRunRecord, RetrievedDocument
from modules.query_rewriter import QueryRewriter, RewriteMode
from pipelines.base import BasePipeline, MockGenerator, MockReranker, PublicIndex


@dataclass(slots=True)
class RewriteRAGConfig:
    top_k: int = 5
    rerank: bool = True
    mode: RewriteMode = "naive"


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
        reranker: MockReranker | None = None,
        generator: MockGenerator | None = None,
    ) -> None:
        self.index = index
        self.config = config or RewriteRAGConfig()
        self.rewriter = rewriter or QueryRewriter()
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        original_query = sample.question
        rewrite_output = self.rewriter.rewrite(original_query, sample=sample, mode=self.config.mode)
        original_documents = self.index.search(original_query, top_k=self.config.top_k)
        rewritten_documents = self.index.search(rewrite_output.rewritten_query, top_k=self.config.top_k)
        if self.config.rerank:
            if original_documents:
                original_documents = self.reranker.rerank(original_query, original_documents)
            if rewritten_documents:
                rewritten_documents = self.reranker.rerank(rewrite_output.rewritten_query, rewritten_documents)
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
                "rerank_enabled": self.config.rerank,
                "original_query": original_query,
                "retrieved_doc_ids": [doc.source_id for doc in rewritten_documents],
                "retrieved_section_ids": [doc.section_id for doc in rewritten_documents],
                "retrieval_comparison": comparison,
                "generator_metadata": answer_result.metadata,
            },
        )
