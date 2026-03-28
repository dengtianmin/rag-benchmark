from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.schema import BenchmarkSample, PipelineRunRecord
from evaluation.retrieval_metrics import hit_at_k
from pipelines.base import BasePipeline, MockGenerator, MockReranker, SupportsGenerate
from retrievers.graph_retriever import GraphRetriever, GraphRetrieverConfig


@dataclass(slots=True)
class GraphEnhancedRAGConfig:
    seed_top_k: int = 5
    expand_k: int = 5
    final_top_k: int = 5
    rerank: bool = True
    rerank_top_n: int | None = None
    use_gold_hints: bool = True
    retrieval_mode: str = "lexical"
    trace_metadata: dict[str, Any] = field(default_factory=dict)


def summarize_graph_retrieval(records: list[PipelineRunRecord]) -> dict[str, float]:
    if not records:
        return {"expanded_count": 0.0, "seed_hit_rate": 0.0, "post_expand_hit_rate": 0.0}
    total = len(records)
    return {
        "expanded_count": sum(record.trace["graph_expansion"]["expanded_count"] for record in records) / total,
        "seed_hit_rate": sum(1 for record in records if record.trace["graph_expansion"]["seed_hit"]) / total,
        "post_expand_hit_rate": sum(1 for record in records if record.trace["graph_expansion"]["post_expand_hit"]) / total,
    }


class GraphEnhancedRAGPipeline(BasePipeline):
    method_name = "graph_enhanced_rag"

    def __init__(
        self,
        retriever: GraphRetriever,
        *,
        config: GraphEnhancedRAGConfig | None = None,
        reranker: Any | None = None,
        generator: SupportsGenerate | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or GraphEnhancedRAGConfig()
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        retrieve_output = self.retriever.retrieve(sample)
        final_documents = retrieve_output.organized.final_documents
        initial_dense_candidates = [document.section_id for document in retrieve_output.seed_documents]
        seed_dense_scores = {
            document.section_id: float(document.metadata.get("dense_score", document.score))
            for document in retrieve_output.seed_documents
            if document.metadata.get("retriever") in {"qdrant_dense", "hybrid_retriever"}
        }
        pre_rerank_documents = [document.model_copy() for document in final_documents]
        if self.config.rerank and final_documents:
            final_documents = self.reranker.rerank(sample.question, final_documents, top_n=self.config.rerank_top_n)
        rerank_scores = {
            document.section_id: float(document.metadata.get("rerank_score"))
            for document in final_documents
            if "rerank_score" in document.metadata
        }
        rerank_trace = self.build_rerank_trace(
            reranker=self.reranker if self.config.rerank else None,
            before_documents=pre_rerank_documents,
            after_documents=final_documents,
            top_n=self.config.rerank_top_n,
        )
        retrieval_result = self.build_retrieval_result(sample.question_id, sample.question, final_documents)
        answer_result = self.generator.generate(sample, final_documents)

        seed_result = self.build_retrieval_result(sample.question_id, sample.question, retrieve_output.seed_documents)
        seed_record = PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=seed_result,
            answer=answer_result,
            rewritten_query=None,
            trace={},
        )
        post_record = PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=None,
            trace={},
        )

        expansion_reasons = {
            candidate.section_id: candidate.reasons for candidate in retrieve_output.expanded_candidates
        }
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=None,
            trace={
                "pipeline": "question -> seed retrieve -> graph-guided expansion -> organize -> rerank -> generate",
                "retrieval_mode": self.config.retrieval_mode,
                "use_rerank": self.config.rerank,
                **self.config.trace_metadata,
                "seed_sections": [doc.section_id for doc in retrieve_output.seed_documents],
                "initial_dense_candidates": initial_dense_candidates,
                "seed_dense_recall_scores": seed_dense_scores,
                "expanded_sections": [candidate.section_id for candidate in retrieve_output.expanded_candidates],
                "expansion_reasons": expansion_reasons,
                **rerank_trace,
                "rerank_scores": rerank_scores,
                "final_reranked_candidates": [doc.section_id for doc in final_documents],
                "final_kept_sections": [doc.section_id for doc in final_documents],
                "graph_expansion": {
                    "expanded_count": len(retrieve_output.expanded_candidates),
                    "seed_hit": bool(hit_at_k(seed_record, self.config.seed_top_k)),
                    "post_expand_hit": bool(hit_at_k(post_record, self.config.final_top_k)),
                },
                "organization_details": retrieve_output.organized.organization_details,
                "generator_metadata": answer_result.metadata,
            },
        )


def build_graph_retriever_config(config: GraphEnhancedRAGConfig) -> GraphRetrieverConfig:
    return GraphRetrieverConfig(
        seed_top_k=config.seed_top_k,
        expand_k=config.expand_k,
        final_top_k=config.final_top_k,
        use_gold_hints=config.use_gold_hints,
    )
