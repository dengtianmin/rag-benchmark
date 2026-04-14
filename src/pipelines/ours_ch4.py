from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from typing import Any

from core.schema import BenchmarkSample, PipelineRunRecord
from modules.question_type_classifier import QuestionTypeClassifier
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator, build_query_analysis_payload
from pipelines.base import BasePipeline, MockGenerator, MockReranker, PublicIndex, SupportsGenerate


AblationMode = Literal["full", "w/o_relation_driven", "w/o_skeleton_rewrite", "w/o_text_compensation"]


@dataclass(slots=True)
class OursCh4Config:
    top_k: int = 5
    skeleton_mode: str = "oracle"
    use_relation_driven: bool = True
    use_skeleton_rewrite: bool = True
    use_text_compensation: bool = True
    text_compensation_strategy: str = "legacy_compensation"
    rerank: bool = True
    rerank_top_n: int | None = None
    retrieval_mode: str = "lexical"
    trace_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ablation(cls, mode: AblationMode, *, top_k: int = 5, skeleton_mode: str = "oracle") -> "OursCh4Config":
        if mode == "full":
            return cls(top_k=top_k, skeleton_mode=skeleton_mode)
        if mode == "w/o_relation_driven":
            return cls(top_k=top_k, skeleton_mode=skeleton_mode, use_relation_driven=False)
        if mode == "w/o_skeleton_rewrite":
            return cls(top_k=top_k, skeleton_mode=skeleton_mode, use_skeleton_rewrite=False)
        if mode == "w/o_text_compensation":
            return cls(top_k=top_k, skeleton_mode=skeleton_mode, use_text_compensation=False)
        raise ValueError(f"Unsupported ablation mode: {mode}")


class OursCh4Pipeline(BasePipeline):
    method_name = "ours_ch4"

    def __init__(
        self,
        text_index: PublicIndex,
        skeleton_extractor: SkeletonExtractor,
        relation_driven_retriever: RelationDrivenRetriever,
        text_compensator: TextCompensator,
        *,
        config: OursCh4Config | None = None,
        reranker: Any | None = None,
        generator: SupportsGenerate | None = None,
    ) -> None:
        self.text_index = text_index
        self.skeleton_extractor = skeleton_extractor
        self.relation_driven_retriever = relation_driven_retriever
        self.text_compensator = text_compensator
        self.config = config or OursCh4Config()
        self.reranker = reranker or MockReranker()
        self.generator = generator or MockGenerator()
        self.question_type_classifier = QuestionTypeClassifier()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        skeleton = self.skeleton_extractor.extract(sample, mode=self.config.skeleton_mode)
        query_analysis = build_query_analysis_payload(
            sample,
            skeleton,
            query_analysis=self.question_type_classifier.classify(sample.question, skeleton),
        )
        retrieve_result = self.relation_driven_retriever.retrieve(
            sample,
            skeleton,
            top_k=self.config.top_k,
            use_relation_driven=self.config.use_relation_driven,
            use_skeleton_rewrite=self.config.use_skeleton_rewrite,
        )
        stage1_candidates = [
            item["section_id"] if isinstance(item, dict) else item
            for item in retrieve_result.details.get("candidate_pool", [])
        ]
        compensation = self.text_compensator.compensate(
            sample,
            skeleton,
            retrieve_result.documents,
            top_k=self.config.top_k,
            enabled=self.config.use_text_compensation,
            query_analysis=query_analysis,
            strategy=self.config.text_compensation_strategy,
        )
        final_documents = compensation.documents
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
        retrieval_query = skeleton.rewritten_query(sample.question) if self.config.use_skeleton_rewrite else sample.question
        retrieval_result = self.build_retrieval_result(sample.question_id, retrieval_query, final_documents)
        answer_result = self.generator.generate(sample, final_documents)
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=retrieval_query if retrieval_query != sample.question else None,
            trace={
                "pipeline": "question -> predicted skeleton -> stage1 text candidate generation -> stage2 relation-driven decision -> conditional text compensation -> rerank -> generate",
                "skeleton": skeleton.to_trace_dict(),
                "skeleton_mode": skeleton.mode,
                "skeleton_details": skeleton.details,
                "query_analysis": query_analysis.to_dict(),
                "retrieval_mode": self.config.retrieval_mode,
                "use_rerank": self.config.rerank,
                **self.config.trace_metadata,
                "stage1_text_candidates": stage1_candidates,
                "initial_dense_candidates": stage1_candidates if self.config.retrieval_mode in {"dense", "hybrid"} else [],
                "relation_driven_candidates": [document.section_id for document in retrieve_result.documents],
                "relation_driven_scores": retrieve_result.scores,
                "relation_driven_details": retrieve_result.details,
                "text_compensation_activated": compensation.activated,
                "text_compensation_reason": compensation.reason,
                "text_compensation_details": compensation.details,
                **rerank_trace,
                "rerank_scores": rerank_scores,
                "final_reranked_candidates": [document.section_id for document in final_documents],
                "final_kept_sections": [document.section_id for document in final_documents],
                "ablation": {
                    "use_relation_driven": self.config.use_relation_driven,
                    "use_skeleton_rewrite": self.config.use_skeleton_rewrite,
                    "use_text_compensation": self.config.use_text_compensation,
                    "text_compensation_strategy": self.config.text_compensation_strategy,
                },
                "generator_metadata": answer_result.metadata,
            },
        )


def summarize_ablation(records: list[PipelineRunRecord]) -> dict[str, float]:
    if not records:
        return {"text_compensation_activation_rate": 0.0}
    total = len(records)
    return {
        "text_compensation_activation_rate": sum(1 for record in records if record.trace["text_compensation_activated"]) / total,
    }
