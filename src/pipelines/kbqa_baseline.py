from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, PipelineRunRecord, RetrievalResult
from core.types import RetrievalMode
from modules.entity_linker import EntityLinkMode, EntityLinker
from modules.kb_executor import KBExecutor
from modules.relation_matcher import RelationMatchMode, RelationMatcher


@dataclass(slots=True)
class KBQABaselineConfig:
    top_k: int = 5
    entity_mode: EntityLinkMode = "gold"
    relation_mode: RelationMatchMode = "gold"


class KBQABaselinePipeline:
    method_name = "kbqa_baseline"

    def __init__(
        self,
        entity_linker: EntityLinker,
        relation_matcher: RelationMatcher,
        kb_executor: KBExecutor,
        *,
        config: KBQABaselineConfig | None = None,
    ) -> None:
        self.entity_linker = entity_linker
        self.relation_matcher = relation_matcher
        self.kb_executor = kb_executor
        self.config = config or KBQABaselineConfig()

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        entity_result = self.entity_linker.link(sample.question, sample=sample, mode=self.config.entity_mode)
        relation_result = self.relation_matcher.match(sample.question, sample=sample, mode=self.config.relation_mode)
        execution = self.kb_executor.execute(sample, entity_result, relation_result, top_k=self.config.top_k)
        retrieval = RetrievalResult(
            question_id=sample.question_id,
            query=sample.question,
            mode=RetrievalMode.TRIPLE,
            retrieved_documents=execution.supporting_documents,
            retrieved_triples=execution.retrieved_triples,
            debug_info={"kbqa": True},
        )
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.method_name,
            sample=sample,
            retrieval=retrieval,
            answer=execution.answer,
            rewritten_query=None,
            trace={
                "pipeline": "question -> entity linking -> relation matching -> subgraph retrieval / execution -> answer normalization",
                "linked_entities": entity_result.linked_entities,
                "entity_link_details": entity_result.details,
                "matched_relations": relation_result.matched_relations,
                "relation_match_details": relation_result.details,
                "execution_trace": execution.execution_trace,
                "supporting_sections": [doc.section_id for doc in execution.supporting_documents],
            },
        )
