from __future__ import annotations

from dataclasses import dataclass

from core.schema import AnswerResult, BenchmarkSample, EvidenceItem, RetrievedDocument, RetrievedTriple
from modules.entity_linker import EntityLinkResult
from modules.graph_expander import GraphIndex
from modules.relation_matcher import RelationMatchResult
from pipelines.base import PublicIndex, overlap_score


@dataclass(slots=True)
class KBExecutionResult:
    answer: AnswerResult
    retrieved_triples: list[RetrievedTriple]
    supporting_documents: list[RetrievedDocument]
    execution_trace: dict


class KBExecutor:
    """Execute lightweight subgraph retrieval over extracted triples."""

    def __init__(self, graph_index: GraphIndex, text_index: PublicIndex) -> None:
        self.graph_index = graph_index
        self.text_index = text_index

    def execute(
        self,
        sample: BenchmarkSample,
        entity_result: EntityLinkResult,
        relation_result: RelationMatchResult,
        *,
        top_k: int = 5,
    ) -> KBExecutionResult:
        triple_scores: dict[tuple[str, str, str, str], float] = {}
        triple_reasons: dict[tuple[str, str, str, str], list[str]] = {}

        def add_score(triple: tuple[str, str, str, str], score: float, reason: str) -> None:
            triple_scores[triple] = triple_scores.get(triple, 0.0) + score
            triple_reasons.setdefault(triple, []).append(reason)

        for entity in entity_result.linked_entities:
            for triple in self.graph_index.entity_to_triples.get(entity.lower(), []):
                add_score(triple, 1.0, f"entity:{entity}")
        for relation in relation_result.matched_relations:
            for triple in self.graph_index.relation_to_triples.get(relation.lower(), []):
                add_score(triple, 1.2, f"relation:{relation}")

        if sample.constraints:
            constraint_text = " ".join(sample.constraints)
            for triple in list(triple_scores):
                _, subject, predicate, obj = triple
                score = overlap_score(constraint_text, " ".join([subject, predicate, obj]))
                if score > 0:
                    add_score(triple, score, "constraint_match")

        ranked_triples = sorted(triple_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        retrieved_triples = []
        supporting_documents = []
        supporting_evidence = []
        for rank, (triple, score) in enumerate(ranked_triples, start=1):
            section_id, subject, predicate, obj = triple
            section_record = self.graph_index.section_records.get(section_id)
            if section_record is None:
                continue
            retrieved_triples.append(
                RetrievedTriple(
                    source_id=section_record.source_id,
                    section_id=section_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    score=score,
                    rank=rank,
                    metadata={"reasons": triple_reasons.get(triple, [])},
                )
            )
            text_doc = self.text_index.by_section_id.get(section_id)
            if text_doc is not None and all(item.section_id != section_id for item in supporting_documents):
                supporting_documents.append(
                    RetrievedDocument(
                        source_id=text_doc.source_id,
                        section_id=text_doc.section_id,
                        content=text_doc.content,
                        score=score,
                        rank=len(supporting_documents) + 1,
                        metadata={
                            **text_doc.metadata,
                            "doc_title": text_doc.doc_title,
                            "section_path": text_doc.section_path,
                            "retriever": "kb_executor",
                        },
                    )
                )
            quote = section_record.evidence_spans[0] if section_record.evidence_spans else section_record.content[:240]
            supporting_evidence.append(EvidenceItem(source_id=section_record.source_id, section_id=section_id, quote=quote))

        if retrieved_triples:
            answer_text = "；".join(triple.object or triple.subject for triple in retrieved_triples[:3] if triple.object or triple.subject)
        else:
            answer_text = "No KB answer found."

        answer = AnswerResult(
            question_id=sample.question_id,
            answer_text=answer_text,
            answer_short=answer_text,
            supporting_evidence=supporting_evidence,
            metadata={"answer_source": "kb_executor", "fallback_used": not bool(retrieved_triples)},
        )
        return KBExecutionResult(
            answer=answer,
            retrieved_triples=retrieved_triples,
            supporting_documents=supporting_documents,
            execution_trace={
                "linked_entities": entity_result.linked_entities,
                "matched_relations": relation_result.matched_relations,
                "triple_count": len(retrieved_triples),
                "fallback_used": not bool(retrieved_triples),
            },
        )
