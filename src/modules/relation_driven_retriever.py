from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex, overlap_score


@dataclass(slots=True)
class RelationDrivenRetrieveResult:
    documents: list[RetrievedDocument]
    scores: dict[str, float]
    details: dict


class RelationDrivenRetriever:
    """Relation-aware retrieval that fuses text index and graph hints."""

    def __init__(self, text_index: PublicIndex, graph_index: GraphIndex) -> None:
        self.text_index = text_index
        self.graph_index = graph_index

    def retrieve(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        top_k: int,
        use_relation_driven: bool = True,
        use_skeleton_rewrite: bool = True,
    ) -> RelationDrivenRetrieveResult:
        query = skeleton.rewritten_query(sample.question) if use_skeleton_rewrite else sample.question
        if not use_relation_driven:
            documents = self.text_index.search(query, top_k=top_k)
            return RelationDrivenRetrieveResult(
                documents=documents,
                scores={document.section_id: document.score for document in documents},
                details={"mode": "text_only", "query": query},
            )

        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}

        for document in self.text_index.search(query, top_k=top_k * 3):
            candidate_scores[document.section_id] = candidate_scores.get(document.section_id, 0.0) + document.score
            candidate_reasons.setdefault(document.section_id, []).append("lexical_query_match")

        for entity in skeleton.entities:
            for section_id in self.graph_index.entity_to_sections.get(entity.lower(), set()):
                candidate_scores[section_id] = candidate_scores.get(section_id, 0.0) + 1.0
                candidate_reasons.setdefault(section_id, []).append(f"entity:{entity}")
        for relation in skeleton.relations:
            for section_id in self.graph_index.relation_to_sections.get(relation.lower(), set()):
                candidate_scores[section_id] = candidate_scores.get(section_id, 0.0) + 1.2
                candidate_reasons.setdefault(section_id, []).append(f"relation:{relation}")
        if skeleton.constraints:
            constraint_text = " ".join(skeleton.constraints)
            for section_id, constraints in self.graph_index.section_to_constraints.items():
                if constraints:
                    score = overlap_score(constraint_text, " ".join(constraints))
                    if score > 0:
                        candidate_scores[section_id] = candidate_scores.get(section_id, 0.0) + score
                        candidate_reasons.setdefault(section_id, []).append("constraint_match")

        ranked_ids = sorted(candidate_scores, key=lambda section_id: candidate_scores[section_id], reverse=True)[:top_k]
        documents = []
        for rank, section_id in enumerate(ranked_ids, start=1):
            base_document = self.text_index.by_section_id.get(section_id)
            if base_document is None:
                continue
            documents.append(
                RetrievedDocument(
                    source_id=base_document.source_id,
                    section_id=base_document.section_id,
                    content=base_document.content,
                    score=candidate_scores[section_id],
                    rank=rank,
                    metadata={
                        **base_document.metadata,
                        "doc_title": base_document.doc_title,
                        "section_path": base_document.section_path,
                        "retriever": "relation_driven",
                        "reasons": candidate_reasons.get(section_id, []),
                    },
                )
            )
        return RelationDrivenRetrieveResult(
            documents=documents,
            scores={section_id: candidate_scores[section_id] for section_id in ranked_ids},
            details={"mode": "relation_driven", "query": query, "reasons": candidate_reasons},
        )
