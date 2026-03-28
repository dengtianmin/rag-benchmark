from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex, overlap_score
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve


@dataclass(slots=True)
class RelationDrivenRetrieveResult:
    documents: list[RetrievedDocument]
    scores: dict[str, float]
    details: dict


class RelationDrivenRetriever:
    """Relation-aware retrieval that fuses text index and graph hints."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self.text_index = text_index
        self.graph_index = graph_index
        self.text_retriever = text_retriever or LexicalTextRetriever(text_index)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
            documents = self.text_retriever.retrieve(query, top_k=top_k)
            return RelationDrivenRetrieveResult(
                documents=documents,
                scores={document.section_id: document.score for document in documents},
                details={
                    "mode": "text_only",
                    "query": query,
                    "dense_scores": {
                        document.section_id: float(document.metadata.get("dense_score", document.score))
                        for document in documents
                    },
                },
            )

        dense_scores: dict[str, float] = {}
        relation_scores: dict[str, float] = {}
        constraint_scores: dict[str, float] = {}
        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}
        dense_documents: dict[str, RetrievedDocument] = {}

        for document in self.text_retriever.retrieve(query, top_k=top_k * 3):
            dense_documents[document.section_id] = document
            dense_score = float(document.metadata.get("dense_score", document.score))
            dense_scores[document.section_id] = dense_score
            candidate_reasons.setdefault(document.section_id, []).append("dense_query_match")

        for entity in skeleton.entities:
            for section_id in self.graph_index.entity_to_sections.get(entity.lower(), set()):
                relation_scores[section_id] = relation_scores.get(section_id, 0.0) + 1.0
                candidate_reasons.setdefault(section_id, []).append(f"entity:{entity}")
        for relation in skeleton.relations:
            for section_id in self.graph_index.relation_to_sections.get(relation.lower(), set()):
                relation_scores[section_id] = relation_scores.get(section_id, 0.0) + 1.2
                candidate_reasons.setdefault(section_id, []).append(f"relation:{relation}")
        if skeleton.constraints:
            constraint_text = " ".join(skeleton.constraints)
            for section_id, constraints in self.graph_index.section_to_constraints.items():
                if constraints:
                    score = overlap_score(constraint_text, " ".join(constraints))
                    if score > 0:
                        constraint_scores[section_id] = constraint_scores.get(section_id, 0.0) + score
                        candidate_reasons.setdefault(section_id, []).append("constraint_match")

        for section_id in set(dense_scores) | set(relation_scores) | set(constraint_scores):
            candidate_scores[section_id] = (
                self.alpha * dense_scores.get(section_id, 0.0)
                + self.beta * relation_scores.get(section_id, 0.0)
                + self.gamma * constraint_scores.get(section_id, 0.0)
            )

        ranked_ids = sorted(candidate_scores, key=lambda section_id: candidate_scores[section_id], reverse=True)[:top_k]
        documents = []
        for rank, section_id in enumerate(ranked_ids, start=1):
            base_dense_document = dense_documents.get(section_id)
            if base_dense_document is not None:
                documents.append(
                    base_dense_document.model_copy(
                        update={
                            "score": candidate_scores[section_id],
                            "rank": rank,
                            "metadata": {
                                **base_dense_document.metadata,
                                "retriever": "relation_driven",
                                "reasons": candidate_reasons.get(section_id, []),
                                "dense_score": dense_scores.get(section_id, 0.0),
                                "relation_score": relation_scores.get(section_id, 0.0),
                                "constraint_score": constraint_scores.get(section_id, 0.0),
                                "fusion_score": candidate_scores[section_id],
                            },
                        }
                    )
                )
                continue
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
                        "dense_score": dense_scores.get(section_id, 0.0),
                        "relation_score": relation_scores.get(section_id, 0.0),
                        "constraint_score": constraint_scores.get(section_id, 0.0),
                        "fusion_score": candidate_scores[section_id],
                    },
                )
            )
        return RelationDrivenRetrieveResult(
            documents=documents,
            scores={section_id: candidate_scores[section_id] for section_id in ranked_ids},
            details={
                "mode": "relation_driven",
                "query": query,
                "reasons": candidate_reasons,
                "dense_scores": {section_id: dense_scores.get(section_id, 0.0) for section_id in ranked_ids},
                "relation_scores": {section_id: relation_scores.get(section_id, 0.0) for section_id in ranked_ids},
                "constraint_scores": {section_id: constraint_scores.get(section_id, 0.0) for section_id in ranked_ids},
                "fusion_weights": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
            },
        )
