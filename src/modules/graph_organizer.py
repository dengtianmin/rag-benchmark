from __future__ import annotations

from dataclasses import dataclass

from core.schema import RetrievedDocument
from modules.graph_expander import ExpandedCandidate, GraphIndex
from pipelines.base import PublicIndex, overlap_score


@dataclass(slots=True)
class OrganizedResult:
    final_documents: list[RetrievedDocument]
    kept_section_ids: list[str]
    organization_details: dict


class GraphOrganizer:
    """Merge, deduplicate, and lightly filter seed and expanded graph candidates."""

    def __init__(self, public_index: PublicIndex, graph_index: GraphIndex) -> None:
        self.public_index = public_index
        self.graph_index = graph_index

    def organize(
        self,
        *,
        query: str,
        seed_documents: list[RetrievedDocument],
        expanded_candidates: list[ExpandedCandidate],
        keep_top_k: int,
    ) -> OrganizedResult:
        merged: dict[str, RetrievedDocument] = {document.section_id: document for document in seed_documents}
        scores: dict[str, float] = {
            document.section_id: document.score + 2.0 for document in seed_documents
        }
        reasons: dict[str, list[str]] = {
            document.section_id: ["seed_retrieval"] for document in seed_documents
        }

        for candidate in expanded_candidates:
            base_document = self.public_index.by_section_id.get(candidate.section_id)
            if base_document is None:
                continue
            document = RetrievedDocument(
                source_id=base_document.source_id,
                section_id=base_document.section_id,
                content=base_document.content,
                score=0.0,
                rank=0,
                metadata={
                    **base_document.metadata,
                    "doc_title": base_document.doc_title,
                    "section_path": base_document.section_path,
                    "retriever": "graph_expansion",
                },
            )
            merged.setdefault(candidate.section_id, document)
            graph_record = self.graph_index.section_records.get(candidate.section_id)
            evidence_bonus = 0.0
            if graph_record and graph_record.evidence_spans:
                evidence_bonus = overlap_score(query, " ".join(graph_record.evidence_spans))
            scores[candidate.section_id] = max(scores.get(candidate.section_id, 0.0), 0.0) + candidate.graph_score + evidence_bonus
            reasons.setdefault(candidate.section_id, []).extend(candidate.reasons)

        ranked_ids = sorted(
            merged,
            key=lambda section_id: (scores.get(section_id, 0.0), len(reasons.get(section_id, []))),
            reverse=True,
        )[:keep_top_k]
        final_documents = []
        for rank, section_id in enumerate(ranked_ids, start=1):
            document = merged[section_id]
            final_documents.append(
                document.model_copy(
                    update={
                        "rank": rank,
                        "score": scores.get(section_id, document.score),
                        "metadata": {
                            **document.metadata,
                            "organization_reasons": reasons.get(section_id, []),
                        },
                    }
                )
            )
        return OrganizedResult(
            final_documents=final_documents,
            kept_section_ids=ranked_ids,
            organization_details={
                "scores": {section_id: scores.get(section_id, 0.0) for section_id in ranked_ids},
                "reasons": {section_id: reasons.get(section_id, []) for section_id in ranked_ids},
            },
        )
