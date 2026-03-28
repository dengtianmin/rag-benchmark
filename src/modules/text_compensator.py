from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from pipelines.base import PublicIndex, overlap_score
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve


@dataclass(slots=True)
class TextCompensationResult:
    documents: list[RetrievedDocument]
    activated: bool
    reason: str
    details: dict


class TextCompensator:
    """Compensate graph-structured retrieval with text details."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
    ) -> None:
        self.text_index = text_index
        self.graph_index = graph_index
        self.text_retriever = text_retriever or LexicalTextRetriever(text_index)

    def compensate(
        self,
        sample: BenchmarkSample,
        base_documents: list[RetrievedDocument],
        *,
        top_k: int,
        enabled: bool = True,
    ) -> TextCompensationResult:
        if not enabled:
            return TextCompensationResult(
                documents=base_documents[:top_k],
                activated=False,
                reason="disabled",
                details={"compensated_count": 0},
            )

        activated = sample.requires_text_compensation or sample.question_type.value == "explanation" or len(base_documents) < 2
        if not activated:
            return TextCompensationResult(
                documents=base_documents[:top_k],
                activated=False,
                reason="not_needed",
                details={"compensated_count": 0},
            )

        merged: dict[str, RetrievedDocument] = {document.section_id: document for document in base_documents}
        reasons: dict[str, str] = {document.section_id: "base" for document in base_documents}
        backfill_scores: dict[str, float] = {}

        for document in self.text_retriever.retrieve(sample.question, top_k=top_k * 2):
            if document.section_id not in merged:
                merged[document.section_id] = document
                retriever_name = str(document.metadata.get("retriever", "retriever_backfill"))
                reasons[document.section_id] = f"{retriever_name}_backfill"
                backfill_scores[document.section_id] = float(document.metadata.get("dense_score", document.score))

        for document in base_documents:
            for entity in self.graph_index.section_to_entities.get(document.section_id, set()):
                for section_id in self.graph_index.entity_to_sections.get(entity, set()):
                    if section_id not in merged and section_id in self.text_index.by_section_id:
                        text_doc = self.text_index.by_section_id[section_id]
                        merged[section_id] = RetrievedDocument(
                            source_id=text_doc.source_id,
                            section_id=text_doc.section_id,
                            content=text_doc.content,
                            score=overlap_score(sample.question, text_doc.content),
                            rank=0,
                            metadata={
                                **text_doc.metadata,
                                "doc_title": text_doc.doc_title,
                                "section_path": text_doc.section_path,
                                "retriever": "text_compensator",
                                "dense_score": 0.0,
                            },
                        )
                        reasons[section_id] = f"shared_entity:{entity}"
                        backfill_scores[section_id] = overlap_score(sample.question, text_doc.content)

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]
        reranked = [document.model_copy(update={"rank": rank}) for rank, document in enumerate(ranked, start=1)]
        return TextCompensationResult(
            documents=reranked,
            activated=True,
            reason="text_detail_needed",
            details={
                "compensated_count": max(0, len(reranked) - len(base_documents[:top_k])),
                "reasons": {document.section_id: reasons.get(document.section_id, "unknown") for document in reranked},
                "backfill_scores": {document.section_id: backfill_scores.get(document.section_id, document.score) for document in reranked},
            },
        )
