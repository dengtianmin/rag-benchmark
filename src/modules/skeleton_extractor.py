from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.schema import BenchmarkSample, QuestionSkeletonLabel
from modules.graph_expander import GraphIndex
from pipelines.base import overlap_score


SkeletonMode = Literal["oracle", "stub_predicted"]


@dataclass(slots=True)
class SkeletonExtractionResult:
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    skeleton: QuestionSkeletonLabel
    mode: SkeletonMode
    details: dict

    def rewritten_query(self, question: str) -> str:
        parts = [question]
        if self.entities:
            parts.append("entities: " + ", ".join(self.entities))
        if self.relations:
            parts.append("relations: " + ", ".join(self.relations))
        if self.constraints:
            parts.append("constraints: " + ", ".join(self.constraints))
        return " | ".join(parts)


class SkeletonExtractor:
    """Extract skeleton used by chapter-4 retrieval rewrite."""

    def __init__(self, graph_index: GraphIndex | None = None) -> None:
        self.graph_index = graph_index

    def extract(self, sample: BenchmarkSample, mode: SkeletonMode = "oracle") -> SkeletonExtractionResult:
        if mode == "oracle":
            skeleton = sample.skeleton_label
            return SkeletonExtractionResult(
                entities=list(skeleton.entities),
                relations=list(skeleton.relations),
                constraints=list(skeleton.constraints),
                skeleton=skeleton,
                mode=mode,
                details={"mode": mode, "source": "benchmark_fields"},
            )

        entities: list[str] = []
        relations: list[str] = []
        if self.graph_index is not None:
            entity_scores = []
            relation_scores = []
            for entity in self.graph_index.entity_to_sections:
                score = overlap_score(sample.question, entity)
                if score > 0:
                    entity_scores.append((entity, score))
            for relation in self.graph_index.relation_to_sections:
                score = overlap_score(sample.question, relation)
                if score > 0:
                    relation_scores.append((relation, score))
            entities = [item for item, _ in sorted(entity_scores, key=lambda pair: pair[1], reverse=True)[:5]]
            relations = [item for item, _ in sorted(relation_scores, key=lambda pair: pair[1], reverse=True)[:5]]
        skeleton = QuestionSkeletonLabel(
            question_type=sample.question_type,
            entities=entities,
            relations=relations,
            constraints=[],
            requires_text_compensation=sample.requires_text_compensation,
        )
        return SkeletonExtractionResult(
            entities=entities,
            relations=relations,
            constraints=[],
            skeleton=skeleton,
            mode=mode,
            details={"mode": mode, "source": "heuristic_stub"},
        )
