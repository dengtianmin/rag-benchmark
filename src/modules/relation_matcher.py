from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.schema import BenchmarkSample
from modules.graph_expander import GraphIndex
from pipelines.base import overlap_score


RelationMatchMode = Literal["gold", "heuristic"]


@dataclass(slots=True)
class RelationMatchResult:
    matched_relations: list[str]
    relation_candidates: dict[str, list[str]]
    details: dict


class RelationMatcher:
    """Lightweight relation matcher over extracted relation vocabulary."""

    def __init__(self, graph_index: GraphIndex) -> None:
        self.graph_index = graph_index

    def match(
        self,
        question: str,
        sample: BenchmarkSample | None = None,
        mode: RelationMatchMode = "heuristic",
    ) -> RelationMatchResult:
        if mode == "gold":
            if sample is None:
                raise ValueError("gold relation matching requires sample.")
            matched_relations = list(sample.relations)
            return RelationMatchResult(
                matched_relations=matched_relations,
                relation_candidates={relation: sorted(self.graph_index.relation_to_sections.get(relation.lower(), set())) for relation in matched_relations},
                details={"mode": "gold", "matched_count": len(matched_relations)},
            )

        scored = []
        for relation in self.graph_index.relation_to_sections:
            score = overlap_score(question, relation)
            if score > 0:
                scored.append((relation, score))
        top_relations = [relation for relation, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:5]]
        return RelationMatchResult(
            matched_relations=top_relations,
            relation_candidates={relation: sorted(self.graph_index.relation_to_sections.get(relation, set())) for relation in top_relations},
            details={"mode": "heuristic", "candidate_count": len(scored)},
        )
