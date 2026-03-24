from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.schema import BenchmarkSample
from modules.graph_expander import GraphIndex
from pipelines.base import overlap_score


EntityLinkMode = Literal["gold", "heuristic"]


@dataclass(slots=True)
class EntityLinkResult:
    linked_entities: list[str]
    entity_candidates: dict[str, list[str]]
    details: dict


class EntityLinker:
    """Lightweight entity linker over extracted benchmark subgraph."""

    def __init__(self, graph_index: GraphIndex) -> None:
        self.graph_index = graph_index

    def link(self, question: str, sample: BenchmarkSample | None = None, mode: EntityLinkMode = "heuristic") -> EntityLinkResult:
        if mode == "gold":
            if sample is None:
                raise ValueError("gold entity linking requires sample.")
            linked_entities = list(sample.entities)
            return EntityLinkResult(
                linked_entities=linked_entities,
                entity_candidates={entity: sorted(self.graph_index.entity_to_sections.get(entity.lower(), set())) for entity in linked_entities},
                details={"mode": "gold", "linked_count": len(linked_entities)},
            )

        scored = []
        for entity in self.graph_index.entity_to_sections:
            score = overlap_score(question, entity)
            if score > 0:
                scored.append((entity, score))
        top_entities = [entity for entity, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:5]]
        return EntityLinkResult(
            linked_entities=top_entities,
            entity_candidates={entity: sorted(self.graph_index.entity_to_sections.get(entity, set())) for entity in top_entities},
            details={"mode": "heuristic", "candidate_count": len(scored)},
        )
