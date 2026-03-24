from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from core.schema import BenchmarkSample, RetrievedDocument
from dataio.loaders import GraphSectionRecord
from pipelines.base import overlap_score


@dataclass(slots=True)
class GraphIndex:
    section_records: dict[str, GraphSectionRecord]
    entity_to_sections: dict[str, set[str]]
    relation_to_sections: dict[str, set[str]]
    entity_to_related_entities: dict[str, set[str]]
    section_to_entities: dict[str, set[str]]
    section_to_relations: dict[str, set[str]]
    section_to_constraints: dict[str, set[str]]

    @classmethod
    def build(cls, records: list[GraphSectionRecord]) -> "GraphIndex":
        section_records = {record.section_id: record for record in records}
        entity_to_sections: dict[str, set[str]] = defaultdict(set)
        relation_to_sections: dict[str, set[str]] = defaultdict(set)
        entity_to_related_entities: dict[str, set[str]] = defaultdict(set)
        section_to_entities: dict[str, set[str]] = {}
        section_to_relations: dict[str, set[str]] = {}
        section_to_constraints: dict[str, set[str]] = {}

        for record in records:
            entity_set = {entity.lower() for entity in record.entities if entity}
            relation_set = {relation.predicate.lower() for relation in record.relations if relation.predicate}
            constraint_set = {constraint.lower() for constraint in record.constraints if constraint}
            section_to_entities[record.section_id] = entity_set
            section_to_relations[record.section_id] = relation_set
            section_to_constraints[record.section_id] = constraint_set

            for entity in entity_set:
                entity_to_sections[entity].add(record.section_id)
            for relation in relation_set:
                relation_to_sections[relation].add(record.section_id)
            for relation in record.relations:
                subject = relation.subject.lower()
                obj = relation.object.lower()
                if subject and obj:
                    entity_to_related_entities[subject].add(obj)
                    entity_to_related_entities[obj].add(subject)

        return cls(
            section_records=section_records,
            entity_to_sections=dict(entity_to_sections),
            relation_to_sections=dict(relation_to_sections),
            entity_to_related_entities=dict(entity_to_related_entities),
            section_to_entities=section_to_entities,
            section_to_relations=section_to_relations,
            section_to_constraints=section_to_constraints,
        )


@dataclass(slots=True)
class ExpandedCandidate:
    section_id: str
    reasons: list[str] = field(default_factory=list)
    graph_score: float = 0.0


class GraphExpander:
    """Lightweight graph-guided candidate expansion adapted to benchmark schema."""

    def __init__(self, graph_index: GraphIndex) -> None:
        self.graph_index = graph_index

    def expand(
        self,
        sample: BenchmarkSample,
        seed_documents: list[RetrievedDocument],
        *,
        max_expand: int,
        use_gold_hints: bool = True,
    ) -> list[ExpandedCandidate]:
        candidates: dict[str, ExpandedCandidate] = {}
        seed_section_ids = {document.section_id for document in seed_documents}

        def add_candidate(section_id: str, reason: str, score_delta: float) -> None:
            if section_id in seed_section_ids:
                return
            candidate = candidates.setdefault(section_id, ExpandedCandidate(section_id=section_id))
            candidate.reasons.append(reason)
            candidate.graph_score += score_delta

        if use_gold_hints:
            for entity in sample.entities:
                for section_id in self.graph_index.entity_to_sections.get(entity.lower(), set()):
                    add_candidate(section_id, f"question_entity:{entity}", 1.0)
                for related in self.graph_index.entity_to_related_entities.get(entity.lower(), set()):
                    for section_id in self.graph_index.entity_to_sections.get(related, set()):
                        add_candidate(section_id, f"question_entity_bridge:{entity}->{related}", 0.8)
            for relation in sample.relations:
                for section_id in self.graph_index.relation_to_sections.get(relation.lower(), set()):
                    add_candidate(section_id, f"question_relation:{relation}", 1.1)
            constraint_text = " ".join(sample.constraints)
            if constraint_text:
                for section_id, constraints in self.graph_index.section_to_constraints.items():
                    if not constraints:
                        continue
                    score = overlap_score(constraint_text, " ".join(constraints))
                    if score > 0:
                        add_candidate(section_id, "question_constraint_match", score)

        for document in seed_documents:
            seed_entities = self.graph_index.section_to_entities.get(document.section_id, set())
            seed_relations = self.graph_index.section_to_relations.get(document.section_id, set())
            for entity in seed_entities:
                for section_id in self.graph_index.entity_to_sections.get(entity, set()):
                    add_candidate(section_id, f"seed_shared_entity:{entity}", 0.7)
                for related in self.graph_index.entity_to_related_entities.get(entity, set()):
                    for section_id in self.graph_index.entity_to_sections.get(related, set()):
                        add_candidate(section_id, f"seed_related_entity:{entity}->{related}", 0.5)
            for relation in seed_relations:
                for section_id in self.graph_index.relation_to_sections.get(relation, set()):
                    add_candidate(section_id, f"seed_shared_relation:{relation}", 0.9)

        ranked = sorted(candidates.values(), key=lambda item: (item.graph_score, len(item.reasons)), reverse=True)
        return ranked[:max_expand]
