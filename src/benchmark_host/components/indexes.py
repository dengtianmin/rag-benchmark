from __future__ import annotations

from dataclasses import dataclass, field

from benchmark_host.schemas.common import KnowledgeExtraction, MarkdownSection
from benchmark_host.utils.text import overlap_score, tokenize


@dataclass(slots=True)
class SectionRecord:
    section: MarkdownSection
    extraction: KnowledgeExtraction | None = None
    neighbors: set[str] = field(default_factory=set)

    @property
    def doc_id(self) -> str:
        return self.section.doc_id

    @property
    def section_id(self) -> str:
        return self.section.section_id

    @property
    def content(self) -> str:
        return self.section.content

    def lexical_score(self, query: str) -> float:
        title = " ".join(self.section.section_path)
        return overlap_score(query, f"{title}\n{self.section.content}")


class CorpusIndex:
    def __init__(self, sections: list[MarkdownSection], extractions: list[KnowledgeExtraction]) -> None:
        extraction_map = {item.section_id: item for item in extractions}
        self.records: dict[str, SectionRecord] = {
            section.section_id: SectionRecord(section=section, extraction=extraction_map.get(section.section_id))
            for section in sections
        }
        self.by_doc: dict[str, list[str]] = {}
        for section in sections:
            self.by_doc.setdefault(section.doc_id, []).append(section.section_id)
        self._build_doc_neighbors()
        self.entity_to_sections = self._build_entity_index()
        self.relation_to_sections = self._build_relation_index()

    def _build_doc_neighbors(self) -> None:
        for section_ids in self.by_doc.values():
            for idx, section_id in enumerate(section_ids):
                if idx > 0:
                    self.records[section_id].neighbors.add(section_ids[idx - 1])
                if idx + 1 < len(section_ids):
                    self.records[section_id].neighbors.add(section_ids[idx + 1])

    def _build_entity_index(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for section_id, record in self.records.items():
            extraction = record.extraction
            if not extraction:
                continue
            for entity in extraction.entities:
                key = entity.normalized_name or entity.name
                if key:
                    mapping.setdefault(key.lower(), set()).add(section_id)
        return mapping

    def _build_relation_index(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for section_id, record in self.records.items():
            extraction = record.extraction
            if not extraction:
                continue
            for relation in extraction.relations:
                if relation.predicate:
                    mapping.setdefault(relation.predicate.lower(), set()).add(section_id)
        return mapping

    def lexical_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scored = [(section_id, record.lexical_score(query)) for section_id, record in self.records.items()]
        ranked = [item for item in sorted(scored, key=lambda pair: pair[1], reverse=True) if item[1] > 0]
        return ranked[:top_k]

    def entity_search(self, entities: list[str]) -> set[str]:
        section_ids: set[str] = set()
        for entity in entities:
            section_ids |= self.entity_to_sections.get(entity.lower(), set())
        return section_ids

    def relation_search(self, relations: list[str]) -> set[str]:
        section_ids: set[str] = set()
        for relation in relations:
            section_ids |= self.relation_to_sections.get(relation.lower(), set())
        return section_ids

    def query_tokens(self, text: str) -> set[str]:
        return set(tokenize(text))
