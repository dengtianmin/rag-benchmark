from __future__ import annotations

from benchmark_host.components.base import Component, ComponentOutput
from benchmark_host.components.indexes import CorpusIndex
from benchmark_host.schemas.common import BenchmarkSample, RetrievalCandidate
from benchmark_host.utils.text import overlap_score


class TraditionalRetriever(Component):
    def __init__(self, index: CorpusIndex, top_k: int) -> None:
        self.index = index
        self.top_k = top_k

    def run(self, sample: BenchmarkSample, query: str | None = None) -> ComponentOutput:
        query_text = query or sample.question
        hits = []
        for section_id, score in self.index.lexical_search(query_text, self.top_k):
            record = self.index.records[section_id]
            hits.append(
                RetrievalCandidate(
                    section_id=section_id,
                    doc_id=record.doc_id,
                    score=score,
                    content=record.content,
                    metadata={"strategy": "lexical"},
                )
            )
        return ComponentOutput({"retrieved": hits})


class RelationDrivenRetriever(Component):
    def __init__(self, index: CorpusIndex, top_k: int, expand_k: int) -> None:
        self.index = index
        self.top_k = top_k
        self.expand_k = expand_k

    def run(self, sample: BenchmarkSample, query: str | None = None) -> ComponentOutput:
        query_text = query or sample.question
        ranked: dict[str, float] = {}
        for section_id, score in self.index.lexical_search(query_text, self.top_k):
            ranked[section_id] = ranked.get(section_id, 0.0) + score

        for section_id in self.index.entity_search(sample.entities):
            ranked[section_id] = ranked.get(section_id, 0.0) + 0.8
        for section_id in self.index.relation_search(sample.relations):
            ranked[section_id] = ranked.get(section_id, 0.0) + 1.0

        for section_id, record in self.index.records.items():
            extraction = record.extraction
            if not extraction:
                continue
            constraint_text = " ".join(sample.constraints)
            if constraint_text:
                ranked[section_id] = ranked.get(section_id, 0.0) + 0.5 * overlap_score(
                    constraint_text, " ".join(extraction.constraints)
                )

        hits = []
        for section_id, score in sorted(ranked.items(), key=lambda pair: pair[1], reverse=True)[: self.top_k + self.expand_k]:
            record = self.index.records[section_id]
            hits.append(
                RetrievalCandidate(
                    section_id=section_id,
                    doc_id=record.doc_id,
                    score=score,
                    content=record.content,
                    metadata={"strategy": "relation_driven"},
                )
            )
        return ComponentOutput({"retrieved": hits})


class GraphRetriever(Component):
    def __init__(self, index: CorpusIndex, top_k: int, expand_k: int, neighbor_hops: int) -> None:
        self.index = index
        self.top_k = top_k
        self.expand_k = expand_k
        self.neighbor_hops = neighbor_hops

    def run(self, sample: BenchmarkSample, seeds: list[RetrievalCandidate] | None = None) -> ComponentOutput:
        if seeds is None:
            seeds = TraditionalRetriever(self.index, self.top_k).run(sample).data["retrieved"]
        ranked: dict[str, float] = {candidate.section_id: candidate.score for candidate in seeds}
        frontier = {candidate.section_id for candidate in seeds}
        for _ in range(self.neighbor_hops):
            next_frontier: set[str] = set()
            for section_id in frontier:
                for neighbor in self.index.records[section_id].neighbors:
                    ranked[neighbor] = max(ranked.get(neighbor, 0.0), ranked[section_id] * 0.85)
                    next_frontier.add(neighbor)
                record = self.index.records[section_id]
                extraction = record.extraction
                if extraction:
                    for entity in extraction.entities:
                        for peer in self.index.entity_to_sections.get((entity.normalized_name or entity.name).lower(), set()):
                            if peer != section_id:
                                ranked[peer] = max(ranked.get(peer, 0.0), ranked[section_id] * 0.8)
                                next_frontier.add(peer)
            frontier = next_frontier
        hits = []
        for section_id, score in sorted(ranked.items(), key=lambda pair: pair[1], reverse=True)[: self.top_k + self.expand_k]:
            record = self.index.records[section_id]
            hits.append(
                RetrievalCandidate(
                    section_id=section_id,
                    doc_id=record.doc_id,
                    score=score,
                    content=record.content,
                    metadata={"strategy": "graph_expand"},
                )
            )
        return ComponentOutput({"retrieved": hits})
