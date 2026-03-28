from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.schema import BenchmarkSample, RetrievedDocument
from dataio.loaders import load_graph_section_records
from modules.graph_expander import ExpandedCandidate, GraphExpander, GraphIndex
from modules.graph_organizer import GraphOrganizer, OrganizedResult
from pipelines.base import PublicIndex
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve


@dataclass(slots=True)
class GraphRetrieverConfig:
    seed_top_k: int = 5
    expand_k: int = 5
    final_top_k: int = 5
    use_gold_hints: bool = True


@dataclass(slots=True)
class GraphRetrieveOutput:
    seed_documents: list[RetrievedDocument]
    expanded_candidates: list[ExpandedCandidate]
    organized: OrganizedResult


class GraphRetriever:
    """Semantic seed retrieval + graph-guided expansion + organization."""

    def __init__(
        self,
        *,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        seed_retriever: SupportsRetrieve | None = None,
        config: GraphRetrieverConfig | None = None,
    ) -> None:
        self.text_index = text_index
        self.graph_index = graph_index
        self.config = config or GraphRetrieverConfig()
        self.seed_retriever = seed_retriever or LexicalTextRetriever(text_index)
        self.expander = GraphExpander(graph_index)
        self.organizer = GraphOrganizer(text_index, graph_index)

    @classmethod
    def from_paths(
        cls,
        *,
        sections_path: str | Path,
        knowledge_path: str | Path,
        seed_retriever: SupportsRetrieve | None = None,
        config: GraphRetrieverConfig | None = None,
    ) -> "GraphRetriever":
        text_index = PublicIndex.from_markdown_sections(sections_path)
        graph_records = load_graph_section_records(knowledge_path)
        graph_index = GraphIndex.build(graph_records)
        return cls(text_index=text_index, graph_index=graph_index, seed_retriever=seed_retriever, config=config)

    def retrieve(self, sample: BenchmarkSample) -> GraphRetrieveOutput:
        seed_documents = self.seed_retriever.retrieve(sample.question, top_k=self.config.seed_top_k)
        expanded_candidates = self.expander.expand(
            sample,
            seed_documents,
            max_expand=self.config.expand_k,
            use_gold_hints=self.config.use_gold_hints,
        )
        organized = self.organizer.organize(
            query=sample.question,
            seed_documents=seed_documents,
            expanded_candidates=expanded_candidates,
            keep_top_k=self.config.final_top_k,
        )
        return GraphRetrieveOutput(
            seed_documents=seed_documents,
            expanded_candidates=expanded_candidates,
            organized=organized,
        )
