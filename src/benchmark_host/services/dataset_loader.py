from __future__ import annotations

from dataclasses import dataclass

from benchmark_host.schemas.common import BenchmarkSample, KnowledgeExtraction, MarkdownSection
from benchmark_host.utils.jsonl import read_jsonl


@dataclass(slots=True)
class CorpusBundle:
    samples: list[BenchmarkSample]
    sections: list[MarkdownSection]
    extractions: list[KnowledgeExtraction]


class DatasetLoader:
    def load(
        self,
        benchmark_dataset_path: str,
        markdown_sections_path: str,
        knowledge_extraction_path: str,
        limit: int | None = None,
    ) -> CorpusBundle:
        samples = read_jsonl(benchmark_dataset_path, BenchmarkSample.from_dict)
        sections = read_jsonl(markdown_sections_path, MarkdownSection.from_dict)
        extractions = read_jsonl(knowledge_extraction_path, KnowledgeExtraction.from_dict)
        if limit is not None:
            samples = samples[:limit]
        return CorpusBundle(samples=samples, sections=sections, extractions=extractions)
