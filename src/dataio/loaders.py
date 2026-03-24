from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from core.schema import BenchmarkSample, DatasetBundle, DatasetSplit
from dataio.normalizers import normalize_benchmark_sample


@dataclass(slots=True)
class GraphRelationRecord:
    subject: str
    predicate: str
    object: str
    description: str = ""


@dataclass(slots=True)
class GraphSectionRecord:
    source_id: str
    section_id: str
    content: str
    section_path: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relations: list[GraphRelationRecord] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    target = Path(path)
    with target.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                yield orjson.loads(raw_line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line at {target}:{line_number}") from exc


def load_benchmark_samples(path: str | Path) -> list[BenchmarkSample]:
    return [normalize_benchmark_sample(record) for record in iter_jsonl(path)]


def load_dataset_bundle(
    benchmark_dataset_path: str | Path,
    *,
    dataset_name: str = "benchmark",
    split_name: str = "full",
) -> DatasetBundle:
    samples = load_benchmark_samples(benchmark_dataset_path)
    split = DatasetSplit(split_name=split_name, samples=samples)
    return DatasetBundle(dataset_name=dataset_name, splits={split_name: split})


def load_graph_section_records(path: str | Path) -> list[GraphSectionRecord]:
    records: list[GraphSectionRecord] = []
    for row in iter_jsonl(path):
        extraction = row.get("extraction", {}) or {}
        entities = []
        for item in extraction.get("entities", []):
            if isinstance(item, dict):
                entities.append(str(item.get("normalized_name") or item.get("name") or "").strip())
            else:
                entities.append(str(item).strip())
        relations = []
        for item in extraction.get("relations", []):
            if not isinstance(item, dict):
                continue
            relations.append(
                GraphRelationRecord(
                    subject=str(item.get("subject") or item.get("head") or "").strip(),
                    predicate=str(item.get("predicate") or item.get("relation") or "").strip(),
                    object=str(item.get("object") or item.get("tail") or "").strip(),
                    description=str(item.get("description", "")).strip(),
                )
            )
        constraints = [str(item).strip() for item in extraction.get("constraints", []) if str(item).strip()]
        evidence_spans = [str(item).strip() for item in extraction.get("evidence_spans", []) if str(item).strip()]
        records.append(
            GraphSectionRecord(
                source_id=str(row.get("doc_id", "")).strip(),
                section_id=str(row.get("section_id", "")).strip(),
                content=str(row.get("content", "")),
                section_path=[str(item) for item in row.get("section_path", [])],
                entities=[item for item in entities if item],
                relations=[item for item in relations if item.predicate],
                constraints=constraints,
                evidence_spans=evidence_spans,
            )
        )
    return records
