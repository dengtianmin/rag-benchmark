from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from benchmark_host.schemas.common import ExperimentPaths


@dataclass(slots=True)
class RetrieverConfig:
    top_k: int = 5
    expand_k: int = 3
    neighbor_hops: int = 1


@dataclass(slots=True)
class GeneratorConfig:
    mode: str = "extractive_stub"
    max_evidence_chars: int = 1200


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    paths: ExperimentPaths
    systems: list[str]
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        paths = ExperimentPaths(
            benchmark_dataset=Path(data["paths"]["benchmark_dataset"]),
            markdown_sections=Path(data["paths"]["markdown_sections"]),
            knowledge_extraction=Path(data["paths"]["knowledge_extraction"]),
            output_dir=Path(data["paths"]["output_dir"]),
        )
        retriever = RetrieverConfig(**data.get("retriever", {}))
        generator = GeneratorConfig(**data.get("generator", {}))
        return cls(
            name=data["name"],
            paths=paths,
            systems=list(data["systems"]),
            retriever=retriever,
            generator=generator,
            limit=data.get("limit"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        data = orjson.loads(Path(path).read_bytes())
        return cls.from_dict(data)
