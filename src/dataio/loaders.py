from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from core.schema import BenchmarkSample, DatasetBundle, DatasetSplit
from dataio.normalizers import normalize_benchmark_sample


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

