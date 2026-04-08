from __future__ import annotations

from core.schema import BenchmarkSample
from dataio.loaders import load_benchmark_samples


def load_lab_dataset(
    dataset_path: str,
    *,
    max_samples: int | None = None,
    question_types: set[str] | None = None,
) -> list[BenchmarkSample]:
    samples = load_benchmark_samples(dataset_path)
    if question_types:
        samples = [sample for sample in samples if str(sample.question_type) in question_types]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
