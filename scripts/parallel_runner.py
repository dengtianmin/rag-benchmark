from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from threading import local
from typing import Callable, TypeVar

from tqdm import tqdm


T = TypeVar("T")
R = TypeVar("R")


def run_samples(
    samples: list[T],
    *,
    build_pipeline: Callable[[], object],
    run_sample: Callable[[object, T], R],
    description: str,
    max_workers: int = 1,
) -> list[R]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if not samples:
        return []

    if max_workers == 1:
        pipeline = build_pipeline()
        return [run_sample(pipeline, sample) for sample in tqdm(samples, desc=description)]

    thread_local = local()

    def _get_pipeline() -> object:
        pipeline = getattr(thread_local, "pipeline", None)
        if pipeline is None:
            pipeline = build_pipeline()
            thread_local.pipeline = pipeline
        return pipeline

    def _execute(index: int, sample: T) -> tuple[int, R]:
        return index, run_sample(_get_pipeline(), sample)

    results: list[R | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future[tuple[int, R]]] = [
            executor.submit(_execute, index, sample) for index, sample in enumerate(samples)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            index, record = future.result()
            results[index] = record
    return [record for record in results if record is not None]
