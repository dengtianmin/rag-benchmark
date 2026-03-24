from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark_host.schemas.common import BenchmarkSample, ExperimentPrediction


class BaselineSystem(ABC):
    name: str

    @abstractmethod
    def predict(self, sample: BenchmarkSample) -> ExperimentPrediction:
        raise NotImplementedError
