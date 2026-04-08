from __future__ import annotations

from abc import ABC, abstractmethod

from core.schema import BenchmarkSample

from rewrite_lab.schema import RewriteCandidate


class BaseLabRewriter(ABC):
    strategy_name = "base"

    @abstractmethod
    def rewrite(self, sample: BenchmarkSample) -> RewriteCandidate:
        raise NotImplementedError

    def batch_rewrite(self, samples: list[BenchmarkSample]) -> list[RewriteCandidate]:
        return [self.rewrite(sample) for sample in samples]
