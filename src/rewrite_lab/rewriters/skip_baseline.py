from __future__ import annotations

from core.schema import BenchmarkSample

from rewrite_lab.rewriters.base import BaseLabRewriter
from rewrite_lab.schema import RewriteCandidate
from rewrite_lab.utils import question_type_label


class SkipBaselineRewriter(BaseLabRewriter):
    strategy_name = "skip_baseline"

    def rewrite(self, sample: BenchmarkSample) -> RewriteCandidate:
        return RewriteCandidate(
            question_id=sample.question_id,
            original_question=sample.question,
            rewritten_query=sample.question,
            strategy_name=self.strategy_name,
            question_type=question_type_label(sample.question_type),
            confidence=1.0,
            flags=["skipped"],
            notes={"reason": "identity_rewrite"},
        )
