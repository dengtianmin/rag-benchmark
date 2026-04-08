from __future__ import annotations

from core.schema import BenchmarkSample
from modules.query_rewriter import QueryRewriter

from rewrite_lab.rewriters.base import BaseLabRewriter
from rewrite_lab.schema import RewriteCandidate
from rewrite_lab.utils import question_type_label


class NaiveCurrentAdapter(BaseLabRewriter):
    """Adapter around the current QueryRewriter naive mode without altering default behavior."""

    strategy_name = "naive_current_adapter"

    def __init__(self) -> None:
        self.rewriter = QueryRewriter()

    def rewrite(self, sample: BenchmarkSample) -> RewriteCandidate:
        output = self.rewriter.rewrite(sample.question, sample=sample, mode="naive")
        return RewriteCandidate(
            question_id=sample.question_id,
            original_question=sample.question,
            rewritten_query=output.rewritten_query,
            strategy_name=self.strategy_name,
            question_type=question_type_label(sample.question_type),
            confidence=None,
            notes=output.details,
        )
