from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core.schema import BenchmarkSample
from modules.skeleton_utils import build_entity_relation_query
from pipelines.base import tokenize


RewriteMode = Literal["naive", "entity_relation"]

QUESTION_FILLERS = {
    "是什么",
    "有哪些",
    "是什么？",
    "有哪些？",
    "如何",
    "如何？",
    "多少",
    "多少？",
    "是否",
    "是否？",
    "请问",
}


@dataclass(slots=True)
class RewriteOutput:
    rewritten_query: str
    details: dict


class QueryRewriter:
    """Shared query rewrite interface for Rewrite-RAG baselines."""

    def rewrite(
        self,
        question: str,
        sample: BenchmarkSample | None = None,
        mode: RewriteMode = "naive",
    ) -> RewriteOutput:
        if mode == "naive":
            return self._naive_rewrite(question)
        if mode == "entity_relation":
            if sample is None:
                raise ValueError("entity_relation mode requires sample.")
            return self._entity_relation_rewrite(question, sample)
        raise ValueError(f"Unsupported rewrite mode: {mode}")

    def _naive_rewrite(self, question: str) -> RewriteOutput:
        normalized = re.sub(r"\s+", " ", question).strip()
        tokens = tokenize(normalized)
        kept_tokens = [token for token in tokens if token not in QUESTION_FILLERS]
        keywords = kept_tokens[:12]
        if keywords:
            rewritten_query = normalized + " | keywords: " + " ".join(keywords)
        else:
            rewritten_query = normalized
        return RewriteOutput(
            rewritten_query=rewritten_query,
            details={
                "strategy": "naive",
                "normalized_question": normalized,
                "keywords": keywords,
            },
        )

    def _entity_relation_rewrite(self, question: str, sample: BenchmarkSample) -> RewriteOutput:
        rewritten_query, details = build_entity_relation_query(sample)
        details["original_question"] = question
        details["strategy"] = "entity_relation"
        return RewriteOutput(rewritten_query=rewritten_query, details=details)
