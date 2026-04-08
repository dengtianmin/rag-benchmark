from __future__ import annotations

import re

from core.schema import BenchmarkSample
from pipelines.base import tokenize

from rewrite_lab.rewriters.base import BaseLabRewriter
from rewrite_lab.schema import RewriteCandidate
from rewrite_lab.utils import (
    dedupe_keep_order,
    extract_attribute_like_terms,
    extract_model_like_tokens,
    is_skip_worthy_question,
    question_type_label,
    strip_question_noise,
)


RELATION_HINTS = ("关系", "原因", "作用", "区别", "支持", "兼容", "影响", "依赖")
GENERIC_STOPWORDS = {
    "产品",
    "设备",
    "系统",
    "功能",
    "这个",
    "那个",
    "一下",
    "一下子",
}


class RuleBasedV1Rewriter(BaseLabRewriter):
    """
    Heuristic retrieval-query generator for offline rewrite experiments.

    Design constraints:
    - Only consumes the raw question text.
    - Keeps model names, English abbreviations, digits, and units when present.
    - Removes conversational and interrogative noise.
    - Falls back to the original question when the rewrite looks too lossy.
    """

    strategy_name = "rule_based_v1"

    def rewrite(self, sample: BenchmarkSample) -> RewriteCandidate:
        original = sample.question.strip()
        flags: list[str] = []

        if is_skip_worthy_question(original):
            compact = self._compress_question(original)
            return RewriteCandidate(
                question_id=sample.question_id,
                original_question=original,
                rewritten_query=compact,
                strategy_name=self.strategy_name,
                question_type=question_type_label(sample.question_type),
                confidence=0.92,
                flags=["skipped", "skip_worthy"],
                extracted_entities=extract_model_like_tokens(original),
                extracted_attributes_or_relations=extract_attribute_like_terms(original),
                notes={"reason": "short_spec_question"},
            )

        cleaned = self._compress_question(original)
        entities = extract_model_like_tokens(original)
        attrs = extract_attribute_like_terms(original)
        relations = [hint for hint in RELATION_HINTS if hint in original]

        segments = []
        segments.extend(entities)
        segments.extend(attrs)
        segments.extend(relations)

        residual_tokens = []
        for token in tokenize(cleaned):
            if len(token) <= 1:
                continue
            if token in GENERIC_STOPWORDS:
                continue
            if any(token in tokenize(part) for part in segments):
                continue
            residual_tokens.append(token)

        segments.extend(residual_tokens[:6])
        rewritten = " ".join(dedupe_keep_order(segments))
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        if not rewritten:
            flags.extend(["malformed", "fallback_original"])
            rewritten = original

        if entities and not self._preserves_entities(entities, rewritten):
            flags.extend(["entity_dropped", "fallback_original"])
            rewritten = original

        confidence = self._score_rewrite(original, rewritten, entities, attrs)
        if confidence < 0.35:
            flags.extend(["low_confidence", "fallback_original"])
            rewritten = original

        return RewriteCandidate(
            question_id=sample.question_id,
            original_question=original,
            rewritten_query=rewritten,
            strategy_name=self.strategy_name,
            question_type=question_type_label(sample.question_type),
            extracted_entities=entities,
            extracted_attributes_or_relations=dedupe_keep_order([*attrs, *relations]),
            confidence=round(confidence, 4),
            flags=dedupe_keep_order(flags),
            notes={"cleaned_question": cleaned},
        )

    def _compress_question(self, question: str) -> str:
        cleaned = strip_question_noise(question)
        cleaned = re.sub(r"\b(吗|呢|呀)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or question.strip()

    def _preserves_entities(self, entities: list[str], rewritten: str) -> bool:
        lowered = rewritten.lower()
        for entity in entities:
            if entity.lower() not in lowered:
                return False
        return True

    def _score_rewrite(self, original: str, rewritten: str, entities: list[str], attrs: list[str]) -> float:
        if rewritten == original:
            return 0.6
        original_len = max(len(tokenize(original)), 1)
        rewritten_len = len(tokenize(rewritten))
        length_score = 1.0 - min(abs(original_len - rewritten_len) / original_len, 1.0)
        entity_score = 1.0 if not entities else (1.0 if self._preserves_entities(entities, rewritten) else 0.0)
        attr_score = 1.0 if not attrs else min(sum(1 for attr in attrs if attr in rewritten) / len(attrs), 1.0)
        return round(0.45 * entity_score + 0.35 * attr_score + 0.20 * length_score, 4)
