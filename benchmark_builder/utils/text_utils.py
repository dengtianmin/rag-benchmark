from __future__ import annotations

from difflib import SequenceMatcher
import re


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def normalize_for_similarity(text: str) -> str:
    text = normalize_text(text)
    return PUNCT_RE.sub("", text)


def is_quote_relevant(answer: str, quote: str) -> bool:
    answer_norm = set(normalize_for_similarity(answer))
    quote_norm = set(normalize_for_similarity(quote))
    if not answer_norm or not quote_norm:
        return False
    overlap = len(answer_norm & quote_norm) / max(1, len(answer_norm))
    return overlap >= 0.2


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_similarity(a), normalize_for_similarity(b)).ratio()
