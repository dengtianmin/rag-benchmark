from __future__ import annotations

import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_./:-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
