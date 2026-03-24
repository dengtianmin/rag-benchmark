from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_./:-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalized_text(prediction) == normalized_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred = tokenize(prediction)
    truth = tokenize(gold)
    if not pred or not truth:
        return 0.0
    pred_counter = Counter(pred)
    truth_counter = Counter(truth)
    overlap = sum((pred_counter & truth_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)


def overlap_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))
