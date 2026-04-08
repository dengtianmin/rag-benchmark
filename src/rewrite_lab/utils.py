from __future__ import annotations

import re
from collections import Counter
from statistics import mean

from core.schema import BenchmarkSample
from pipelines.base import tokenize


MODEL_TOKEN_PATTERN = re.compile(r"[A-Za-z]+[-/]?[A-Za-z0-9._-]*\d+[A-Za-z0-9._/-]*|\d+(?:\.\d+)?\s*(?:g|gb|tb|m|mm|cm|kg|w|v|mah|mbps|gbps|pps)", re.IGNORECASE)
QUESTION_NOISE_PHRASES = (
    "是什么",
    "有哪些",
    "如何",
    "多少",
    "是否",
    "请问",
    "为什么",
    "分别",
    "什么关系",
    "通过什么过程",
    "在什么情况下",
)
ATTRIBUTE_HINTS = (
    "交换容量",
    "包转发率",
    "外形尺寸",
    "协议",
    "配置状态",
    "状态",
    "参数",
    "规格",
    "尺寸",
    "功耗",
    "吞吐",
    "速率",
    "带宽",
    "容量",
    "版本",
    "型号",
    "区别",
    "作用",
    "原因",
    "支持",
)
PUNCT_TO_SPACE = str.maketrans({char: " " for char in "，。！？?、:：;；()（）[]【】{}<>\"'"})


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text))


def question_type_label(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def extract_model_like_tokens(text: str) -> list[str]:
    return dedupe_keep_order([match.group(0).strip() for match in MODEL_TOKEN_PATTERN.finditer(text or "")])


def extract_attribute_like_terms(text: str) -> list[str]:
    text = text or ""
    matches = [hint for hint in ATTRIBUTE_HINTS if hint in text]
    return dedupe_keep_order(matches)


def strip_question_noise(text: str) -> str:
    cleaned = text or ""
    for phrase in QUESTION_NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = cleaned.translate(PUNCT_TO_SPACE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_skip_worthy_question(question: str) -> bool:
    normalized = strip_question_noise(question)
    tokens = tokenize(normalized)
    model_like = extract_model_like_tokens(question)
    attr_like = extract_attribute_like_terms(question)
    return len(tokens) <= 8 and bool(model_like) and bool(attr_like)


def build_reference_fields(sample: BenchmarkSample) -> dict[str, list[str]]:
    entities = dedupe_keep_order([*getattr(sample, "entities", [])])
    relations = dedupe_keep_order([*getattr(sample, "relations", [])])
    constraints = dedupe_keep_order([*getattr(sample, "constraints", [])])
    if not entities:
        entities = extract_model_like_tokens(sample.question)
    if not relations:
        relations = extract_attribute_like_terms(sample.question)
    return {
        "entities": entities,
        "attributes_or_relations": relations,
        "constraints": constraints,
    }


def count_overlap(reference: list[str], text: str) -> tuple[int, int]:
    if not reference:
        return 0, 0
    normalized_text = normalize_phrase(text)
    hit = 0
    for item in reference:
        if not item:
            continue
        normalized_item = normalize_phrase(item)
        if normalized_item and normalized_item in normalized_text:
            hit += 1
    return hit, len(reference)


def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))
