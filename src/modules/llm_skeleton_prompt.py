from __future__ import annotations

import json


def build_llm_skeleton_messages(
    *,
    question: str,
    entity_candidates: list[str],
    relation_candidates: list[str],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是问题骨架抽取器。\n"
        "你的任务是从问题中抽取 entities、relations、constraints。\n"
        "只允许输出严格 JSON 对象，不要解释，不要输出 Markdown。"
    )
    user_prompt = (
        "请根据问题抽取骨架。\n\n"
        "输出格式固定为：\n"
        '{'
        '"entities":["..."],'
        '"relations":["..."],'
        '"constraints":["..."]'
        '}\n\n'
        "规则：\n"
        "1. entities 优先从给定 entity_candidates 中选择。\n"
        "2. relations 优先从给定 relation_candidates 中选择。\n"
        "3. 如果候选缺失，可以补充少量必要项，但必须保持简洁。\n"
        "4. constraints 只抽取问题中的显式约束，如版本、环境、条件、范围、时间。\n"
        "5. 某一类没有内容时返回空列表。\n"
        "6. 严禁输出 JSON 以外的任何文本。\n\n"
        f"question: {question}\n"
        f"entity_candidates: {entity_candidates}\n"
        f"relation_candidates: {relation_candidates}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_llm_skeleton_payload(raw_content: str) -> dict[str, list[str]]:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM skeleton output is not valid JSON: {_preview(raw_content)}") from exc

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM skeleton output is double-encoded but invalid: {_preview(raw_content)}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"LLM skeleton output must be a JSON object: {_preview(raw_content)}")

    normalized: dict[str, list[str]] = {}
    for field in ("entities", "relations", "constraints"):
        value = payload.get(field, [])
        if value is None:
            normalized[field] = []
            continue
        if not isinstance(value, list):
            raise ValueError(f"LLM skeleton field '{field}' must be a list.")
        normalized[field] = [str(item) for item in value]
    return normalized


def _preview(raw_content: str, max_chars: int = 200) -> str:
    compact = " ".join(raw_content.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."
