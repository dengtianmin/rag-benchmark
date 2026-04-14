from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(slots=True)
class LLMQueryRewritePayload:
    must_keep_terms: list[str]
    sparse_rewrite: str
    dense_rewrite: str

    def to_dict(self) -> dict[str, object]:
        return {
            "must_keep_terms": list(self.must_keep_terms),
            "sparse_rewrite": self.sparse_rewrite,
            "dense_rewrite": self.dense_rewrite,
        }


def build_llm_query_rewrite_messages(
    *,
    question: str,
    entities: list[str],
    relations: list[str],
    constraints: list[str],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是企业知识库检索查询重写器。\n"
        "你的任务是基于问题和给定骨架槽位，生成两个面向检索的改写查询，而不是回答问题。\n\n"
        "硬性规则：\n"
        "1. 严禁回答问题，严禁补充答案、结论、命令、步骤、原因、配置值、错误处理方案。\n"
        "2. 严禁引入原问题和给定骨架中没有明确出现的具体事实。\n"
        "3. 必须保留关键术语原文：产品名、设备名、协议名、型号、版本号、错误码、命令、参数名、专有缩写。\n"
        "4. 允许补充抽象检索目标词，例如：原因、影响、条件、定义、区别、步骤、方法、验证方式、适用场景。\n"
        "5. sparse_rewrite 面向 lexical / BM25：短、关键词密集、尽量保留原词、避免虚构连接词。\n"
        "6. dense_rewrite 面向 dense / embedding：语义完整、关系明确、自然语言表达，但仍然禁止补充未给出的具体事实。\n"
        "7. 输出必须是严格 JSON 对象，不要输出 Markdown，不要输出解释。\n"
        "8. 输出字段只允许：must_keep_terms, sparse_rewrite, dense_rewrite。"
    )
    one_shot_input = {
        "question": "Cisco Catalyst 9300 在 IOS XE 17.9 下 OSPF 邻居无法建立的原因是什么？",
        "entities": ["Cisco Catalyst 9300", "IOS XE 17.9", "OSPF"],
        "relations": ["原因"],
        "constraints": ["邻居无法建立"],
    }
    one_shot_output = {
        "must_keep_terms": ["Cisco Catalyst 9300", "IOS XE 17.9", "OSPF", "邻居无法建立"],
        "sparse_rewrite": "Cisco Catalyst 9300 IOS XE 17.9 OSPF 邻居无法建立 原因",
        "dense_rewrite": "Cisco Catalyst 9300 在 IOS XE 17.9 下 OSPF 邻居无法建立的原因和排查线索",
    }
    actual_input = {
        "question": question,
        "entities": entities,
        "relations": relations,
        "constraints": constraints,
    }
    user_prompt = (
        "请参考下面的一次示例，然后处理实际输入。\n\n"
        "示例输入：\n"
        f"{json.dumps(one_shot_input, ensure_ascii=False, indent=2)}\n\n"
        "示例输出：\n"
        f"{json.dumps(one_shot_output, ensure_ascii=False, indent=2)}\n\n"
        "实际输入：\n"
        f"{json.dumps(actual_input, ensure_ascii=False, indent=2)}\n\n"
        "请只输出 JSON 对象。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_llm_query_rewrite_payload(raw_content: str) -> LLMQueryRewritePayload:
    payload = _parse_top_level_object(raw_content)
    must_keep_terms = _normalize_string_list(payload.get("must_keep_terms"))
    sparse_rewrite = _normalize_text_field(payload.get("sparse_rewrite"))
    dense_rewrite = _normalize_text_field(payload.get("dense_rewrite"))
    if not sparse_rewrite:
        raise ValueError("LLM rewrite field 'sparse_rewrite' must be a non-empty string.")
    if not dense_rewrite:
        raise ValueError("LLM rewrite field 'dense_rewrite' must be a non-empty string.")
    return LLMQueryRewritePayload(
        must_keep_terms=must_keep_terms,
        sparse_rewrite=sparse_rewrite,
        dense_rewrite=dense_rewrite,
    )


def _parse_top_level_object(raw_content: str) -> dict:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM rewrite output is not valid JSON: {_preview(raw_content)}") from exc
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM rewrite output is double-encoded but invalid: {_preview(raw_content)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LLM rewrite output must be a JSON object: {_preview(raw_content)}")
    return payload


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("LLM rewrite field 'must_keep_terms' must be a list.")
    normalized: list[str] = []
    for item in value:
        text = _normalize_text_field(item)
        if not text or text in normalized:
            continue
        normalized.append(text)
    return normalized


def _normalize_text_field(value: object) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _preview(raw_content: str, max_chars: int = 200) -> str:
    compact = " ".join(raw_content.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."
