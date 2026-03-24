SYSTEM_PROMPT = """你是 benchmark 问答样本构建器。

你要基于结构化知识和原文证据，生成受约束、可验证、适合企业知识库问答评测的 QA。
绝对禁止编造证据之外的事实。输出必须是严格 JSON 对象，不要输出解释。
"""


USER_PROMPT_TEMPLATE = """请基于以下 section 内容和知识抽取结果，生成候选 QA。

生成约束：
1. 问题类型只允许：fact、relation、multi_evidence、explanation。
2. fact：单一事实或属性，不要过泛。
3. relation：必须体现实体之间的关系约束，不能退化成普通属性题。
4. multi_evidence：必须至少依赖两个证据。
5. explanation：答案需要归纳解释，不能只是摘抄一个短句。
6. 每条 QA 必须能被 evidence 支撑。
7. 若 require_evidence 为 true，则每条 QA 必须提供 evidence。
8. 若 require_strong_constraints 为 true，则尽量让问题包含实体、关系、条件三者中的多个元素。
9. 不要生成“请介绍”“请简述”之类过泛问题。
10. 不要输出重复或改写式重复问题。

控制参数：
- max_per_section: {max_per_section}
- type_quota: {type_quota}
- allow_cross_section: {allow_cross_section}
- allow_cross_document: {allow_cross_document}
- require_strong_constraints: {require_strong_constraints}
- require_evidence: {require_evidence}

输出 JSON schema：
{{
  "items": [
    {{
      "question": "string",
      "answer_short": "string",
      "answer_long": "string",
      "question_type": "fact|relation|multi_evidence|explanation",
      "entities": ["string"],
      "relations": ["string"],
      "constraints": ["string"],
      "requires_text_compensation": true,
      "evidence": [
        {{
          "doc_id": "string",
          "section_id": "string",
          "quote": "string"
        }}
      ],
      "source_scope": "single_section|single_doc|cross_doc"
    }}
  ]
}}

当前 section 元数据：
- doc_id: {doc_id}
- section_id: {section_id}
- doc_title: {doc_title}
- section_path: {section_path}

Section 原文：
{content}

知识抽取结果：
{knowledge_json}
"""
