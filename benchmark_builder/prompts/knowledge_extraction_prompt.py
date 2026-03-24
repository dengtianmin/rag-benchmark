SYSTEM_PROMPT = """你是企业产品文档知识抽取器。

你的任务是从给定 Markdown section 中抽取结构化知识，只能依据输入文本，不允许补充常识，不允许编造。
如果某类信息不存在，返回空数组，不要输出解释。
输出必须是严格 JSON 对象，不要使用 Markdown 代码块。
"""


USER_PROMPT_TEMPLATE = """请从下面的企业产品文档片段中抽取结构化知识。

要求：
1. 只依据输入内容抽取，不允许引入证据外的新事实。
2. evidence_spans.quote 必须是原文中的连续片段或等价短摘录。
3. procedures 仅在文本出现明确步骤、流程、阶段时输出。
4. constraints 用于记录阈值、版本要求、条件限制、适用范围等。
5. attributes 用于记录实体属性和对应值。
6. relation 的 head 和 tail 必须来自文本中的实体或可直接定位的对象。
7. 无法确定时返回空数组。

输出 JSON schema：
{{
  "entities": [
    {{"name": "string", "type": "string", "normalized_name": "string"}}
  ],
  "relations": [
    {{"head": "string", "relation": "string", "tail": "string", "description": "string"}}
  ],
  "constraints": [
    {{"target": "string", "type": "string", "value": "string"}}
  ],
  "attributes": [
    {{"entity": "string", "attribute": "string", "value": "string"}}
  ],
  "procedures": [
    {{"name": "string", "steps": ["string"]}}
  ],
  "evidence_spans": [
    {{"quote": "string", "reason": "string"}}
  ]
}}

文档标题：{doc_title}
Section 路径：{section_path}
Section 内容：
{content}
"""
