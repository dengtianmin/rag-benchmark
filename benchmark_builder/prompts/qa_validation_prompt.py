SYSTEM_PROMPT = """你是 benchmark QA 审核器。

你需要根据问题、答案和证据判断样本是否适合作为评测数据。
只依据给定 evidence 和 section 内容，不允许脑补。
输出必须是严格 JSON，不要输出解释文字。
"""


USER_PROMPT_TEMPLATE = """请审核下面的 QA 候选样本。

审核标准：
1. answer 是否被 evidence 直接支撑。
2. evidence 是否足够完整，是否遗漏关键条件。
3. question_type 是否与样本实际推理形式一致。
4. question 是否清晰、非平凡、适合 benchmark。
5. answer 是否存在幻觉或超出 evidence 的新事实。
6. 对不通过样本，reject_reasons 要写清楚。

拒绝条件：
- evidence 为空或明显不足
- 题型标注错误
- 问题过于宽泛或过于简单
- answer 引入 evidence 之外的新事实
- question 与 evidence 不对应

输出 JSON schema：
{{
  "qid": "string",
  "is_valid": true,
  "support_score": 0.0,
  "completeness_score": 0.0,
  "type_consistent": true,
  "is_nontrivial": true,
  "has_hallucination": false,
  "reject_reasons": ["string"]
}}

候选 QA：
{qa_json}

所属 section 原文：
{content}
"""
