# Benchmark 构造流程说明

## 1. 流水线总览

本项目采用面向批处理的流水线构造方法，从企业产品 Markdown 文档出发，经过解析、知识抽取、QA 生成、QA 校验四个阶段，最终得到结构化 benchmark 数据集。

对应代码入口位于：

- [`main.py`](/home/paper/Benchmark/main.py)
- [`benchmark_builder/cli.py`](/home/paper/Benchmark/benchmark_builder/cli.py)

## 2. 阶段一：Markdown 解析与切分

实现位置：

- [`benchmark_builder/pipelines/parse_markdown.py`](/home/paper/Benchmark/benchmark_builder/pipelines/parse_markdown.py)
- [`benchmark_builder/utils/markdown_parser.py`](/home/paper/Benchmark/benchmark_builder/utils/markdown_parser.py)

处理逻辑：

1. 递归扫描 `data/` 下的所有 `.md`
2. 使用标题层级切分 section
3. 保留 `section_path`
4. 去掉图片链接噪声
5. 根据内容粗粒度识别 `paragraph/list/table/mixed`
6. 生成稳定的 `doc_id` 和 `section_id`

输出：

- [`artifacts/markdown_sections.jsonl`](/home/paper/Benchmark/artifacts/markdown_sections.jsonl)

## 3. 阶段二：知识抽取

实现位置：

- [`benchmark_builder/pipelines/extract_knowledge.py`](/home/paper/Benchmark/benchmark_builder/pipelines/extract_knowledge.py)
- [`benchmark_builder/clients/deepseek_client.py`](/home/paper/Benchmark/benchmark_builder/clients/deepseek_client.py)
- [`benchmark_builder/prompts/knowledge_extraction_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/knowledge_extraction_prompt.py)

处理逻辑：

1. 读取每个 section 的原文
2. 调用 DeepSeek 模型抽取结构化知识
3. 约束模型输出为严格 JSON
4. 用 Pydantic 校验返回结构
5. 失败时记录原始文本和错误信息

抽取内容包括：

- 实体 `entities`
- 关系 `relations`
- 约束 `constraints`
- 属性 `attributes`
- 流程 `procedures`
- 证据片段 `evidence_spans`

输出：

- [`artifacts/full_run/knowledge_extraction.jsonl`](/home/paper/Benchmark/artifacts/full_run/knowledge_extraction.jsonl)

## 4. 阶段三：候选 QA 生成

实现位置：

- [`benchmark_builder/pipelines/generate_qa.py`](/home/paper/Benchmark/benchmark_builder/pipelines/generate_qa.py)
- [`benchmark_builder/prompts/qa_generation_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/qa_generation_prompt.py)

处理逻辑：

1. 读取 section 原文和知识抽取结果
2. 对每个 section 调用 LLM 生成受约束的候选 QA
3. 限制题型为四类：
   - `fact`
   - `relation`
   - `multi_evidence`
   - `explanation`
4. 应用基础规则过滤不合格样本
5. 对问题做去重和近似去重

当前过滤规则包括：

- 删除空 `question`、`answer_short`、`answer_long`
- 删除 `evidence` 为空的样本
- `multi_evidence` 至少 2 条证据
- `relation` 必须有 `relations`
- `explanation` 长答案不能过短
- 证据与答案明显无关时剔除

输出：

- [`artifacts/full_run/qa_candidates.jsonl`](/home/paper/Benchmark/artifacts/full_run/qa_candidates.jsonl)

## 5. 阶段四：QA 校验

实现位置：

- [`benchmark_builder/pipelines/validate_qa.py`](/home/paper/Benchmark/benchmark_builder/pipelines/validate_qa.py)
- [`benchmark_builder/prompts/qa_validation_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/qa_validation_prompt.py)

处理逻辑：

1. 读取候选 QA 和来源 section
2. 调用 DeepSeek 进行结构化审查
3. 评分并判断是否保留

审查维度包括：

- 答案是否被 evidence 支撑
- evidence 是否足够完整
- `question_type` 是否一致
- 问题是否清晰、非平凡
- 是否存在幻觉
- 是否适合作为 benchmark 样本

输出：

- `artifacts/qa_validation.jsonl`

## 6. 阶段五：最终导出

实现位置：

- [`benchmark_builder/pipelines/build_dataset.py`](/home/paper/Benchmark/benchmark_builder/pipelines/build_dataset.py)

处理逻辑：

1. 读取候选 QA
2. 只保留校验通过样本
3. 补充验证分数字段
4. 生成最终 JSONL / JSON / CSV 输出

输出：

- `outputs/benchmark_dataset.jsonl`
- `outputs/benchmark_dataset.json`
- `outputs/benchmark_summary.csv`

## 7. 模型调用与 Prompt 设计

Prompt 分别独立维护在：

- [`knowledge_extraction_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/knowledge_extraction_prompt.py)
- [`qa_generation_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/qa_generation_prompt.py)
- [`qa_validation_prompt.py`](/home/paper/Benchmark/benchmark_builder/prompts/qa_validation_prompt.py)

DeepSeek Client 支持：

- `system` / `user` prompt
- JSON 输出模式
- 重试
- 原始响应记录
- 解析失败保留

## 8. 断点续跑机制

### 8.1 已支持复用的阶段

- `extract-knowledge`
  - 根据已存在的 `knowledge_extraction.jsonl` 按 `section_id` 跳过
- `generate-qa`
  - 根据 `qa_generation_progress.jsonl` 按 `section_id` 跳过
  - 即使某个 section 生成 0 条 QA，也会被视为已处理
- `validate-qa`
  - 根据 `qa_validation.jsonl` 按 `qid` 跳过

### 8.2 为什么需要 `qa_generation_progress.jsonl`

如果只根据 `qa_candidates.jsonl` 判断是否处理过，某个 section 若生成 0 条 QA，就不会留下痕迹，下次 `--resume` 还会重复请求模型。

为解决这个问题，项目额外保存：

- [`artifacts/full_run/qa_generation_progress.jsonl`](/home/paper/Benchmark/artifacts/full_run/qa_generation_progress.jsonl)

### 8.3 增量落盘

当前实现已改为：

- 知识抽取增量落盘
- QA 生成增量落盘
- QA 校验增量落盘

这意味着：

- 中途停止不会整阶段丢失结果
- 后续可直接继续跑

## 9. 当前方法的优点

- 工程化程度高，适合批量处理
- 结构化中间产物可审计
- 支持断点续跑
- Prompt 独立维护，便于迭代
- schema 明确，适合后续扩展

## 10. 当前方法的局限

- 超长 section 可能导致模型输出截断
- 当前大多数样本还是 `single_section`
- `requires_text_compensation` 还没有被充分打出来
- 当前校验通过率偏高，说明审查 prompt 仍可继续收紧
