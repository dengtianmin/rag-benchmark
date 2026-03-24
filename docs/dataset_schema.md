# Benchmark 数据格式与 Schema

## 1. 最终输出文件

最终数据集计划输出到 [`outputs/`](/home/paper/Benchmark/outputs) 下，包含：

- `benchmark_dataset.jsonl`
- `benchmark_dataset.json`
- `benchmark_summary.csv`

当前完整可查看样本位于：

- [`outputs/two_file_demo/benchmark_dataset.jsonl`](/home/paper/Benchmark/outputs/two_file_demo/benchmark_dataset.jsonl)
- [`outputs/two_file_demo/benchmark_dataset.json`](/home/paper/Benchmark/outputs/two_file_demo/benchmark_dataset.json)
- [`outputs/two_file_demo/benchmark_summary.csv`](/home/paper/Benchmark/outputs/two_file_demo/benchmark_summary.csv)

## 2. 最终样本记录格式

每条最终样本对应 `FinalBenchmarkRecord`，字段如下。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `qid` | `str` | 样本唯一标识 |
| `question` | `str` | 问题文本 |
| `answer_short` | `str` | 简短答案 |
| `answer_long` | `str` | 长答案，通常包含解释性展开 |
| `question_type` | `fact/relation/multi_evidence/explanation` | 问题类型 |
| `entities` | `list[str]` | 涉及的实体 |
| `relations` | `list[str]` | 涉及的关系 |
| `constraints` | `list[str]` | 限制条件、范围、阈值等 |
| `requires_text_compensation` | `bool` | 是否需要文本补偿推理 |
| `evidence` | `list[object]` | 支撑证据列表 |
| `source_scope` | `single_section/single_doc/cross_doc` | 证据来源范围 |
| `doc_id` | `str` | 来源文档 ID |
| `section_id` | `str` | 来源 section ID |
| `support_score` | `float` | 证据支撑分 |
| `completeness_score` | `float` | 证据完整性分 |

其中 `evidence` 的子结构如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `doc_id` | `str` | 证据来源文档 ID |
| `section_id` | `str` | 证据来源 section ID |
| `quote` | `str` | 原文证据摘录 |

## 3. question_type 的含义

### `fact`

用于测试单一事实、单一属性、明确参数或配置项。

典型例子：

- 某型号的功耗是多少
- 某产品支持什么协议
- 某公司所在地是哪里

### `relation`

用于测试实体与实体、实体与能力、实体与条件之间的关系。

典型例子：

- 某产品通过什么机制实现某能力
- 某平台与某服务之间是什么关系
- 某功能如何支撑某业务目标

### `multi_evidence`

用于测试需要两个或以上证据共同支撑的问题。

典型例子：

- 同时询问“适用环境 + 支持功能”
- 同时询问“对象 + 原因”
- 同时归纳两个证据片段才能回答的问题

### `explanation`

用于测试归纳说明类问题。

典型例子：

- 为什么某方案能提升可靠性
- 某功能如何实现冷热数据管理
- 某安全能力如何构成完整防护链路

## 4. 中间产物格式

### 4.1 `markdown_sections.jsonl`

对应 `MarkdownSection`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `doc_id` | `str` | 文档唯一 ID |
| `file_path` | `str` | 原始 Markdown 路径 |
| `doc_title` | `str` | 文档标题 |
| `section_id` | `str` | section 唯一 ID |
| `block_id` | `str` | 当前实现中与 `section_id` 一致 |
| `section_path` | `list[str]` | 标题层级路径 |
| `content` | `str` | section 原文 |
| `block_type` | `paragraph/list/table/mixed` | 块类型 |
| `char_len` | `int` | 文本长度 |

### 4.2 `knowledge_extraction.jsonl`

对应 `KnowledgeExtractionResult`。

主要字段：

- `doc_id`
- `section_id`
- `section_path`
- `content`
- `extraction`
- `raw_response_text`
- `parse_error`
- `success`

其中 `extraction` 下包含：

- `entities`
- `relations`
- `constraints`
- `attributes`
- `procedures`
- `evidence_spans`

### 4.3 `qa_candidates.jsonl`

对应 `QACandidate`。

字段基本与最终样本一致，但尚未带：

- `support_score`
- `completeness_score`

### 4.4 `qa_validation.jsonl`

对应 `QAValidationResult`。

字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `qid` | `str` | 候选 QA ID |
| `is_valid` | `bool` | 是否通过 |
| `support_score` | `float` | 证据支撑分 |
| `completeness_score` | `float` | 证据完整性分 |
| `type_consistent` | `bool` | 题型是否一致 |
| `is_nontrivial` | `bool` | 是否非平凡 |
| `has_hallucination` | `bool` | 是否存在幻觉 |
| `reject_reasons` | `list[str]` | 拒绝原因 |
| `raw_response_text` | `str` | 模型原始返回 |
| `parse_error` | `str \| null` | 解析错误信息 |

### 4.5 `qa_generation_progress.jsonl`

这是为了支持断点续跑新增的 section 级进度文件。

字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `section_id` | `str` | 已处理的 section |
| `status` | `str` | 当前固定为 `completed` |
| `qa_count` | `int` | 该 section 生成的 QA 数量 |

它的作用是记录：

- 某个 section 已经做过 QA 生成
- 即使该 section 最终生成了 `0` 条 QA，也不会在下次 `--resume` 时重复调用模型

## 5. 枚举字段说明

### `source_scope`

- `single_section`: 证据全部来自同一 section
- `single_doc`: 证据来自同一文档的多个 section
- `cross_doc`: 证据来自多个文档

当前默认配置下，多数样本是 `single_section`。

### `requires_text_compensation`

该字段用于标记问题是否需要超越单一结构化关系、依赖文本细节进行补偿性理解。

当前实现中：

- 字段已经保留在 schema 中
- 但标注策略还偏弱
- 当前运行结果里大量样本为 `False`

## 6. 汇总统计格式

`benchmark_summary.csv` 采用简单的 `metric,value` 结构。

示例：

- `total_samples`
- `avg_evidence_count`
- `question_type::fact`
- `question_type::relation`
- `doc_id::<某文档ID>`
- `source_scope::single_section`
- `requires_text_compensation::False`
