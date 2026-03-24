# 当前运行报告

## 1. 报告说明

本文档记录当前项目在编写文档时的运行状态，分为：

- 两文件 demo 完整结果
- 全量运行实时快照

这份报告不是固定不变的论文最终统计，而是当前工作区可复核的运行记录。

## 2. 两文件 demo 完整结果

目录：

- [`artifacts/two_file_demo/`](/home/paper/Benchmark/artifacts/two_file_demo)
- [`outputs/two_file_demo/`](/home/paper/Benchmark/outputs/two_file_demo)

### 2.1 规模

- 文档数：2
- section 数：17
- 候选 QA 数：53
- 校验结果数：53
- 最终样本数：51

### 2.2 题型分布

- `fact`: 16
- `relation`: 16
- `multi_evidence`: 8
- `explanation`: 11

### 2.3 其他统计

- 平均 evidence 数：1.4902
- `source_scope = single_section`: 51
- `requires_text_compensation = False`: 51

### 2.4 当前观察

- 题型基本覆盖四类
- 已能生成多证据问题
- 但 `requires_text_compensation` 还没有体现出来
- 一部分样本偏简单，仍需进一步加强筛选

## 3. 全量运行实时快照

目录：

- [`artifacts/full_run/`](/home/paper/Benchmark/artifacts/full_run)
- [`outputs/full_run/`](/home/paper/Benchmark/outputs/full_run)

### 3.1 当前状态

在本文档写入时，全量任务处于：

- 知识抽取已完成
- QA 生成进行中
- QA 校验尚未开始
- 最终导出尚未完成

### 3.2 当前规模快照

- Markdown 文档数：80
- section 数：1129
- 知识抽取结果数：1129
- 候选 QA 数：605
- QA 生成进度记录数：179

### 3.3 已生成文件

已存在：

- [`artifacts/full_run/markdown_sections.jsonl`](/home/paper/Benchmark/artifacts/full_run/markdown_sections.jsonl)
- [`artifacts/full_run/knowledge_extraction.jsonl`](/home/paper/Benchmark/artifacts/full_run/knowledge_extraction.jsonl)
- [`artifacts/full_run/qa_candidates.jsonl`](/home/paper/Benchmark/artifacts/full_run/qa_candidates.jsonl)
- [`artifacts/full_run/qa_generation_progress.jsonl`](/home/paper/Benchmark/artifacts/full_run/qa_generation_progress.jsonl)

尚未生成完成：

- `artifacts/full_run/qa_validation.jsonl`
- `outputs/full_run/benchmark_dataset.jsonl`
- `outputs/full_run/benchmark_dataset.json`
- `outputs/full_run/benchmark_summary.csv`

## 4. 目前暴露出的质量问题

基于当前运行日志和样本快照，已经能看到以下问题：

### 4.1 简单事实题比例仍偏高

例如：

- 地址
- 型号
- 功耗
- 部署环境

这类题适合做基础测试，但对“关系驱动检索 / 文本证据补偿”的代表性还不够强。

### 4.2 `requires_text_compensation` 标注偏弱

当前 demo 结果中全部为 `False`，这说明：

- schema 已经设计好
- 但 prompt 和规则还没有把这一维度真正打出来

### 4.3 超长 section 的模型截断问题

在全量日志中，个别大规格表或长列表 section 出现：

- `finish_reason: length`

这意味着某些抽取结果可能不完整。

### 4.4 大量样本仍是 `single_section`

当前默认配置下，样本主要依赖单 section，这有利于稳定，但会限制更复杂的多跳或跨证据检索实验。

## 5. 推荐的下一步优化

### 5.1 强化题型约束

优先提升以下样本占比：

- `relation`
- `multi_evidence`
- `explanation`

并进一步压低过于直接的事实题比例。

### 5.2 强化 `requires_text_compensation`

建议加入更严格判定规则，例如：

- 问题必须同时涉及关系与限定条件
- 答案不能被单个属性值直接命中
- 必须依赖原文说明性语句或归纳性证据

### 5.3 针对超长 section 做二次拆分

对特别长的表格 section 或超长规格块，可进一步按表格行组、段落簇、字符长度做二次切分，降低截断概率。

### 5.4 收紧校验 prompt

当前校验阶段通过率偏高，后续可以更明确拒绝：

- 仅做简单摘抄的问题
- 实体约束不够强的问题
- 过于模板化的问题

## 6. 如何更新本报告

当全量运行结束后，建议补写以下最终统计：

- 候选 QA 总数
- 校验结果总数
- 最终样本总数
- 各题型最终分布
- 每文档贡献数前若干名
- `single_section/single_doc/cross_doc` 分布
- `requires_text_compensation` 分布
- 平均 evidence 数量

如果后续需要把这份报告整理为论文附录，可以在此基础上再压缩成正式统计版。
