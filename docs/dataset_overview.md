# Benchmark 数据集概览

## 1. 数据集用途

本项目构建的是一个面向企业产品知识库问答的 benchmark 数据集，主要用于支持以下研究方向：

- 知识图谱引导检索
- 关系驱动检索
- 文本证据补偿
- 企业产品文档问答评测

该数据集不是开放领域百科问答，而是基于企业产品文档构建的高约束 QA 数据。问题与答案要求被文档证据直接支撑，并经过二次自动校验后才进入最终集合。

## 2. 数据来源

数据源位于项目根目录下的 [`data/`](/home/paper/Benchmark/data)。

当前数据源特点：

- 全部为 Markdown 文档
- 每个 `.md` 文件通常对应一份产品彩页、产品规格页、部署说明或功能说明
- 文档内容可能包含：
  - 标题层级
  - 普通段落
  - 列表
  - HTML 表格
  - FAQ 风格说明
  - 产品规格与参数
  - 部署与运维描述
  - 安全能力、协议、组件、功能清单

## 3. 数据集构造目标

本 benchmark 试图覆盖的不只是简单事实问答，还包括：

- 基于实体与属性的事实提问
- 基于实体间关系的关系型提问
- 需要多个证据片段共同支撑的多证据提问
- 需要归纳和解释的说明型提问

因此，最终样本会保留如下核心字段：

- `question`
- `answer_short`
- `answer_long`
- `question_type`
- `entities`
- `relations`
- `constraints`
- `evidence`
- `source_scope`
- `requires_text_compensation`

## 4. 当前构造流程

数据集构造采用“四阶段流水线”：

1. Markdown 解析与 section 切分
2. 基于每个 section 的知识抽取
3. 基于 section 原文和抽取结果生成候选 QA
4. 对候选 QA 做结构化证据校验并导出最终数据集

对应的 CLI 命令为：

```bash
python main.py parse-md
python main.py extract-knowledge
python main.py generate-qa
python main.py validate-qa
python main.py build-dataset
python main.py run-all
```

## 5. 当前可用数据结果

### 5.1 两文件 demo 完整结果

两文件 demo 的完整输出位于：

- [`artifacts/two_file_demo/`](/home/paper/Benchmark/artifacts/two_file_demo)
- [`outputs/two_file_demo/`](/home/paper/Benchmark/outputs/two_file_demo)

已完成结果：

- 解析 section 数：17
- 候选 QA 数：53
- 校验结果数：53
- 最终样本数：51

题型分布：

- `fact`: 16
- `relation`: 16
- `multi_evidence`: 8
- `explanation`: 11

其他统计：

- 平均 evidence 数量：1.4902
- `source_scope`: 全部为 `single_section`
- `requires_text_compensation`: 当前全部为 `False`

### 5.2 全量运行快照

全量运行目录为：

- [`artifacts/full_run/`](/home/paper/Benchmark/artifacts/full_run)
- [`outputs/full_run/`](/home/paper/Benchmark/outputs/full_run)

本文档编写时的快照状态为：

- Markdown 文档数：80
- section 数：1129
- 知识抽取结果数：1129
- QA 候选数：605
- QA 生成进度记录数：179

说明：

- 该全量任务仍在运行中
- 目前已完成知识抽取，并进入 QA 生成阶段
- 最终的 `qa_validation.jsonl`、`benchmark_dataset.jsonl` 和汇总统计尚未生成完成

## 6. 数据集的边界

该 benchmark 当前的设计边界如下：

- 只基于企业产品文档构造，不依赖外部知识库
- 问答必须由文档证据支撑
- 默认优先保留单 section 可验证样本
- 当前跨 section / 跨文档样本能力在配置上预留，但默认未启用

## 7. 当前已知不足

当前结果还存在一些需要后续优化的点：

- `requires_text_compensation` 标注偏弱，当前多数样本未体现这一研究维度
- 一部分问题仍偏简单，尤其是地址、型号、参数类事实题
- 大多数样本仍是 `single_section`
- 个别超长规格页会触发模型输出长度截断，影响抽取完整性

## 8. 建议阅读顺序

建议按照以下顺序阅读文档：

1. [`dataset_overview.md`](/home/paper/Benchmark/docs/dataset_overview.md)
2. [`build_pipeline.md`](/home/paper/Benchmark/docs/build_pipeline.md)
3. [`dataset_schema.md`](/home/paper/Benchmark/docs/dataset_schema.md)
4. [`current_run_report.md`](/home/paper/Benchmark/docs/current_run_report.md)
