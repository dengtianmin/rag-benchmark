# Unified Experiment Host

本目录对应第 4 章实验宿主的最小可运行版本，目标是把不同 baseline 放到统一协议、统一索引、统一 evaluator 下。

## 当前进度

- Step 1 已完成：冻结统一数据协议
- Step 2 已完成：公共底座
- Step 3 已完成：`Traditional RAG`
- Step 4 已完成：`Rewrite-RAG`
- Step 5 已完成：`Graph-enhanced RAG`
- Step 6 已完成：`KBQA baseline` 与 `Ours-Ch4` 骨架

Step 1 对应文件：

- [schema.py](/home/paper/Benchmark/src/core/schema.py)
- [types.py](/home/paper/Benchmark/src/core/types.py)
- [loaders.py](/home/paper/Benchmark/src/dataio/loaders.py)
- [normalizers.py](/home/paper/Benchmark/src/dataio/normalizers.py)
- [inspect_dataset.py](/home/paper/Benchmark/scripts/inspect_dataset.py)
- [test_schema_normalization.py](/home/paper/Benchmark/tests/test_schema_normalization.py)

后续 baseline / pipeline 代码位于：

- [traditional_rag.py](/home/paper/Benchmark/src/pipelines/traditional_rag.py)
- [rewrite_rag.py](/home/paper/Benchmark/src/pipelines/rewrite_rag.py)
- [graph_enhanced_rag.py](/home/paper/Benchmark/src/pipelines/graph_enhanced_rag.py)
- [kbqa_baseline.py](/home/paper/Benchmark/src/pipelines/kbqa_baseline.py)
- [ours_ch4.py](/home/paper/Benchmark/src/pipelines/ours_ch4.py)
- [run_ablation.py](/home/paper/Benchmark/scripts/run_ablation.py)

## 当前包含

- `Traditional RAG`
- `Rewrite-RAG`
- `Graph-enhanced RAG`
- `KBQA baseline`
- `Ours-Ch4`

## 当前工程结构

```text
src/
  core/
  dataio/
  evaluation/
  modules/
  pipelines/
  prompts/
  retrievers/
scripts/
  inspect_dataset.py
  run_traditional_rag.py
  run_rewrite_rag.py
  run_graph_enhanced_rag.py
  run_kbqa_baseline.py
  run_ours_ch4.py
  run_ablation.py
tests/
  test_schema_normalization.py
  test_traditional_rag.py
  test_rewrite_rag.py
  test_graph_enhanced_rag.py
  test_step6_kbqa_ours.py
```

## 当前约束

- 只使用本地 `jsonl` 数据
- 不依赖 `external/` 运行
- 生成器为本地 `extractive_stub`
- `Skeleton Extraction / Retrieval Rewrite / Text Evidence Compensation` 先保留为可替换组件

## Step 1 统一协议

Step 1 固定了主项目内部输入输出协议，避免后续每个 baseline 各自处理 `qid/doc_id/section_id` 映射。

### 统一输入样本协议

- `EvidenceItem`
  - 统一使用 `source_id` + `section_id` 标识证据来源
- `QuestionSkeletonLabel`
  - 保存 `question_type / entities / relations / constraints / requires_text_compensation`
- `BenchmarkSample`
  - 后续所有 baseline 的统一输入对象
- `DatasetSplit`
  - 单个 split 的样本集合
- `DatasetBundle`
  - 多 split 数据集容器

### 统一检索输出协议

- `RetrievedDocument`
  - 文本段检索结果
- `RetrievedTriple`
  - 图结构 / 三元组检索结果
- `RetrievalResult`
  - 单次检索结果容器
- `AnswerResult`
  - 回答结果容器
- `PipelineRunRecord`
  - 单条问题完整运行记录，后续 evaluator 可直接消费

### 统一约束

- `question_id` 必须存在
- 所有证据必须能映射到 `source_id` 和 `section_id`
- `entities / relations / constraints` 可以为空，但字段必须存在
- `answer_short / answer_long` 至少一个非空
- `PipelineRunRecord` 保持 evaluator 兼容边界

### 归一化策略

当前数据集中的字段会被归一化为项目内部协议：

- `qid -> question_id`
- `doc_id -> source_id`
- `evidence[].doc_id -> evidence[].source_id`
- 原始记录保留在 `BenchmarkSample.raw_record`

这样做的目的是让主工程内部协议稳定，而不是把已有 jsonl 原样透传到后续模块。

## Step 1 使用方式

查看数据集统计：

```bash
python scripts/inspect_dataset.py --dataset outputs/two_file_demo/benchmark_dataset.jsonl
```

运行协议层测试：

```bash
pytest tests/test_schema_normalization.py -q
```

## Step 3 到 Step 6 运行方式

### Traditional RAG

```bash
python scripts/run_traditional_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --top-k 5 \
  --output-dir outputs/experiments/traditional_rag
```

### Rewrite-RAG

```bash
python scripts/run_rewrite_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --mode entity_relation \
  --top-k 5 \
  --output-dir outputs/experiments/rewrite_rag
```

### Graph-enhanced RAG

```bash
python scripts/run_graph_enhanced_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --seed-top-k 5 \
  --expand-k 5 \
  --top-k 5 \
  --output-dir outputs/experiments/graph_enhanced_rag
```

### KBQA baseline

```bash
python scripts/run_kbqa_baseline.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --entity-mode gold \
  --relation-mode gold \
  --output-dir outputs/experiments/kbqa_baseline
```

### Ours-Ch4

```bash
python scripts/run_ours_ch4.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --skeleton-mode oracle \
  --output-dir outputs/experiments/ours_ch4
```

### Ablation

```bash
python scripts/run_ablation.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --output-dir outputs/experiments/ablation
```

`run_ablation.py` 当前统一运行以下变体：

- `full`
- `w/o_relation_driven`
- `w/o_skeleton_rewrite`
- `w/o_text_compensation`

## Baseline 边界

### Traditional RAG

- 纯文本检索对照组
- 不做 rewrite
- 不做图扩展

### Rewrite-RAG

- 在检索前进行 query rewrite
- 支持 `naive` 和 `entity_relation` 两种模式
- 用于观察 rewrite 是否提升 evidence 覆盖

### Graph-enhanced RAG

- 借鉴 `KG2RAG` 的 `semantic seed retrieval + graph-guided expansion + organization` 思想
- 图结构来自 `knowledge_extraction.jsonl`
- 不是 Ours-Ch4

### KBQA baseline

- 轻量 `entity linking -> relation matching -> subgraph execution`
- 知识源是抽取子图，不是外部标准知识库
- 用于对比结构化问答上界

### Ours-Ch4

- 对应论文第 4 章三模块：
  - `Relation-driven Retrieval`
  - `Skeleton Extraction & Retrieval Rewrite`
  - `Text Evidence Compensation`
- 当前以骨架版本实现，便于后续替换真实模块

## 当前测试

默认测试入口：

```bash
pytest -q
```

当前主项目测试已经通过 `pytest.ini` 排除了 `external/` 目录，因此默认只验证主工程自身代码。

## 当前目录建议

```text
src/
  core/
    schema.py
    types.py
  dataio/
    loaders.py
    normalizers.py
  benchmark_host/
tests/
  test_schema_normalization.py
scripts/
  inspect_dataset.py
```

## 为什么要先做 Step 1

- 后续 baseline 可以共享同一套输入输出协议
- evaluator 不需要为不同 baseline 写多套适配层
- 后续接入 `external/` 方法思想时，只需要写 adapter，不会污染主项目数据格式
- 消融实验时可以稳定记录 `retrieval -> answer -> trace`

## 与 external 的关系

- 本步没有直接依赖任何 `external/` 项目
- 只借鉴了类似 `haystack` 的抽象思想：组件之间通过清晰协议连接
- 没有复制任何 external 代码，也没有把 external 数据格式引入主项目

## 运行

```bash
pip install -e .
bash scripts/run_ch4_demo.sh
```

输出位于 `outputs/experiments/ch4_demo/`。
