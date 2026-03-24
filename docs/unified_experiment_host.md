# Unified Experiment Host

本目录对应第 4 章实验宿主的最小可运行版本，目标是把不同 baseline 放到统一协议、统一索引、统一 evaluator 下。

## 当前进度

- Step 1 已完成：冻结统一数据协议
- Step 2 及之后的 baseline / pipeline 骨架已存在于 `src/benchmark_host/`

Step 1 对应文件：

- [schema.py](/home/paper/Benchmark/src/core/schema.py)
- [types.py](/home/paper/Benchmark/src/core/types.py)
- [loaders.py](/home/paper/Benchmark/src/dataio/loaders.py)
- [normalizers.py](/home/paper/Benchmark/src/dataio/normalizers.py)
- [inspect_dataset.py](/home/paper/Benchmark/scripts/inspect_dataset.py)
- [test_schema_normalization.py](/home/paper/Benchmark/tests/test_schema_normalization.py)

## 当前包含

- `Traditional RAG`
- `Rewrite-RAG`
- `Graph-enhanced RAG`
- `KBQA baseline`
- `Ours-Ch4`

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
