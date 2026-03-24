# Benchmark Dataset Builder

一个面向企业产品知识库问答 benchmark 构建与实验复现的 Python 工程。项目当前包含两部分：

- `benchmark_builder/`
  - 数据集构造流水线，从 `data/` 中的 Markdown 产品文档出发，完成 section 切分、知识抽取、受约束 QA 生成、证据一致性校验和最终数据集导出。
- `src/`
  - 统一实验宿主的研究代码。当前已完成 Step 1 到 Step 6，已具备：
    - `Traditional RAG`
    - `Rewrite-RAG`
    - `Graph-enhanced RAG`
    - `KBQA baseline`
    - `Ours-Ch4`
    - `Ablation runner`

## 功能概览

- 递归读取 `data/` 下全部 Markdown 文档
- 按标题层级切分 section，保留 `section_path`
- 使用 DeepSeek API 抽取实体、关系、约束、属性、流程和证据片段
- 基于抽取结果生成四类 QA：`fact`、`relation`、`multi_evidence`、`explanation`
- 对 QA 做 LLM 二次证据校验，过滤幻觉、证据不足和类型错误样本
- 输出 JSONL / JSON / CSV 三种结果文件
- 支持日志、中间缓存、重试、断点续跑、dry-run 和并发配置

## 安装方式

```bash
/home/huang/miniconda3/bin/conda create -y -n paper_benchmark python=3.11
source /home/huang/miniconda3/etc/profile.d/conda.sh
conda activate paper_benchmark
pip install -r requirements.txt
```

如果当前 shell 里已经可以直接使用 `conda`，也可以执行：

```bash
conda create -y -n paper_benchmark python=3.11
conda activate paper_benchmark
pip install -r requirements.txt
```

当前项目已按上述方式完成环境创建和依赖安装，环境名为 `paper_benchmark`。

## 环境变量配置

复制 `.env.example` 为 `.env` 并填写：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

默认从环境变量读取，也可以修改 `config/default_config.json` 中的工程参数。

## 目录结构

```text
.
├── benchmark_builder
│   ├── clients
│   ├── pipelines
│   ├── prompts
│   └── utils
├── src
│   ├── core
│   ├── dataio
│   ├── evaluation
│   ├── modules
│   ├── pipelines
│   ├── prompts
│   └── retrievers
├── tests
├── config
│   └── default_config.json
├── configs
├── data
├── artifacts
├── outputs
├── logs
├── scripts
├── main.py
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── README.md
```

## 统一实验宿主

第 4 章统一实验宿主当前已经落地最小骨架，说明文档见：

- [unified_experiment_host.md](/home/paper/Benchmark/docs/unified_experiment_host.md)

统一协议和数据接入代码位于：

- [schema.py](/home/paper/Benchmark/src/core/schema.py)
- [types.py](/home/paper/Benchmark/src/core/types.py)
- [loaders.py](/home/paper/Benchmark/src/dataio/loaders.py)
- [normalizers.py](/home/paper/Benchmark/src/dataio/normalizers.py)

实验宿主核心入口当前包括：

- [traditional_rag.py](/home/paper/Benchmark/src/pipelines/traditional_rag.py)
- [rewrite_rag.py](/home/paper/Benchmark/src/pipelines/rewrite_rag.py)
- [graph_enhanced_rag.py](/home/paper/Benchmark/src/pipelines/graph_enhanced_rag.py)
- [kbqa_baseline.py](/home/paper/Benchmark/src/pipelines/kbqa_baseline.py)
- [ours_ch4.py](/home/paper/Benchmark/src/pipelines/ours_ch4.py)

## 使用命令

单步执行：

```bash
conda activate paper_benchmark
python main.py parse-md
python main.py extract-knowledge
python main.py generate-qa
python main.py validate-qa
python main.py build-dataset
```

全流程执行：

```bash
conda activate paper_benchmark
python main.py run-all
```

常用参数：

```bash
conda activate paper_benchmark
python main.py run-all --input-dir data --output-dir outputs --max-files 20 --concurrency 8 --resume --dry-run
```

如果你不想先 `conda activate`，也可以直接使用：

```bash
conda run -n paper_benchmark python main.py run-all
```

## Step 1: Inspect Dataset

统一协议层提供了一个最小检查脚本，用于快速验证 `benchmark_dataset.jsonl` 是否能被主项目 schema 正常接收：

```bash
python scripts/inspect_dataset.py --dataset outputs/two_file_demo/benchmark_dataset.jsonl
```

它会输出：

- 样本总数
- `question_type` 分布
- `source_scope` 分布
- `requires_text_compensation` 分布
- 随机 3 条样本预览

如果你要运行协议层单元测试：

```bash
pytest tests/test_schema_normalization.py -q
```

## 研究型 Baseline 运行

下面这些脚本都直接使用本项目的 benchmark schema 和本地 `jsonl` 数据，不依赖 `external/` 目录运行。

Traditional RAG:

```bash
python scripts/run_traditional_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --top-k 5 \
  --output-dir outputs/experiments/traditional_rag
```

Rewrite-RAG:

```bash
python scripts/run_rewrite_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --mode entity_relation \
  --top-k 5 \
  --output-dir outputs/experiments/rewrite_rag
```

Graph-enhanced RAG:

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

KBQA baseline:

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

Ours-Ch4:

```bash
python scripts/run_ours_ch4.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --skeleton-mode oracle \
  --output-dir outputs/experiments/ours_ch4
```

Ablation:

```bash
python scripts/run_ablation.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --output-dir outputs/experiments/ablation
```

所有实验脚本统一输出：

- `predictions.jsonl`
- `metrics.json`

其中 `run_ablation.py` 会为各变体分别输出结果，并额外生成 `comparison.json`。

## 测试

项目默认已经通过 [pytest.ini](/home/paper/Benchmark/pytest.ini) 忽略 `external/` 目录，因此直接运行：

```bash
pytest -q
```

当前默认测试覆盖：

- schema / loader / normalization
- Traditional RAG
- Rewrite-RAG
- Graph-enhanced RAG
- KBQA baseline
- Ours-Ch4

## 每一步输出说明

- `parse-md`
  - 输出 `artifacts/markdown_sections.jsonl`
  - 每行一个 `MarkdownSection`
- `extract-knowledge`
  - 输出 `artifacts/knowledge_extraction.jsonl`
  - 每行一个 `KnowledgeExtractionResult`
- `generate-qa`
  - 输出 `artifacts/qa_candidates.jsonl`
  - 每行一个 `QACandidate`
- `validate-qa`
  - 输出 `artifacts/qa_validation.jsonl`
  - 每行一个 `QAValidationResult`
- `build-dataset`
  - 输出 `outputs/benchmark_dataset.jsonl`
  - 输出 `outputs/benchmark_dataset.json`
  - 输出 `outputs/benchmark_summary.csv`

## 配置说明

核心工程配置位于 [config/default_config.json](/home/paper/Benchmark/config/default_config.json)。

- `llm.temperature` / `llm.max_tokens`
  - 控制 DeepSeek 调用参数
- `qa_generation.max_per_section`
  - 控制每个 section 最多生成多少条问题
- `qa_generation.type_quota`
  - 控制四类题型配额
- `qa_generation.allow_cross_section`
  - 是否允许跨 section 生成
- `qa_generation.allow_cross_document`
  - 是否允许跨文档生成
- `qa_generation.require_strong_constraints`
  - 是否尽量要求实体、关系、条件联合出现
- `validation.support_score_threshold`
  - 支撑度阈值
- `validation.completeness_score_threshold`
  - 完整性阈值

## 如何调整 Prompt

所有 Prompt 都在 `benchmark_builder/prompts/` 下：

- [knowledge_extraction_prompt.py](/home/paper/Benchmark/benchmark_builder/prompts/knowledge_extraction_prompt.py)
- [qa_generation_prompt.py](/home/paper/Benchmark/benchmark_builder/prompts/qa_generation_prompt.py)
- [qa_validation_prompt.py](/home/paper/Benchmark/benchmark_builder/prompts/qa_validation_prompt.py)

可以直接修改角色设定、输出 schema 和约束描述。建议优先保持 JSON schema 稳定，再逐步加强题型规则和拒绝条件。

## 如何调整 question_type 配额

在 [config/default_config.json](/home/paper/Benchmark/config/default_config.json) 中修改：

```json
"type_quota": {
  "fact": 1,
  "relation": 1,
  "multi_evidence": 1,
  "explanation": 1
}
```

## 如何切换模型

直接修改 `.env` 或环境变量：

```bash
export DEEPSEEK_MODEL=deepseek-chat
export DEEPSEEK_BASE_URL=https://api.deepseek.com
```

如果代理层已经兼容 OpenAI 风格 `/chat/completions`，本项目可以直接复用。

## 常见问题

1. `extract-knowledge` 没有输出内容

通常是因为未配置 `DEEPSEEK_API_KEY`，或者开启了 `--dry-run`。

2. JSON 解析失败怎么办

原始模型响应会保存在 `knowledge_extraction.jsonl` 或 `qa_validation.jsonl` 的 `raw_response_text` 字段，同时日志会写入 `logs/benchmark_builder.log`，便于排查。

3. 如何减少人工参与

优先调高 Prompt 约束和验证阈值，再通过 `max_per_section`、题型配额和相似度阈值控制数据质量。

4. 是否支持断点续跑

支持。默认启用 `--resume`，已有中间文件会被读取并跳过已完成样本。

5. conda 环境已经创建好了，怎么进入

```bash
source /home/huang/miniconda3/etc/profile.d/conda.sh
conda activate paper_benchmark
```

6. 为什么直接 `pytest` 不再去跑 `external/` 测试

因为主项目已经通过 [pytest.ini](/home/paper/Benchmark/pytest.ini) 将 `external/` 目录排除。`external/` 仅作为方法参考来源，不是主工程测试对象。

## 质量控制规则

项目内置以下基础过滤规则：

- 删除空 `question` / `answer` 样本
- 删除 `evidence` 为空的样本
- 删除 quote 与答案明显无关的样本
- 去除重复问题和高度相似问题
- 限制每个 section 的样本数量
- `multi_evidence` 必须至少两个证据
- `relation` 类型必须包含 `relations`
- `explanation` 类型答案长度不能过短

## 说明

当前实现优先保证工程化、可批处理和可审计。Markdown 切分、近似去重和证据相关性判断使用的是可解释的基础规则；后续如果你需要，我可以继续把表格解析、多 section 组合采样和更强的相似度去重扩展进去。
