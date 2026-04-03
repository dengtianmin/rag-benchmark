# 实验运行手册

## 1. 文档目标

本文档用于说明当前仓库中实验的标准运行方式，覆盖：

- 数据集构建流水线如何运行
- 检索索引如何准备
- 各 baseline / ours / ablation 如何运行
- 命令行参数与环境变量的含义
- 结果产物如何解读
- 复现实验时推荐遵循的最佳实践

本文档面向“从原始 Markdown 文档开始，到最终实验指标落盘”为止的完整链路。

## 2. 实验全景

当前仓库的实验分为两层：

1. 数据集构建层
   - 入口：[`main.py`](/home/paper/Benchmark/main.py)
   - CLI：[`benchmark_builder/cli.py`](/home/paper/Benchmark/benchmark_builder/cli.py)
   - 目标：从 `data/` 下 Markdown 文档生成 `benchmark_dataset.jsonl`
2. 统一实验宿主层
   - 入口：[`scripts/`](/home/paper/Benchmark/scripts)
   - 目标：在统一 schema、统一 evaluator、统一输入数据上运行不同方法并输出指标

推荐把整个实验生命周期理解为下面 4 个阶段：

1. 准备环境与密钥
2. 构建 benchmark 数据集
3. 构建向量索引与设置运行时检索参数
4. 运行 baseline / ours / ablation，并归档结果

## 3. 目录与关键产物

### 3.1 输入目录

- `data/`
  - 原始 Markdown 文档

### 3.2 中间产物目录

- `artifacts/markdown_sections.jsonl`
  - Markdown 切分结果
- `artifacts/knowledge_extraction.jsonl`
  - 每个 section 的结构化知识抽取结果
- `artifacts/qa_candidates.jsonl`
  - 候选 QA
- `artifacts/qa_generation_progress.jsonl`
  - QA 生成断点续跑进度
- `artifacts/qa_validation.jsonl`
  - QA 校验结果

### 3.3 最终数据集目录

- `outputs/benchmark_dataset.jsonl`
  - 主实验输入数据集
- `outputs/benchmark_dataset.json`
  - JSON 版本
- `outputs/benchmark_summary.csv`
  - 汇总统计

### 3.4 实验结果目录

- `outputs/experiments/<method>/predictions.jsonl`
- `outputs/experiments/<method>/metrics.json`
- `outputs/experiments/all_baselines/summary.json`
- `outputs/experiments/ablation/comparison.json`

## 4. 环境准备

### 4.1 Python 环境

推荐环境：

- Python `3.11`
- Conda 环境名：`paper_benchmark`

安装方式：

```bash
source /home/huang/miniconda3/etc/profile.d/conda.sh
conda create -y -n paper_benchmark python=3.11
conda activate paper_benchmark
pip install -r requirements.txt
```

### 4.2 基础环境变量

数据集构建阶段至少需要配置一组 LLM 参数。当前仓库默认兼容 DeepSeek：

```dotenv
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

构建器内部也支持更通用的 LLM 变量：

```dotenv
LLM_PROVIDER=deepseek
LLM_API_STYLE=openai
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_DISABLE_AUTH=false
```

如果只配置 `DEEPSEEK_*`，当前工程也能正常运行，因为 `LLM_*` 会回退到 `DEEPSEEK_*`。

### 4.3 向量检索相关环境变量

如果实验使用 `dense` 或 `hybrid` 检索，需要配置 embedding 与 Qdrant：

```dotenv
ZHIPUAI_API_KEY=your_zhipu_key
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32

VECTOR_DB_BACKEND=qdrant
RETRIEVAL_MODE=dense
RETRIEVAL_TOP_K=5
RETRIEVAL_CANDIDATE_MULTIPLIER=3

QDRANT_USE_LOCAL=false
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=benchmark_sections
QDRANT_PATH=artifacts/qdrant
```

### 4.4 Reranker 与 Generator 环境变量

当前实验宿主会从运行时配置读取 reranker 与 generator 设置：

```dotenv
USE_RERANK=true
RERANK_ENABLED=true
RERANK_BACKEND=tei
TEI_RERANK_URL=http://127.0.0.1:8080
TEI_RERANK_TIMEOUT=30
TEI_RERANK_API_KEY=
TEI_RERANK_MAX_RETRIES=1
RERANK_TOP_N=5
RERANK_ALLOW_MOCK_FALLBACK=false

GENERATOR_BACKEND=mock
GENERATOR_API_KEY=
GENERATOR_BASE_URL=https://api.openai.com/v1
GENERATOR_MODEL=
GENERATOR_TIMEOUT=60
GENERATOR_MAX_TOKENS=512
GENERATOR_TEMPERATURE=0
GENERATOR_JSON_MODE=true
```

说明：

- `GENERATOR_BACKEND=mock` 是当前默认值，便于先验证实验链路
- 如果要做真实生成式回答评测，应切换到 `GENERATOR_BACKEND=llm`
- `USE_RERANK=false` 可全局关闭精排
- `RERANK_ENABLED` 和 `USE_RERANK` 同时生效，建议两者保持一致

## 5. 配置优先级

当前仓库存在两套配置入口：

1. `config/default_config.json`
2. 环境变量 `.env`

实际生效顺序建议按下面理解：

1. 代码默认值
2. `config/default_config.json`
3. `.env` / shell 环境变量
4. 命令行参数

也就是说：

- 路径、并发、是否 dry-run 等，优先由命令行控制
- 检索、rerank、generator、embedding 等运行时行为，优先由环境变量控制

## 6. 数据集构建流程

### 6.1 全流程命令

```bash
conda activate paper_benchmark
python main.py run-all \
  --input-dir data \
  --artifacts-dir artifacts/full_run \
  --output-dir outputs/full_run \
  --concurrency 4 \
  --resume
```

这条命令会顺序执行：

1. `parse-md`
2. `extract-knowledge`
3. `generate-qa`
4. `validate-qa`
5. `build-dataset`

### 6.2 分阶段命令

```bash
python main.py parse-md --input-dir data --artifacts-dir artifacts/full_run
python main.py extract-knowledge --artifacts-dir artifacts/full_run --concurrency 4 --resume
python main.py generate-qa --artifacts-dir artifacts/full_run --concurrency 4 --resume
python main.py validate-qa --artifacts-dir artifacts/full_run --concurrency 4 --resume
python main.py build-dataset --artifacts-dir artifacts/full_run --output-dir outputs/full_run
```

建议全量实验采用“分阶段执行 + 每阶段检查产物”的方式，而不是只用一条 `run-all` 盲跑到底。

### 6.3 构建器通用命令行参数

所有 `main.py` 子命令共享以下参数：

| 参数 | 含义 | 默认值 | 建议 |
|---|---|---:|---|
| `--config` | 配置文件路径 | `config/default_config.json` | 固定使用默认配置，必要时另建实验专用配置 |
| `--input-dir` | 原始 Markdown 输入目录 | `data` | 每次实验固定输入快照，不要边跑边改 |
| `--artifacts-dir` | 中间产物目录 | `artifacts` | 每次正式实验使用独立目录 |
| `--output-dir` | 最终输出目录 | `outputs` | 每次正式实验使用独立目录 |
| `--max-files` | 最多处理多少个 Markdown 文件 | `null` | 调试时用 `2/5/20`，正式实验留空 |
| `--concurrency` | 并发线程数 | `4` | 从 `2` 或 `4` 起步，观察 API 稳定性 |
| `--resume/--no-resume` | 是否断点续跑 | `--resume` | 正式跑批必须开启 |
| `--dry-run` | 只走流程不调用真实模型 | `false` | 用于烟雾测试，不用于正式实验 |

### 6.4 构建器配置项说明

下面这些字段来自 [`config/default_config.json`](/home/paper/Benchmark/config/default_config.json)。

#### 顶层运行控制

| 配置项 | 含义 | 当前默认值 |
|---|---|---:|
| `log_level` | 日志级别 | `INFO` |
| `request_timeout` | 单次 LLM 请求超时秒数 | `120` |
| `retry_attempts` | 失败重试次数 | `3` |
| `retry_backoff_min` | 最小退避秒数 | `1` |
| `retry_backoff_max` | 最大退避秒数 | `20` |
| `resume` | 是否默认断点续跑 | `true` |
| `dry_run` | 是否默认 dry run | `false` |

#### `llm`

| 配置项 | 含义 | 当前默认值 |
|---|---|---:|
| `temperature` | 构建阶段模型采样温度 | `0.1` |
| `max_tokens` | 单次输出 token 上限 | `2048` |
| `json_mode` | 是否要求结构化 JSON 输出 | `true` |

#### `qa_generation`

| 配置项 | 含义 | 当前默认值 |
|---|---|---:|
| `max_per_section` | 每个 section 最多保留多少条 QA | `4` |
| `type_quota.fact` | `fact` 题目标配额 | `1` |
| `type_quota.relation` | `relation` 题目标配额 | `1` |
| `type_quota.multi_evidence` | `multi_evidence` 题目标配额 | `1` |
| `type_quota.explanation` | `explanation` 题目标配额 | `1` |
| `allow_cross_section` | 是否允许跨 section 组题 | `false` |
| `allow_cross_document` | 是否允许跨文档组题 | `false` |
| `require_strong_constraints` | 是否要求题目具备强约束 | `true` |
| `require_evidence` | 是否强制 evidence | `true` |
| `explanation_min_length` | 解释题最短长答案长度 | `40` |
| `similarity_threshold` | 近似去重阈值 | `0.9` |

#### `validation`

| 配置项 | 含义 | 当前默认值 |
|---|---|---:|
| `support_score_threshold` | 证据支撑阈值 | `0.8` |
| `completeness_score_threshold` | 答案完整性阈值 | `0.75` |

### 6.5 各阶段产物说明

| 阶段 | 输出文件 | 作用 |
|---|---|---|
| `parse-md` | `markdown_sections.jsonl` | 后续所有实验的文本索引输入 |
| `extract-knowledge` | `knowledge_extraction.jsonl` | 图检索、KBQA、ours 的结构化输入 |
| `generate-qa` | `qa_candidates.jsonl` | 待校验的候选题目 |
| `generate-qa` | `qa_generation_progress.jsonl` | 记录 section 级处理进度，用于 resume |
| `validate-qa` | `qa_validation.jsonl` | 记录是否通过校验与评分 |
| `build-dataset` | `benchmark_dataset.jsonl/.json/.csv` | 最终实验输入与摘要统计 |

## 7. 向量索引准备

### 7.1 何时需要建索引

只有在 `RETRIEVAL_MODE=dense` 或 `RETRIEVAL_MODE=hybrid` 时，才需要提前构建 Qdrant 向量索引。

如果使用 `RETRIEVAL_MODE=lexical`，可以不构建 Qdrant 索引。

补充说明：

- 当前 `RETRIEVAL_MODE=lexical` 使用的是项目内置 BM25 稀疏检索
- 因此 `lexical` 已经不再是早期的简单 token overlap baseline

### 7.2 启动 Qdrant

```bash
docker run -d \
  --name qdrant-benchmark \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

### 7.3 构建索引

```bash
export QDRANT_USE_LOCAL=false
export QDRANT_URL=http://127.0.0.1:6333

python scripts/build_qdrant_index.py \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --manifest-path artifacts/full_run/qdrant_index_manifest.json \
  --recreate
```

### 7.4 `build_qdrant_index.py` 参数说明

| 参数 | 含义 | 默认值 | 建议 |
|---|---|---:|---|
| `--sections` | 输入 section 文件 | `artifacts/two_file_demo/markdown_sections.jsonl` | 指向本次实验对应的 artifacts |
| `--config` | 运行时配置文件 | `config/default_config.json` | 大多数情况下无需改 |
| `--manifest-path` | 索引构建清单输出路径 | `artifacts/qdrant_index_manifest.json` | 每个实验单独保存 |
| `--limit` | 只索引前 N 个 section | `null` | 调试时使用 |
| `--recreate` | 先删 collection 再重建 | `false` | 正式全量重跑时使用 |
| `--dry-run` | 只读输入并输出摘要 | `false` | 校验输入时使用 |

### 7.5 相关环境变量说明

| 环境变量 | 含义 |
|---|---|
| `EMBEDDING_MODEL` | 生成向量所用 embedding 模型名称 |
| `EMBEDDING_DIM` | 向量维度，必须与 collection 一致 |
| `EMBEDDING_BATCH_SIZE` | embedding 批大小 |
| `QDRANT_USE_LOCAL` | `true` 表示本地目录模式，`false` 表示远程服务 |
| `QDRANT_URL` | 远程 Qdrant 地址 |
| `QDRANT_PATH` | 本地模式下的持久化目录 |
| `QDRANT_COLLECTION` | collection 名称 |

## 8. Baseline 与 Ours 运行

### 8.1 通用原则

所有实验脚本默认输入都是：

- `--dataset`: `benchmark_dataset.jsonl`
- `--sections`: `markdown_sections.jsonl`
- `--knowledge`: `knowledge_extraction.jsonl`，仅图方法需要

所有脚本的共同输出形式都是：

- `predictions.jsonl`
- `metrics.json`

建议每组实验都显式指定 `--output-dir`，不要依赖默认目录覆盖旧结果。

### 8.2 Traditional RAG

```bash
python scripts/run_traditional_rag.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/traditional_rag_dense
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--dataset` | benchmark 数据集 |
| `--sections` | section 索引输入 |
| `--top-k` | 初始检索保留文档数，也是 `hit@k` 的 `k` |
| `--limit` | 仅评测前 N 个样本 |
| `--disable-rerank` | 禁用 reranker |
| `--max-workers` | 样本级并发数 |
| `--output-dir` | 输出目录 |

说明：

- `Traditional RAG` 在 `RETRIEVAL_MODE=lexical` 下，当前使用 BM25 作为一阶段 sparse retriever
- 同一套 BM25 底座也会被 `Rewrite-RAG`、`Graph-enhanced RAG` 的 seed retrieval、`Ours-Ch4` 的文本召回复用

### 8.3 Rewrite-RAG

```bash
python scripts/run_rewrite_rag.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --mode entity_relation \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/rewrite_rag_dense
```

额外参数：

| 参数 | 含义 |
|---|---|
| `--mode naive` | 使用简单 rewrite |
| `--mode entity_relation` | 使用实体-关系驱动 rewrite，当前更推荐 |

### 8.4 Graph-enhanced RAG

```bash
python scripts/run_graph_enhanced_rag.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --seed-top-k 5 \
  --expand-k 5 \
  --top-k 5 \
  --hint-mode gold \
  --max-workers 4 \
  --output-dir outputs/experiments/graph_enhanced_rag_dense
```

额外参数：

| 参数 | 含义 |
|---|---|
| `--seed-top-k` | 初始种子检索数量 |
| `--expand-k` | 图扩展时追加候选数量 |
| `--top-k` | 最终保留给生成器或评估器的文档数 |
| `--hint-mode gold` | 使用 gold hint，偏上界设置 |
| `--hint-mode none` | 不使用 hint，更接近真实部署设置 |

### 8.5 KBQA baseline

```bash
python scripts/run_kbqa_baseline.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --entity-mode gold \
  --relation-mode gold \
  --output-dir outputs/experiments/kbqa_baseline
```

额外参数：

| 参数 | 含义 |
|---|---|
| `--entity-mode gold` | 使用 gold 实体，偏上界 |
| `--entity-mode heuristic` | 使用启发式实体链接 |
| `--relation-mode gold` | 使用 gold 关系，偏上界 |
| `--relation-mode heuristic` | 使用启发式关系匹配 |

### 8.6 Ours-Ch4

```bash
python scripts/run_ours_ch4.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --skeleton-mode oracle \
  --max-workers 4 \
  --output-dir outputs/experiments/ours_ch4_dense
```

额外参数：

| 参数 | 含义 |
|---|---|
| `--skeleton-mode oracle` | 使用 oracle skeleton，偏上界 |
| `--skeleton-mode stub_predicted` | 使用占位预测 skeleton，更接近真实链路 |

### 8.7 Ablation

```bash
python scripts/run_ablation.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --skeleton-mode oracle \
  --max-workers 4 \
  --output-dir outputs/experiments/ablation_dense
```

脚本会自动运行以下 4 个变体：

- `full`
- `w/o_relation_driven`
- `w/o_skeleton_rewrite`
- `w/o_text_compensation`

主要输出：

- `outputs/experiments/ablation_dense/comparison.json`
- 每个变体各自的 `metrics.json` 与 `predictions.jsonl`

### 8.8 一键跑全部 baseline

```bash
python scripts/run_all_baselines.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --seed-top-k 5 \
  --expand-k 5 \
  --rewrite-mode entity_relation \
  --graph-hint-mode gold \
  --kbqa-entity-mode gold \
  --kbqa-relation-mode gold \
  --skeleton-mode oracle \
  --max-workers 4 \
  --output-dir outputs/experiments/all_baselines_full \
  --include-ablation
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--rewrite-mode` | Rewrite-RAG 的 rewrite 模式 |
| `--graph-hint-mode` | Graph-enhanced RAG 的 hint 模式 |
| `--kbqa-entity-mode` | KBQA 的实体模式 |
| `--kbqa-relation-mode` | KBQA 的关系模式 |
| `--skeleton-mode` | Ours-Ch4 的 skeleton 模式 |
| `--include-ablation` | 是否顺带执行 ours 的 ablation |

输出核心文件：

- `summary.json`
- `traditional_rag/metrics.json`
- `rewrite_rag/metrics.json`
- `graph_enhanced_rag/metrics.json`
- `kbqa_baseline/metrics.json`
- `ours_ch4/metrics.json`

## 9. 运行时环境变量说明

### 9.1 检索相关

| 环境变量 | 作用 | 常见取值 |
|---|---|---|
| `RETRIEVAL_MODE` | 检索模式 | `lexical` / `dense` / `hybrid` |
| `RETRIEVAL_TOP_K` | 默认 top-k | `5` |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | 候选扩展倍数 | `3` |
| `VECTOR_DB_BACKEND` | 向量库后端 | `qdrant` |

### 9.2 Rerank 相关

| 环境变量 | 作用 | 常见取值 |
|---|---|---|
| `USE_RERANK` | 是否启用精排 | `true` / `false` |
| `RERANK_ENABLED` | rerank 逻辑总开关 | `true` / `false` |
| `RERANK_BACKEND` | 精排后端 | `tei` |
| `TEI_RERANK_URL` | TEI 服务地址 | `http://127.0.0.1:8080` |
| `TEI_RERANK_TIMEOUT` | 请求超时秒数 | `30` |
| `TEI_RERANK_MAX_RETRIES` | 重试次数 | `1` |
| `RERANK_TOP_N` | 精排后保留条数 | `5` |
| `RERANK_ALLOW_MOCK_FALLBACK` | 失败时是否退回 mock | `false` |

### 9.3 本地模型 reranker 相关

| 环境变量 | 作用 | 常见取值 |
|---|---|---|
| `RERANKER_MODEL_PATH` | 本地模型目录 | `/home/paper/Benchmark/models/bge-reranker-v2-m3` |
| `RERANKER_DEVICE` | 运行设备 | `cpu` / `cuda:0` |
| `RERANKER_USE_FP16` | 是否半精度 | `false` / `true` |
| `RERANKER_QUERY_MAX_LENGTH` | query 最大长度 | `256` |
| `RERANKER_PASSAGE_MAX_LENGTH` | passage 最大长度 | `512` |

### 9.4 Generator 相关

| 环境变量 | 作用 | 常见取值 |
|---|---|---|
| `GENERATOR_BACKEND` | 回答生成后端 | `mock` / `llm` |
| `GENERATOR_API_KEY` | 真实 LLM API key | 空或真实值 |
| `GENERATOR_BASE_URL` | LLM 服务地址 | OpenAI 兼容地址 |
| `GENERATOR_MODEL` | 模型名 | 例如 `gpt-4o-mini` |
| `GENERATOR_TIMEOUT` | 请求超时秒数 | `60` |
| `GENERATOR_MAX_TOKENS` | 生成长度上限 | `512` |
| `GENERATOR_TEMPERATURE` | 采样温度 | `0` |
| `GENERATOR_JSON_MODE` | 是否强制 JSON 响应 | `true` |

## 10. 推荐实验配方

### 10.1 烟雾测试

目标：验证脚本、路径和依赖是否正确。

建议：

- `--limit 3` 或 `--max-files 2`
- `RETRIEVAL_MODE=lexical`
- `USE_RERANK=false`
- `GENERATOR_BACKEND=mock`
- 全部输出写到临时目录，例如 `outputs/smoke/*`

### 10.2 标准对比实验

目标：比较不同 pipeline 的统一效果。

建议：

- 固定同一份 `benchmark_dataset.jsonl`
- 固定同一个 `markdown_sections.jsonl`
- 固定同一组环境变量
- 所有方法使用同一个 `top-k`
- 使用 `run_all_baselines.py` 统一跑

### 10.3 Dense 检索实验

目标：评估向量检索或 hybrid 检索的收益。

建议：

- 先固定 `EMBEDDING_MODEL` 与 `EMBEDDING_DIM`
- 重新构建一次 Qdrant index
- 分别运行 `lexical`、`dense`、`hybrid`
- 每次只改变一个变量

### 10.4 Ablation 实验

目标：衡量 ours 各子模块的边际贡献。

建议：

- 保持 `top-k`、generator、rerank、dataset 不变
- 只用 `run_ablation.py`
- 直接对 `comparison.json` 做表格汇总

## 11. 结果文件解读

### 11.1 `predictions.jsonl`

单条样本级输出，适合做误差分析。通常用于：

- 检查命中证据是否正确
- 检查模型回答是否偏离 gold answer
- 检查某类题型是否系统性失败

### 11.2 `metrics.json`

方法级汇总指标，通常包含：

- `answer`
  - 回答类指标，例如 `em`、`token_f1`
- `retrieval`
  - 检索类指标，例如 `hit@k`、`mrr`
- 方法特定统计
  - 如 rewrite 效果、graph 扩展统计、text compensation 激活率

### 11.3 `summary.json`

只在 `run_all_baselines.py` 下生成，用于统一汇总多方法结果。适合：

- 直接生成论文表格
- 对比不同方法是否共享相同输入与参数
- 检查是否有某个方法漏跑

## 12. 业界最佳实践建议

### 12.1 目录隔离

每次正式实验都单独建目录，例如：

```text
artifacts/exp_20260328_dense/
outputs/exp_20260328_dense/
outputs/experiments/exp_20260328_dense/
```

不要把不同实验轮次混在 `artifacts/full_run` 里反复覆盖。

### 12.2 固定输入快照

正式实验开始后，不要修改：

- `data/`
- `markdown_sections.jsonl`
- `knowledge_extraction.jsonl`
- `benchmark_dataset.jsonl`

否则不同方法之间不再可比。

### 12.3 变量控制

一次实验只改一个因素，例如：

- 只改 `RETRIEVAL_MODE`
- 只改 `top-k`
- 只改 `rewrite-mode`
- 只改 `skeleton-mode`

如果同时改多个参数，后续很难解释因果。

### 12.4 小样本先行

正式跑全量前，先做：

```bash
python scripts/run_all_baselines.py --limit 5 --max-workers 2 ...
```

确认：

- 路径无误
- 指标文件能正常落盘
- generator / reranker / qdrant 可访问

再扩大到全量。

### 12.5 保留运行元数据

建议每次实验额外保存：

- git commit id
- `.env` 脱敏快照
- 运行命令全文
- 开始时间与结束时间
- 实验目标说明

当前项目里 `metrics.json` 会保留部分运行参数，但仍建议额外写一份实验日志。

### 12.6 并发保守上调

`max_workers` 和 `concurrency` 不建议一开始就拉满。更稳妥的策略是：

1. 先用 `1`
2. 再试 `2`
3. 再试 `4`
4. 稳定后再试更高值

因为实验链路里同时涉及：

- 外部 API
- 本地 I/O
- 可能的 reranker / 向量库服务

### 12.7 上界与真实设置分开报

以下配置偏上界，不应与真实部署设置混报：

- `hint-mode=gold`
- `entity-mode=gold`
- `relation-mode=gold`
- `skeleton-mode=oracle`

论文或报告中建议单独标注为 `oracle` / `gold-assisted`。

## 13. 推荐命令模板

### 13.1 构建全量数据集

```bash
python main.py run-all \
  --input-dir data \
  --artifacts-dir artifacts/full_run \
  --output-dir outputs/full_run \
  --concurrency 4 \
  --resume
```

### 13.2 构建 dense 检索索引

```bash
export RETRIEVAL_MODE=dense
export QDRANT_USE_LOCAL=false
export QDRANT_URL=http://127.0.0.1:6333

python scripts/build_qdrant_index.py \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --manifest-path artifacts/full_run/qdrant_index_manifest.json \
  --recreate
```

### 13.3 跑全套 baseline

```bash
python scripts/run_all_baselines.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --seed-top-k 5 \
  --expand-k 5 \
  --rewrite-mode entity_relation \
  --graph-hint-mode gold \
  --kbqa-entity-mode gold \
  --kbqa-relation-mode gold \
  --skeleton-mode oracle \
  --max-workers 4 \
  --output-dir outputs/experiments/all_baselines_full \
  --include-ablation
```

## 14. 常见问题

### 14.1 什么时候必须提供 `--knowledge`

以下方法必须提供：

- `run_graph_enhanced_rag.py`
- `run_kbqa_baseline.py`
- `run_ours_ch4.py`
- `run_ablation.py`
- `run_all_baselines.py`

因为它们依赖 `knowledge_extraction.jsonl` 中的结构化图信息。

### 14.2 为什么 `dense` 模式跑不起来

通常有 4 类原因：

1. 没有配置 `ZHIPUAI_API_KEY`
2. 没有先构建 Qdrant 索引
3. `EMBEDDING_DIM` 与已有 collection 不一致
4. `QDRANT_URL` / `QDRANT_COLLECTION` 指向错误

### 14.3 为什么正式实验不建议用默认目录

因为默认目录更适合 demo 或开发调试。正式实验如果直接覆盖：

- 难以回溯
- 难以比较不同轮次
- 容易把旧结果和新结果混在一起

## 15. 相关文档

- [`build_pipeline.md`](/home/paper/Benchmark/docs/build_pipeline.md)
- [`vector_retrieval_usage.md`](/home/paper/Benchmark/docs/vector_retrieval_usage.md)
- [`unified_experiment_host.md`](/home/paper/Benchmark/docs/unified_experiment_host.md)
- [`current_run_report.md`](/home/paper/Benchmark/docs/current_run_report.md)
