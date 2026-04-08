# 向量检索接入使用说明

## 1. 环境变量

最少需要配置以下环境变量：

```dotenv
ZHIPUAI_API_KEY=your_zhipu_key
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=32

VECTOR_DB_BACKEND=qdrant
RETRIEVAL_MODE=dense
QDRANT_USE_LOCAL=false
QDRANT_PATH=artifacts/qdrant
QDRANT_COLLECTION=benchmark_sections
QDRANT_URL=http://127.0.0.1:6333

USE_RERANK=true
RERANKER_MODEL_PATH=/path/to/BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu
RERANKER_USE_FP16=false
RERANKER_QUERY_MAX_LENGTH=256
RERANKER_PASSAGE_MAX_LENGTH=512
```

说明：

- `RETRIEVAL_MODE` 支持 `lexical` / `dense` / `hybrid`
- 当前 `lexical` 模式使用 BM25 稀疏检索，不再是旧的 overlap heuristic
- `QDRANT_USE_LOCAL=true` 时使用本地持久化模式
- `QDRANT_USE_LOCAL=false` 时使用远程 `QDRANT_URL`
- `USE_RERANK=false` 可以关闭本地 BGE 精排
- 远端 Docker Qdrant 场景下，首次使用前需要先执行一次索引构建，把 section 向量写入远端 collection

## 2. 索引构建

如果你使用远端 Docker Qdrant，先启动服务：

```bash
docker run -d \
  --name qdrant-benchmark \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

然后构建 Qdrant 索引：

```bash
source /home/huang/miniconda3/etc/profile.d/conda.sh
conda activate paper_benchmark

export QDRANT_USE_LOCAL=false
export QDRANT_URL=http://127.0.0.1:6333

python scripts/build_qdrant_index.py \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --manifest-path artifacts/qdrant_index_manifest.remote.json \
  --recreate
```

如果你仍然想使用本地目录模式，再切回：

```bash
export QDRANT_USE_LOCAL=true
export QDRANT_PATH=artifacts/qdrant
```

只做读取与统计、不真正写入时：

```bash
python scripts/build_qdrant_index.py \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --dry-run \
  --limit 20
```

## 3. 模型放置路径

本地 BGE reranker 需要提前准备模型目录，例如：

```text
/models/BAAI/bge-reranker-v2-m3
```

然后设置：

```dotenv
RERANKER_MODEL_PATH=/models/BAAI/bge-reranker-v2-m3
```

如果机器有 GPU，可设置：

```dotenv
RERANKER_DEVICE=cuda:0
RERANKER_USE_FP16=true
```

CPU 模式推荐：

```dotenv
RERANKER_DEVICE=cpu
RERANKER_USE_FP16=false
```

## 4. 运行方式

Traditional RAG：

```bash
python scripts/run_traditional_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/traditional_rag_dense
```

Rewrite-RAG：

```bash
python scripts/run_rewrite_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --mode entity_relation \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/rewrite_rag_dense
```

Graph-enhanced RAG：

```bash
python scripts/run_graph_enhanced_rag.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --seed-top-k 5 \
  --expand-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/graph_enhanced_rag_dense
```

Ours-Ch4：

```bash
python scripts/run_ours_ch4.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --skeleton-mode oracle \
  --max-workers 4 \
  --output-dir outputs/experiments/ours_ch4_dense
```

Ablation：

```bash
python scripts/run_ablation.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/ablation_dense
```

统一跑全部 baseline：

```bash
python scripts/run_all_baselines.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --sections artifacts/two_file_demo/markdown_sections.jsonl \
  --knowledge artifacts/two_file_demo/knowledge_extraction.jsonl \
  --limit 5 \
  --top-k 5 \
  --max-workers 4 \
  --output-dir outputs/experiments/all_baselines_concurrent
```

当前该脚本只包含 3 个 baseline：`Traditional RAG`、`Graph-enhanced RAG`、`KBQA baseline`。

并发实现说明：

- 检索器实例在脚本级共享，避免重复初始化 Qdrant client
- generator / reranker / pipeline 在线程内独立创建，避免共享有状态 `requests.Session`
- `max_workers` 默认是 `1`，逐步提升到 `4 / 8 / 16` 更稳妥

如果 generator 走 vLLM，8K 上下文很容易在 `top-k=5` 时被长 section 顶满。16K 是更稳妥的起点，但仍建议配合控制 `GENERATOR_MAX_TOKENS` 或裁剪上下文长度。

## 5. 推荐验证组合

### lexical vs dense

先设置：

```dotenv
RETRIEVAL_MODE=lexical
USE_RERANK=false
```

运行：

```bash
python scripts/run_traditional_rag.py --output-dir outputs/experiments/traditional_rag_lexical
```

这里的 `traditional_rag_lexical` 当前对应的是 BM25 baseline。

再设置：

```dotenv
RETRIEVAL_MODE=dense
USE_RERANK=false
```

运行：

```bash
python scripts/run_traditional_rag.py --output-dir outputs/experiments/traditional_rag_dense
```

### dense vs dense + rerank

先设置：

```dotenv
RETRIEVAL_MODE=dense
USE_RERANK=false
```

运行：

```bash
python scripts/run_rewrite_rag.py --output-dir outputs/experiments/rewrite_rag_dense_only
```

再设置：

```dotenv
RETRIEVAL_MODE=dense
USE_RERANK=true
```

运行：

```bash
python scripts/run_rewrite_rag.py --output-dir outputs/experiments/rewrite_rag_dense_rerank
```

### Ours-Ch4 full vs w/o text compensation

设置：

```dotenv
RETRIEVAL_MODE=dense
USE_RERANK=true
```

运行：

```bash
python scripts/run_ablation.py --output-dir outputs/experiments/ablation_dense
```

查看：

- `full`
- `w_o_text_compensation`

### Ours-Ch4 full vs w/o relation-driven scoring

同样查看：

- `full`
- `w_o_relation_driven`

## 6. trace 字段说明

当前主线 pipeline 的 trace 中会补充以下字段：

- `embedding_model`
- `embedding_dim`
- `qdrant_collection`
- `retrieval_mode`
- `use_rerank`
- `initial_dense_candidates`
- `final_reranked_candidates`

不同 pipeline 还会补充各自细节，例如：

- Traditional：
  - `dense_recall_scores`
  - `rerank_scores`

- Rewrite：
  - `original_dense_recall_scores`
  - `rewritten_dense_recall_scores`
  - `original_rerank_scores`
  - `rewritten_rerank_scores`

- Graph-enhanced：
  - `seed_dense_recall_scores`
  - `organization_details`

- Ours-Ch4：
  - `relation_driven_details`
  - `text_compensation_details`
  - `rerank_scores`

## 7. 最小回归检查清单

建议按以下顺序执行：

1. `build_qdrant_index.py` 能构建索引
2. `Traditional RAG` 能运行
3. `Rewrite-RAG` 能运行
4. `Graph-enhanced RAG` 能运行
5. `Ours-Ch4` 能运行
6. `Ablation` 能运行

推荐命令：

```bash
python scripts/build_qdrant_index.py --sections artifacts/two_file_demo/markdown_sections.jsonl --recreate
python scripts/run_traditional_rag.py --limit 5
python scripts/run_rewrite_rag.py --limit 5
python scripts/run_graph_enhanced_rag.py --limit 5
python scripts/run_ours_ch4.py --limit 5
python scripts/run_ablation.py --limit 5
```

## 8. 常见问题

### Qdrant 本地模式打不开

症状：

- 报 storage folder already accessed

原因：

- 同一个本地 Qdrant path 被多个进程同时占用

处理：

- 关闭其他进程
- 或改用远程 Qdrant server

### 智谱 embedding 调用失败

优先检查：

- `ZHIPUAI_API_KEY` 是否配置
- 网络是否可访问智谱 API
- `EMBEDDING_DIM` 是否与索引维度一致

### reranker 加载失败

优先检查：

- `RERANKER_MODEL_PATH` 是否指向本地实际目录
- 目录里是否有完整模型文件
- `FlagEmbedding`、`torch` 是否安装

### 检索无结果或效果异常

优先检查：

- 是否先构建了索引
- `QDRANT_COLLECTION` 是否与构建时一致
- `EMBEDDING_MODEL` 与已建索引是否一致
- `RETRIEVAL_MODE` 是否仍为 `lexical`
