# 向量检索接入变更记录

## 1. 文档目的

本文档记录当前仓库已完成的向量检索接入工作，便于后续继续开发、复现实验和环境迁移。

当前状态以 `src/` 主线实验代码为准，不涉及 `external/` 目录。

## 2. 当前使用环境

推荐 conda 环境：

```bash
/home/huang/miniconda3/bin/conda create -y -n paper_benchmark python=3.11
source /home/huang/miniconda3/etc/profile.d/conda.sh
conda activate paper_benchmark
pip install -r requirements.txt
```

当前项目开发与测试使用环境名：

- `paper_benchmark`

建议在该环境中补齐以下关键依赖：

- `qdrant-client`
- `zhipuai`
- `FlagEmbedding`
- `transformers`
- `torch`
- `numpy`
- `python-dotenv`
- `tenacity`

如果环境中使用了 `SOCKS` 代理，还需要：

- `socksio`

## 3. 当前向量检索技术栈

当前接入方案如下：

- 向量数据库：本地或远程 `Qdrant`
- embedding：智谱 `Embedding-3`
- reranker：本地 `BGE reranker`
- chat 模型：兼容 OpenAI API 的本地 `vLLM`，通常部署 `Llama`

当前配置读取优先级：

1. 环境变量
2. `config/default_config.json`

统一配置模块：

- [runtime_config.py](/home/paper/Benchmark/src/runtime_config.py)

## 4. 本轮已完成的代码改造

### 4.1 配置与依赖层

已完成：

- 扩展 [requirements.txt](/home/paper/Benchmark/requirements.txt)
- 扩展 [.env.example](/home/paper/Benchmark/.env.example)
- 扩展 [default_config.json](/home/paper/Benchmark/config/default_config.json)
- 新增统一配置读取 [runtime_config.py](/home/paper/Benchmark/src/runtime_config.py)

### 4.2 Embedding 层

已完成：

- 新增 [zhipu_embedder.py](/home/paper/Benchmark/src/embedders/zhipu_embedder.py)

能力：

- 批量 embedding
- 自动分批
- 空文本过滤
- 重试和异常处理
- 维度与返回条数校验

### 4.3 向量存储层

已完成：

- 新增 [qdrant_store.py](/home/paper/Benchmark/src/retrievers/qdrant_store.py)

能力：

- 本地 path 模式
- 远程 URL 模式
- `ensure_collection`
- `recreate_collection`
- `upsert_sections`
- `search_by_vector`
- `search_by_query_vector`

### 4.4 本地 rerank 层

已完成：

- 新增 [bge_reranker.py](/home/paper/Benchmark/src/rerankers/bge_reranker.py)

能力：

- 本地模型加载
- CPU / GPU 切换
- fp16 开关
- query-doc 对打分
- rerank 重排

### 4.5 离线索引构建

已完成：

- 新增 [build_qdrant_index.py](/home/paper/Benchmark/scripts/build_qdrant_index.py)

### 4.6 统一文本检索抽象

已完成：

- 新增 [text_retriever.py](/home/paper/Benchmark/src/retrievers/text_retriever.py)

支持：

- `lexical`
- `dense`
- `hybrid`

### 4.7 Pipeline 接入范围

当前已经接入统一检索/重排链路的实验：

- [traditional_rag.py](/home/paper/Benchmark/src/pipelines/traditional_rag.py)
- [rewrite_rag.py](/home/paper/Benchmark/src/pipelines/rewrite_rag.py)
- [graph_enhanced_rag.py](/home/paper/Benchmark/src/pipelines/graph_enhanced_rag.py)
- [ours_ch4.py](/home/paper/Benchmark/src/pipelines/ours_ch4.py)

对应入口脚本：

- [run_traditional_rag.py](/home/paper/Benchmark/scripts/run_traditional_rag.py)
- [run_rewrite_rag.py](/home/paper/Benchmark/scripts/run_rewrite_rag.py)
- [run_graph_enhanced_rag.py](/home/paper/Benchmark/scripts/run_graph_enhanced_rag.py)
- [run_ours_ch4.py](/home/paper/Benchmark/scripts/run_ours_ch4.py)
- [run_ablation.py](/home/paper/Benchmark/scripts/run_ablation.py)

## 5. 当前实验链路

### Traditional RAG

- `query -> dense/lexical/hybrid recall -> optional BGE rerank -> generator`

### Rewrite-RAG

- `original query -> dense/lexical/hybrid recall -> optional rerank`
- `rewritten query -> dense/lexical/hybrid recall -> optional rerank`

### Graph-enhanced RAG

- `seed retrieval -> graph expansion -> organization -> optional BGE rerank -> generator`

### Ours-Ch4

- `skeleton rewrite -> relation-driven retrieval`
- `dense score + relation score + constraint score` 融合
- `text compensation` 支持 dense backfill
- `final candidates -> optional BGE rerank -> generator`

## 6. trace 中新增的关键字段

当前主线输出 `trace` 中已补充：

- `embedding_model`
- `embedding_dim`
- `qdrant_collection`
- `retrieval_mode`
- `use_rerank`
- `initial_dense_candidates`
- `final_reranked_candidates`

## 7. 当前文档入口

设计方案：

- [integration_plan_vector_stack.md](/home/paper/Benchmark/docs/integration_plan_vector_stack.md)

使用说明：

- [vector_retrieval_usage.md](/home/paper/Benchmark/docs/vector_retrieval_usage.md)

本文档：

- [vector_stack_change_log.md](/home/paper/Benchmark/docs/vector_stack_change_log.md)

## 8. 当前验证状态

已通过的主要测试包括：

- `tests/test_zhipu_embedder.py`
- `tests/test_qdrant_store.py`
- `tests/test_bge_reranker.py`
- `tests/test_build_qdrant_index_script.py`
- `tests/test_traditional_rag.py`
- `tests/test_rewrite_rag.py`
- `tests/test_graph_enhanced_rag.py`
- `tests/test_step6_kbqa_ours.py`
- `tests/test_run_ablation_script.py`

## 9. 当前人工运行前需准备的事项

运行真实 dense 实验前，需要人工确认：

1. 智谱 API Key 已配置
2. 本地 BGE 模型目录已准备
3. Qdrant 使用方式已确定
4. vLLM 的 Llama 服务已启动
5. 已先执行索引构建

## 10. 推荐的最小回归顺序

建议每次环境变更后按这个顺序检查：

1. `build_qdrant_index.py --dry-run`
2. `build_qdrant_index.py --recreate`
3. `run_traditional_rag.py --limit 5`
4. `run_rewrite_rag.py --limit 5`
5. `run_graph_enhanced_rag.py --limit 5`
6. `run_ours_ch4.py --limit 5`
7. `run_ablation.py --limit 5`
