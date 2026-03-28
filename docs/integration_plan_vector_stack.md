# 向量检索能力接入前置审计与最小侵入改造方案

## 1. 目标与边界

本文档用于当前仓库的“向量检索能力接入”前置审计，目标是在不大改现有实验结构的前提下，为后续接入以下技术栈提供设计方案：

- 向量数据库：`qdrant-client`
- embedding：智谱 `Embedding-3`
- rerank：本地 `BGE reranker`

本次仅输出设计，不直接修改业务代码。改造边界如下：

- 尽量不破坏现有 benchmark schema 和输出格式
- 不修改 `external/` 目录
- 优先保留现有脚本入口和 pipeline 结构
- 需要为后续逐步接入 `Traditional RAG`、`Rewrite-RAG`、`Graph-enhanced RAG`、`Ours-Ch4` 预留统一能力层

## 2. 当前代码结构梳理

### 2.1 当前真实运行主线

当前真实实验入口主要在 `scripts/` + `src/` 这一条线上：

- `scripts/run_traditional_rag.py`
- `scripts/run_rewrite_rag.py`
- `scripts/run_graph_enhanced_rag.py`
- `scripts/run_ours_ch4.py`
- `scripts/run_all_baselines.py`

它们直接调用：

- `src/pipelines/traditional_rag.py`
- `src/pipelines/rewrite_rag.py`
- `src/pipelines/graph_enhanced_rag.py`
- `src/pipelines/ours_ch4.py`

公共数据协议与输出契约位于：

- `src/core/schema.py`
- `src/dataio/loaders.py`
- `src/dataio/normalizers.py`
- `src/evaluation/retrieval_metrics.py`

这部分已经形成一条可运行、可测试、可输出 `predictions.jsonl` / `metrics.json` 的主路径，后续向量能力应优先接入这里。

### 2.2 当前检索相关模块分层

当前 `src/` 里的检索相关实现大致分四层：

1. 数据协议层
- `src/core/schema.py`
- `RetrievedDocument`
- `RetrievalResult`
- `PipelineRunRecord`

2. 文本索引与基础组件层
- `src/pipelines/base.py`
- `PublicIndex`
- `MockEmbedder`
- `MockReranker`
- `MockGenerator`

3. pipeline 层
- `src/pipelines/traditional_rag.py`
- `src/pipelines/rewrite_rag.py`
- `src/pipelines/graph_enhanced_rag.py`
- `src/pipelines/ours_ch4.py`

4. 图与 chapter-4 辅助模块
- `src/retrievers/graph_retriever.py`
- `src/modules/graph_expander.py`
- `src/modules/graph_organizer.py`
- `src/modules/relation_driven_retriever.py`
- `src/modules/text_compensator.py`
- `src/modules/query_rewriter.py`
- `src/modules/skeleton_extractor.py`

### 2.3 `benchmark_host/` 的定位

仓库中还存在一套 `src/benchmark_host/*` 实验宿主骨架，包含：

- `src/benchmark_host/components/indexes.py`
- `src/benchmark_host/components/retrievers.py`
- `src/benchmark_host/pipelines/experiment.py`
- `src/benchmark_host/config/models.py`

但从当前 README、脚本入口、测试覆盖和 `scripts/` 的使用方式看，这套代码更像“平行原型”或早期统一宿主方案，不是当前主实验入口。

结论：

- 本轮向量栈接入应优先改造 `src/` 主线
- `benchmark_host/` 仅做后续对齐参考，不作为第一阶段落点

## 3. 当前检索 / embedding / rerank 抽象审计

## 3.1 当前检索实现是否为占位

结论：当前文本检索主实现本质上是占位式 lexical baseline，不是实际向量检索。

证据如下：

- `src/pipelines/base.py` 中 `PublicIndex.search()` 逐条遍历 section，用 `overlap_score(query, document.full_text)` 打分
- `overlap_score()` 基于 token 集合重叠，不依赖任何 embedding 模型
- 检索结果 `metadata["retriever"]` 被写死为 `"mock_lexical"`

这说明当前 `Traditional RAG` 和 `Rewrite-RAG` 的检索核心是：

- 词面召回
- 小规模 demo 可运行
- 不具备真实 dense retrieval 能力

## 3.2 当前 embedding 抽象是否为占位

结论：是，占位实现。

`src/pipelines/base.py` 中：

- `MockEmbedder` 文档字符串明确写了 `Placeholder embedder`
- `score()` 仍然调用 `overlap_score()`

现状说明：

- 仓库中没有真实 embedding client
- 没有离线建库流程
- 没有向量维度、批量 embedding、缓存、collection 管理等能力

## 3.3 当前 rerank 抽象是否为占位

结论：是，占位实现。

`src/pipelines/base.py` 中：

- `MockReranker.rerank()` 用 `overlap_score(query, item.content)` 重新排序
- `metadata` 只追加 `"reranked": True`

现状说明：

- 没有 cross-encoder / sequence classification 模型
- 没有本地模型加载
- 没有 batched rerank
- 没有 score calibration

## 3.4 图检索与 chapter-4 模块是否为真实实现

结论：是“可运行逻辑骨架”，但检索底座仍依赖占位式 lexical search。

具体判断：

- `src/retrievers/graph_retriever.py`
  - seed retrieval 依赖 `PublicIndex.search()`
  - graph expansion / organization 有真实流程，但不是基于向量召回

- `src/modules/relation_driven_retriever.py`
  - 文本部分依赖 `PublicIndex.search()`
  - 再叠加 `GraphIndex` 中的 entity / relation / constraint hints
  - 更像“heuristic fusion”，不是 dense + graph fusion

- `src/modules/text_compensator.py`
  - backfill 仍然调用 `PublicIndex.search()`
  - 属于合理的补偿逻辑，但基础召回仍是 lexical

- `src/modules/query_rewriter.py`
- `src/modules/skeleton_extractor.py`
  - 这两者不是向量能力占位，而是 query 改写 / skeleton 提取的轻量启发式实现
  - 其中 `stub_predicted` 明确属于启发式占位

总体判断：

- pipeline 编排与 trace/output contract 已经具备
- 真正缺的是“统一可替换的检索后端层”

## 4. 设计原则

为降低侵入性，推荐遵循以下原则：

1. 保留 `PipelineRunRecord`、`RetrievalResult`、`RetrievedDocument` 不变
2. 保留 `scripts/run_*.py` 的主入口文件和参数风格
3. 让各 pipeline 继续面向“检索器接口 / reranker 接口”工作，而不是直接面向 Qdrant SDK
4. 将“建索引 / embedding / 向量检索 / rerank”从 `pipelines/base.py` 中拆出成独立模块
5. 第一阶段只替换文本召回与 rerank，不改变 graph / skeleton / compensation 的上层编排

## 5. 最小侵入改造方案

## 5.1 总体方案

建议引入一层新的“向量检索基础设施模块”，把当前 `PublicIndex + MockEmbedder + MockReranker` 的职责拆开：

- 文档加载与 section 转换：保留现有 JSONL 输入格式
- embedding client：新增智谱 Embedding-3 封装
- vector store：新增 Qdrant 访问封装
- retriever：新增 dense retriever 适配器
- reranker：新增本地 BGE reranker 适配器
- index builder：新增离线建库脚本或可复用函数

pipeline 只拿到统一接口，例如：

- `TextRetriever.retrieve(query, top_k) -> list[RetrievedDocument]`
- `DocumentReranker.rerank(query, documents) -> list[RetrievedDocument]`

这样可以做到：

- `Traditional RAG` 直接换底座
- `Rewrite-RAG` 继续只改 query，不改输出结构
- `Graph-enhanced RAG` 只替换 seed retrieval
- `Ours-Ch4` 只替换 relation-driven / compensation 中的文本召回

## 5.2 推荐新增抽象

建议新增以下协议或基础类：

- `SectionCorpus`
  - 从 `markdown_sections.jsonl` 加载 section
  - 持有 `by_section_id`、`by_doc_id`
  - 负责把 Qdrant hit 转回 `RetrievedDocument`

- `EmbeddingClient`
  - `embed_query(text)`
  - `embed_documents(texts)`

- `VectorStore`
  - `ensure_collection(...)`
  - `upsert_sections(...)`
  - `search(query_vector, top_k, filters=None)`

- `TextRetriever`
  - 统一给 pipeline 调用
  - 屏蔽 lexical / dense / hybrid 的差异

- `DocumentReranker`
  - 统一 rerank 接口

第一阶段不建议一次性做复杂 hybrid 检索抽象，先让 dense path 跑通。

## 5.3 对现有 pipeline 的接入方式

### Traditional RAG

当前：

- `TraditionalRAGPipeline` 依赖 `PublicIndex.search()`

建议：

- 改为依赖 `TextRetriever`
- 保留 `TraditionalRAGConfig(top_k, rerank)`
- 不改 `run()` 的输出字段

### Rewrite-RAG

当前：

- rewrite 后再次调用 `index.search()`

建议：

- 保留 query rewrite 逻辑不变
- 将 `index.search()` 替换为 `TextRetriever.retrieve()`
- comparison 逻辑保持不变

### Graph-enhanced RAG

当前：

- `GraphRetriever` 内部的 seed retrieval 使用 `PublicIndex.search()`

建议：

- `GraphRetriever` 注入 `TextRetriever`
- graph expansion / organization 保持原样
- 第一阶段仅把 seed retrieval 切到 dense 检索

### Ours-Ch4

当前：

- `RelationDrivenRetriever` 和 `TextCompensator` 都依赖 `PublicIndex.search()`

建议：

- 将两者依赖改为 `TextRetriever`
- 保留 skeleton / graph hints / text compensation 逻辑
- 这样可以把 relation-driven 方案演进为“dense seed + graph hint fusion”

## 5.4 配置接入方案

当前配置分散在：

- `config/default_config.json`
- `configs/ch4_demo.json`
- 各 `scripts/run_*.py` 参数

建议新增一组 retrieval stack 配置，并尽量向下兼容现有参数：

推荐新增配置块：

```json
{
  "retrieval_stack": {
    "backend": "qdrant",
    "collection_name": "benchmark_sections",
    "vector_size": 1024,
    "distance": "Cosine",
    "embedding_provider": "zhipu",
    "embedding_model": "embedding-3",
    "reranker_backend": "bge_local",
    "reranker_model_name_or_path": "BAAI/bge-reranker-v2-m3",
    "use_rerank": true
  }
}
```

同时建议保留现有：

- `top_k`
- `seed_top_k`
- `expand_k`
- `disable-rerank`

避免一次性修改所有脚本调用方式。

## 5.5 数据落库与 collection 设计

Qdrant collection 中每条 point 建议以一个 markdown section 为单位，主键直接使用现有 `section_id`。

payload 建议至少保留：

- `section_id`
- `source_id`
- `doc_title`
- `section_path`
- `content`
- `file_path`
- `block_type`
- `char_len`

理由：

- 可以无损映射回当前 `RetrievedDocument`
- 后续支持 doc / section 级过滤
- 不影响现有输出 schema

## 5.6 rerank 接入方案

本地 BGE reranker 建议作为可插拔组件，不直接写进 pipeline。

推荐行为：

- dense retriever 先从 Qdrant 召回 top-N
- reranker 在内存中重排
- pipeline 最终仍输出 `RetrievedDocument`

建议元数据补充：

- `metadata["retriever"] = "qdrant_dense"`
- `metadata["embedding_model"] = "embedding-3"`
- `metadata["reranker"] = "bge_local"` 或 `None`
- `metadata["vector_score"]`
- `metadata["rerank_score"]`

这不会破坏现有 schema，因为 `metadata` 已是开放字段。

## 6. 需要新增的模块列表

建议新增模块尽量集中在 `src/retrieval/` 或 `src/infra/retrieval/` 下。为减少和现有 `src/retrievers/` 的 graph 检索命名冲突，推荐新建 `src/retrieval/`。

推荐新增：

- `src/retrieval/__init__.py`
- `src/retrieval/corpus.py`
  - section 语料加载与 hit -> `RetrievedDocument` 转换

- `src/retrieval/interfaces.py`
  - `EmbeddingClient`
  - `TextRetriever`
  - `DocumentReranker`
  - `VectorStore`

- `src/retrieval/qdrant_store.py`
  - `QdrantClient` 封装
  - collection ensure / upsert / search

- `src/retrieval/zhipu_embedding.py`
  - 智谱 `Embedding-3` client 封装
  - 批量 embedding

- `src/retrieval/bge_reranker.py`
  - 本地 BGE reranker 封装

- `src/retrieval/dense_retriever.py`
  - query embedding + qdrant search + hit convert

- `src/retrieval/index_builder.py`
  - 从 `markdown_sections.jsonl` 建 collection
  - 支持首次建库和增量更新

- `src/retrieval/factory.py`
  - 按配置创建 retriever / reranker / corpus

可选新增：

- `scripts/build_vector_index.py`
  - 单独的向量建库脚本

- `tests/test_dense_retriever.py`
- `tests/test_bge_reranker.py`
- `tests/test_vector_index_builder.py`

## 7. 需要修改的现有文件列表

以下文件建议修改，且都属于主线、侵入相对可控。

### 高优先级

- `src/pipelines/base.py`
  - 目标：逐步去掉 `MockEmbedder` / `MockReranker` 的中心地位
  - 做法：保留 `BasePipeline.build_retrieval_result()`，但将 `PublicIndex` 的职责迁移到新 retrieval 层

- `src/pipelines/traditional_rag.py`
  - 目标：依赖新 `TextRetriever` / `DocumentReranker`

- `src/pipelines/rewrite_rag.py`
  - 目标：将原始 query / 改写 query 的检索都切到新 retriever

- `src/retrievers/graph_retriever.py`
  - 目标：seed retrieval 切换为新 retriever

- `src/modules/relation_driven_retriever.py`
  - 目标：文本召回从 `PublicIndex.search()` 切到新 retriever

- `src/modules/text_compensator.py`
  - 目标：补偿召回从 `PublicIndex.search()` 切到新 retriever

- `src/pipelines/graph_enhanced_rag.py`
  - 目标：构造函数改为接收新 graph retriever 依赖组合

- `src/pipelines/ours_ch4.py`
  - 目标：接收新 retriever / reranker 依赖

### 配置与入口

- `scripts/run_traditional_rag.py`
- `scripts/run_rewrite_rag.py`
- `scripts/run_graph_enhanced_rag.py`
- `scripts/run_ours_ch4.py`
- `scripts/run_all_baselines.py`

目标：

- 保留原有参数
- 增加少量向量栈参数或配置文件读取
- 默认不改变输出位置与格式

### 依赖声明

- `requirements.txt`
- `pyproject.toml`

建议新增依赖：

- `qdrant-client`
- 智谱 SDK 或基于 `requests` 的轻量 client 依赖
- `sentence-transformers` 或项目选定的本地 BGE 推理依赖
- `torch`

说明：

- 若担心依赖过重，可先在文档阶段预留，实际分两步安装
- 但 reranker 要落地，本地模型依赖最终绕不过去

### 测试

- `tests/test_traditional_rag.py`
- `tests/test_rewrite_rag.py`
- `tests/test_graph_enhanced_rag.py`
- `tests/test_step6_kbqa_ours.py`
- `tests/test_run_all_baselines_script.py`

目标：

- 保持现有 schema / output 断言不变
- 放宽对 `"mock_lexical"`、`MockReranker` 这类实现细节的假设
- 增加对新 retriever contract 的测试

## 8. 不建议本轮修改的文件

以下内容本轮不建议碰：

- `external/` 全部内容
- `src/core/schema.py`
  - 当前 schema 已能承接新检索结果，没必要重构

- `src/dataio/loaders.py`
  - 除非要复用 section 转换函数，否则不必动 benchmark sample 加载逻辑

- `src/evaluation/retrieval_metrics.py`
  - 指标依赖 section_id 命中，不依赖具体检索实现

- `src/modules/query_rewriter.py`
- `src/modules/skeleton_extractor.py`
- `src/modules/skeleton_utils.py`
  - 这些属于上层 query / skeleton 逻辑，不是向量能力接入阻塞点

## 9. 推荐开发顺序

建议按“先打通基础设施，再逐条 pipeline 切换”的顺序推进。

### 第 1 阶段：抽离统一检索接口

1. 新增 `src/retrieval/interfaces.py`
2. 新增 `src/retrieval/corpus.py`
3. 让 pipeline 从依赖 `PublicIndex` 转为依赖 `TextRetriever` 协议
4. 暂时用一个 lexical adapter 适配现有行为，保证测试不炸

目标：

- 先把“调用方式”稳定下来
- 不立即引入 Qdrant 和模型依赖

### 第 2 阶段：接入 Qdrant + 智谱 Embedding-3

1. 实现 `zhipu_embedding.py`
2. 实现 `qdrant_store.py`
3. 实现 `dense_retriever.py`
4. 新增 `scripts/build_vector_index.py`
5. 在 `Traditional RAG` 上先跑通 dense retrieval

目标：

- 最先替换最简单、链路最短的 pipeline

### 第 3 阶段：接入本地 BGE reranker

1. 实现 `bge_reranker.py`
2. 替换 `Traditional RAG` / `Rewrite-RAG` 的 `MockReranker`
3. 在 `metadata` 中补充 `vector_score` / `rerank_score`

目标：

- 不改输出 schema
- 提升 dense top-k 的排序质量

### 第 4 阶段：扩展到 Rewrite-RAG

1. 将 `RewriteRAGPipeline` 的两次检索都切换到 dense retriever
2. 保持 rewrite comparison 逻辑不变
3. 验证输出 JSONL 与 metrics 兼容

### 第 5 阶段：扩展到 Graph-enhanced RAG

1. 将 `GraphRetriever` 的 seed retrieval 切到 dense retriever
2. 保留现有 graph expansion / organizer 逻辑
3. 必要时为 graph organizer 引入融合分数策略

### 第 6 阶段：扩展到 Ours-Ch4

1. 将 `RelationDrivenRetriever` 改为 dense seed + graph hints
2. 将 `TextCompensator` 改为 dense backfill
3. 最后再讨论 relation-driven score fusion 的调权

这个顺序的好处是：

- 每一步都能保持可运行
- 每一步都能复用已有测试与输出格式
- 风险从低到高逐步推进

## 10. 风险与注意事项

### 10.1 当前测试对“可运行骨架”有依赖

现有测试更多是在验证：

- pipeline 能跑通
- 输出字段完整
- trace 有内容

不是在验证真实向量效果。因此接入新栈时要避免：

- 把测试绑死在真实服务可用性上
- 让单元测试依赖在线 embedding API
- 让 CI 必须有 GPU 才能跑

建议：

- 单元测试使用 mock retriever / fake vector store
- 真正的端到端验证放到可选脚本或集成测试

### 10.2 reranker 本地依赖较重

BGE reranker 会引入：

- 模型权重下载
- `torch`
- 可能的 GPU / CPU 性能差异

建议：

- 默认提供 `--disable-rerank`
- 保留无 rerank 路径
- 让 reranker 初始化延迟到真正需要时

### 10.3 Qdrant 建库是新流程，不应塞进现有实验脚本

不建议让 `run_traditional_rag.py` 在运行时顺手建索引，因为这会：

- 增加脚本副作用
- 让实验时间不可控
- 让错误边界混乱

建议独立脚本：

- 先建库
- 再运行实验

### 10.4 `benchmark_host/` 与 `src/` 双线并存

如果后续要统一代码，建议在主线接通后再回收 `benchmark_host/`。本轮不要同时改两条线，否则会扩大改造面。

## 11. 本轮审计结论

结论可以概括为四点：

1. 当前仓库的实验主线已经具备稳定的 pipeline 编排、评测与输出协议
2. 当前 retrieval / embedding / rerank 核心能力基本都是占位实现，尤其是 `PublicIndex`、`MockEmbedder`、`MockReranker`
3. 最适合的最小侵入方案不是重写 pipeline，而是在 `src/` 主线下新增统一 retrieval 基础设施层，并逐步替换文本召回与 rerank 底座
4. 推荐按 `Traditional RAG -> Rewrite-RAG -> Graph-enhanced RAG -> Ours-Ch4` 的顺序逐步接入，这样能最大限度保留现有脚本入口、schema 和输出格式

## 12. 建议的下一步执行项

进入真正编码阶段时，建议按以下最小任务拆解：

1. 新建 `src/retrieval/` 基础模块与接口
2. 先做 lexical adapter，替换 `PublicIndex` 的直接耦合
3. 新增 `scripts/build_vector_index.py`
4. 接入 `Qdrant + 智谱 Embedding-3`
5. 先在 `Traditional RAG` 上完成端到端验证
6. 再接入本地 `BGE reranker`
7. 最后逐步切到 `Rewrite-RAG`、`Graph-enhanced RAG`、`Ours-Ch4`
