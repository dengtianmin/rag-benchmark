# TEI Reranker Usage

本仓库支持通过本地 Docker 启动的 TEI 服务执行 rerank，并保留 `mock` 作为显式 fallback backend。

## 必要配置

在 `.env` 或环境变量中设置：

```bash
USE_RERANK=true
RERANK_BACKEND=tei
TEI_RERANK_URL=http://127.0.0.1:8080
TEI_RERANK_TIMEOUT=30
TEI_RERANK_API_KEY=
TEI_RERANK_MAX_RETRIES=1
RERANK_TOP_N=5
RERANK_ALLOW_MOCK_FALLBACK=false
```

如果需要切回占位实现：

```bash
RERANK_BACKEND=mock
```

## 启动本地 TEI 服务

示例前提：

- TEI 服务地址：`http://127.0.0.1:8080`
- rerank 接口：`POST /rerank`
- 本地模型目录：`/home/paper/Benchmark/models/bge-reranker-v2-m3`

参考启动方式：

```bash
docker run --rm -p 8080:80 \
  -v /home/paper/Benchmark/models/bge-reranker-v2-m3:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data \
  --task rerank
```

实际镜像标签和参数请按本机已验证版本调整。

## 行为说明

- 现有召回逻辑保持不变，rerank 在召回后执行。
- `Traditional RAG`、`Rewrite-RAG`、`Graph-enhanced RAG`、`Ours-Ch4` 已接入统一 reranker 工厂。
- benchmark 主输出字段不变，新增信息落在 `trace` 中。
- 当 `RERANK_ALLOW_MOCK_FALLBACK=true` 时，TEI 调用失败会回退到 `mock`，并在 `trace` 中记录 fallback 与错误信息。

## 常见问题

`TEI rerank request timed out`

- 检查容器是否正常运行。
- 检查 `TEI_RERANK_URL` 是否可达。
- 增大 `TEI_RERANK_TIMEOUT`。

`TEI rerank request failed with status ...`

- 检查 `/rerank` 接口路径是否正确。
- 检查请求体是否符合当前 TEI 版本要求。
- 查看容器日志确认模型是否加载成功。

`Unexpected TEI rerank response format`

- 当前实现兼容 `{"results": [...]}`、`{"data": [...]}` 和直接数组格式。
- 如果本地 TEI 返回结构不同，需要按实际响应补充解析逻辑。
