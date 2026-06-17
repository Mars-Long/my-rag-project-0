# RAG0 — 现代检索增强生成框架

一个从头重写的 RAG（Retrieval-Augmented Generation）框架，具有混合搜索、异步支持和清晰的分层架构。

## 特性

- **三段式流水线**：索引 → 检索 → 生成，职责清晰
- **混合搜索**：密集向量（Milvus）+ 稀疏检索（BM25）→ RRF 融合
- **异步优先**：全链路 `asyncio`，支持 SSE 流式输出
- **多向量策略**：small-to-big、文本摘要、表格摘要
- **可插拔架构**：装饰器注册表（Loader、Splitter、VectorStore）消除硬编码工厂
- **结构化日志**：structlog — 开发时彩色控制台，生产环境 JSON
- **Docker 支持**：多阶段 Dockerfile + docker-compose（Milvus + API + UI）

## 快速开始

### 1. 克隆并配置

```bash
git clone <repo-url> rag0
cd rag0
cp .env.example .env        # 编辑 .env，填入你的 API Key
cp config.yaml.example config.yaml
```

### 2. 启动 Milvus

```bash
docker compose up -d milvus
```

### 3. 安装并启动

```bash
pip install -e ".[server]"
python -m rag0 create-tables
python -m rag0 serve
```

打开 http://localhost:7861/docs 查看 API 文档。

### 可选：启动 Streamlit 前端

```bash
streamlit run ui/streamlit_app.py
```

## 项目结构

```
src/rag0/
├── config.py                # Pydantic Settings（YAML + 环境变量）
├── container.py             # 依赖注入容器
├── exceptions.py            # 分层异常（10 类）
├── types.py                 # ScoredDocument, Message
├── logging.py               # structlog 配置
├── telemetry.py             # Langfuse（可选）
├── cli.py                   # 命令行入口
├── connectors/              # 外部服务抽象
│   ├── registry.py          # 服务注册表（装饰器模式）
│   ├── llm.py               # LiteLLM（100+ LLM 提供商）
│   ├── embeddings.py        # sentence-transformers
│   ├── vector_store.py      # Milvus + BM25 稀疏检索
│   └── database.py          # SQLAlchemy 2.0 + Repository
├── indexing/                # 索引流水线
│   ├── loaders.py           # PDF, DOCX, TXT 加载器
│   ├── splitters.py         # 中文文本分割器
│   └── multi_vector.py      # 多向量策略
├── retrieval/               # 检索流水线
│   ├── query_expansion.py   # 多查询 + HyDE
│   ├── routing.py           # 查询→文件路由
│   ├── fusion.py            # RRF + 混合搜索
│   └── reranker.py          # 交叉编码器 + LLM 重排序
├── chains/                  # 流水线编排
│   ├── indexing.py
│   ├── retrieval.py
│   └── generation.py
└── api/                     # FastAPI 服务
    ├── app.py               # 应用工厂
    ├── deps.py              # 依赖注入
    ├── middleware.py         # 统一异常处理
    └── routes/
        ├── health.py        # 健康检查
        ├── knowledge.py     # 知识库 CRUD
        └── chat.py          # SSE 流式对话
```

## 配置

所有配置项可通过以下方式设置（优先级从低到高）：

1. `config.py` 中的默认值
2. `.env` 文件
3. `config.yaml`
4. `RAG0_` 前缀的环境变量

```bash
# 环境变量示例
export DEEPSEEK_API_KEY=sk-xxx
export RAG0_LLM__MODEL_NAME=deepseek-chat
export RAG0_EMBEDDING__MODEL_NAME=BAAI/bge-large-zh-v1.5
```

## 技术栈

| 组件 | 方案 |
|------|------|
| LLM | LiteLLM（OpenAI / DeepSeek / Anthropic 等 100+ 提供商） |
| Embedding | sentence-transformers（HuggingFace 模型） |
| 向量存储 | Milvus（pymilvus 直连） |
| 稀疏检索 | BM25（rank-bm25） |
| 数据库 | SQLite + SQLAlchemy 2.0 + Alembic |
| API | FastAPI + SSE 流式 |
| 日志 | structlog |
| 配置 | pydantic-settings |

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
