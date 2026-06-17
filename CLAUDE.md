# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境搭建与常用命令

```bash
# 创建虚拟环境 + 安装依赖
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 数据库迁移
python -m rag0 create-tables

# 启动 API 服务 (默认 0.0.0.0:7861)
python -m rag0 serve

# 启动 Streamlit UI (需要先启动 API)
streamlit run ui/streamlit_app.py

# 校验配置
python -m rag0 config-validate
```

### 代码质量检查

```bash
ruff check src/ tests/          # Lint
ruff check --fix src/ tests/    # 自动修复
mypy src/rag0/                  # 类型检查（CI 用 --ignore-missing-imports）
pytest tests/unit/ -v           # 单元测试 (49 个用例)
pytest tests/unit/ -v --cov=rag0 --cov-report=term-missing  # 含覆盖率
pytest tests/unit/test_fusion.py::TestReciprocalRankFusion::test_two_lists_with_overlap -v  # 单测
```

### Docker

```bash
docker compose up -d milvus     # 仅启动 Milvus
docker compose up -d            # 全部服务 (Milvus + API)
docker compose --profile ui up  # 含 Streamlit
```

## 架构

这是一个从旧 `rag0` 项目完全重写的 RAG 框架。核心是**三段式流水线**：

```
用户请求 → FastAPI → RetrievalChain → GenerateChain → SSE 流式响应
                         │
文件上传 → IndexingChain (load → split → multi_vector → store)
```

### 依赖注入

全局入口：`src/rag0/container.py::Container`。在 `create_app()` 时创建一次，通过 `src/rag0/api/deps.py::get_container()` 注入到所有路由。所有组件（LLM、Embedding、VectorStore、Database）均由 Container 统一管理，不存在模块级全局单例。

### 服务注册表

`src/rag0/connectors/registry.py` — 用装饰器替代旧代码的 `if/elif/pass` 工厂：

```python
@loader_registry.register(".pdf")   # 自动注册
class PDFLoader: ...

loader = loader_registry.get(".pdf")  # 按 key 获取
```

三个全局注册表实例：`loader_registry`、`splitter_registry`、`vector_store_registry`。

### 配置系统

`src/rag0/config.py` — 基于 `pydantic-settings.BaseSettings`。加载优先级（低→高）：

```
默认值 → .env → config.yaml → RAG0_ 前缀环境变量
```

子配置类：`LLMConfig`、`EmbeddingConfig`、`VectorStoreConfig`、`DatabaseConfig`、`SplitterConfig`、`RerankerConfig`、`ServerConfig`、`TelemetryConfig`。通过 `RagConfig` 聚合，`get_config()` 返回 LRU 缓存的单例。

### 异常体系

`src/rag0/exceptions.py` — 10 类分层异常。`src/rag0/api/middleware.py` 负责将领域异常映射到 HTTP 状态码（`ValidationError`→400, `ConnectionError`→503, `Rag0Error`→500）。

### 包结构

- **`connectors/`** — 外部服务层：`llm.py`（LiteLLM + 熔断）、`embeddings.py`（sentence-transformers + LRU 缓存）、`vector_store.py`（Milvus 直连 + BM25）、`database.py`（SQLAlchemy 2.0 + Repository）
- **`indexing/`** — 索引流水线：`loaders.py`（PDF/DOCX/TXT）、`splitters.py`（中文递归分割）、`multi_vector.py`（small-to-big/摘要/表格摘要）
- **`retrieval/`** — 检索流水线：`query_expansion.py`（多查询 + HyDE）、`routing.py`（查询→文件路由）、`fusion.py`（RRF + 混合搜索）、`reranker.py`（CrossEncoder + LLM 重排序）
- **`chains/`** — 流水线编排器：`indexing.py`、`retrieval.py`、`generation.py`
- **`api/`** — FastAPI 服务：`app.py`（工厂）、`deps.py`（DI）、`middleware.py`（异常映射）、`routes/health.py`、`routes/knowledge.py`、`routes/chat.py`
- **`caching.py`** — LLM 语义缓存 + Embedding 缓存（内存 LRU + 可选 diskcache）
- **`telemetry.py`** — Langfuse 追踪（惰性加载，可选）
- **`cli.py`** — `python -m rag0 serve|create-tables|config-validate`

### 关键设计决策

| 决策 | 理由 |
|------|------|
| LiteLLM 替代手写 OpenAI wrapper | 统一 100+ 提供商接口，内置重试/降级/速率限制 |
| pymilvus 直连替代 LangChain wrapper | 减少抽象层，支持更多 pymilvus 特性 |
| pydantic-settings 替代 NVIDIA ConfigWizard | 旧 377 行 → 新 205 行，标准生态，IDE 友好 |
| 装饰器注册表替代 if/elif 工厂 | 扩展只需一个装饰器，消除 `pass` 空分支 |
| structlog 替代 logging.config | 结构化键值日志，JSON/控制台双模式 |
| `asyncio.gather` 替代 ThreadPoolExecutor | 真并行，可配置并发数，部分失败不中断 |
| SQLAlchemy 2.0 + Alembic | 声明式模型，数据库迁移可追溯 |

## 两条核心数据流

**索引**：文件 → `loader_registry` 选 Loader → `splitter_registry` 选 Splitter → 分配 UUID → [可选: multi-vector] → `embedding.embed_documents()` → `vector_store.add_documents()` (Milvus) + `file_repo.add_file()` (SQLite)

**检索**：query → [可选: multi-query/HyDE/路由] → `embedding.embed_query()` → dense(Milvus) + sparse(BM25) → RRF 融合 → 去重 → CrossEncoder 重排序 → top-k `ScoredDocument`

## 重写历史总结

本项目（`my-rag-project-0`）是对 `C:\Users\11207\Desktop\proj\rag0` 的完全重写。

### 旧工程主要问题（已全部修复）

| 严重度 | 问题 | 修复位置 |
|--------|------|---------|
| 🔴 Bug | `split_documents` 只处理 `documents[:1]` | `indexing/splitters.py` |
| 🔴 Bug | `NameError: text` 未定义 | `indexing/loaders.py:DOCXLoader` |
| 🔴 Bug | `@lru_cache` 多KB共享向量存储 | `connectors/vector_store.py` |
| 🔴 Bug | `clear_knowledge_base` 发错误 JSON | RESTful DELETE 路由 |
| 🔴 安全 | API Key 明文硬编码 | `.env` + 标准环境变量 |
| 🟡 设计 | 模块级全局单例 | `container.py` DI |
| 🟡 设计 | 工厂 `pass` 空分支 | `registry.py` 装饰器 |
| 🟡 Bug | LLM 错误返回 `''` 静默吞错 | `LLMConnectionError` + 熔断 |
| 🟡 Bug | `LLMReranker` 定义但不可达 | `type=="llm"` 路由 |
| 🟡 Bug | SSE async 内同步阻塞 | 真异步 `AsyncIterator` |
| 🟢 缺失 | 零测试文件 | 5 个测试模块, 49 用例 |
| 🟢 缺失 | 无混合搜索 | BM25 + RRF |
| 🟢 缺失 | 无缓存层 | `caching.py` |

### 新旧工程对比

| 指标 | 旧 (rag0) | 新 (my-rag-project-0) |
|------|:---:|:---:|
| Python 文件 | 63 | 49 |
| 代码行数 | 4,278 | ~5,400 |
| 空文件/Stub | 18 | 9 (仅 `__init__`) |
| 测试文件 | 0 | 5 |
| 测试用例 | 0 | 49 |
| 裸 `except:` | 2 | 1 |
| print() 调试 | 24 | 10 (仅CLI) |
| API Key 硬编码 | 2 把 | 0 |
| Docker 支持 | 无 | Dockerfile + compose |
| CI/CD | 无 | GitHub Actions (ruff+mypy+pytest) |

### CI 配置注意事项

- CI 使用 Python 3.11/3.12，`pyproject.toml` 要求 `>=3.11`，但代码兼容 3.10+
- Coverage 阈值设为 20%（单元测试覆盖 config/fusion/caching/splitter/generation，connectors/API 需集成测试）
- 新增依赖时必须加入 `pyproject.toml` 的 `dependencies`，本地 conda 环境可能有遗留包
- `# noqa: B008` 用于 FastAPI 的 `Body()/File()/Form()` 默认值（标准 FastAPI 模式）
- `# type: ignore[override]` 用于 LangChain splitter 的 Liskov 替换违反
