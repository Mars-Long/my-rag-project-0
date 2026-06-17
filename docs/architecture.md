# RAG0 Architecture

> 一份 5 分钟读懂整个系统设计的文档。

## 系统全景

```
                          ┌──────────────┐
                          │   Streamlit  │  (ui/streamlit_app.py)
                          │     UI       │  HTTP-only, 不直接访问DB
                          └──────┬───────┘
                                 │ SSE / JSON
                                 ▼
                          ┌──────────────┐
                          │   FastAPI    │  (src/rag0/api/)
                          │  Port 7861   │  CORS + RequestID + ExceptionMiddleware
                          └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │Indexing  │ │Retrieval │ │Generation│
            │Chain     │ │Chain     │ │Chain     │
            └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │             │             │
                 └──────┬──────┘             │
                        │                    │
              ┌─────────┼──────────┐         │
              ▼         ▼          ▼         ▼
        ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────┐
        │ Milvus  │ │BM25  │ │SQLite  │ │LiteLLM│
        │(dense)  │ │(sparse)│ │ 元数据 │ │(生成) │
        └─────────┘ └──────┘ └────────┘ └──────┘
              ▲                     ▲
              │                     │
        ┌──────────┐          ┌──────────┐
        │ Embedding│          │ Alembic  │
        │(bge-large│          │(迁移管理) │
        │ -zh-v1.5)│          └──────────┘
        └──────────┘
```

## 目录地图

```
src/rag0/
├── config.py          ← 一切配置的入口 (pydantic-settings, YAML+env)
├── container.py       ← DI 容器, 组装所有组件
├── exceptions.py      ← 10 类分层异常
├── types.py           ← ScoredDocument, Message, IndexingResult
├── logging.py         ← structlog (dev彩色 / prod JSON)
├── caching.py         ← LLM语义缓存 + Embedding缓存
├── telemetry.py       ← Langfuse 可选追踪
├── cli.py             ← python -m rag0 serve|create-tables|config-validate
│
├── connectors/        ← 外部服务层 (无业务逻辑)
│   ├── registry.py    ← @register_loader / @register_splitter 装饰器
│   ├── llm.py         ← LiteLLM (100+提供商, 熔断, 重试, 异步)
│   ├── embeddings.py  ← sentence-transformers (GPU自动检测, LRU缓存)
│   ├── vector_store.py← Milvus直连 + BM25稀疏检索
│   └── database.py    ← SQLAlchemy 2.0 ORM + Repository
│
├── indexing/          ← 索引流水线组件
│   ├── loaders.py     ← PDFLoader, DOCXLoader, TextLoader
│   ├── splitters.py   ← ChineseRecursiveTextSplitter, ChineseTextSplitter
│   └── multi_vector.py← small-to-big, 文本摘要, 表格摘要
│
├── retrieval/         ← 检索流水线组件
│   ├── query_expansion.py ← 多查询生成 + HyDE
│   ├── routing.py     ← 查询→文件路由 (JSON结构化输出)
│   ├── fusion.py      ← RRF融合 + 混合搜索编排
│   └── reranker.py    ← CrossEncoder + LLM重排序
│
├── chains/            ← 流水线编排器
│   ├── indexing.py    ← load→split→multi_vector→store
│   ├── retrieval.py   ← expand→route→search→fuse→rerank
│   └── generation.py  ← augment→generate (异步+流式)
│
└── api/               ← FastAPI 服务
    ├── app.py         ← create_app() 工厂
    ├── deps.py        ← get_container() DI
    ├── middleware.py   ← 统一异常→HTTP状态码映射
    └── routes/
        ├── health.py  ← GET /health, /health/ready
        ├── knowledge.py← 知识库CRUD (GET/POST/DELETE)
        └── chat.py    ← POST /chat (SSE流式)
```

## 两条核心数据流

### 流 1：文档索引

```
用户上传 PDF
      │
      ▼
POST /knowledge-bases/{name}/documents
      │
      ▼
IndexingChain.index_files([path1, path2, ...])
      │
      ├─► load:   PDFLoader.load() → [Document(page1), Document(page2), ...]
      │                (PyMuPDF提取文字+表格, RapidOCR识别图片)
      │
      ├─► split:  ChineseRecursiveTextSplitter.split_documents()
      │                (中文感知的递归分割: 段落→句子→分号→逗号→字符)
      │
      ├─► multi_vector (可选):
      │       split_smaller_chunks()    → 子分块 (parent_id关联)
      │       generate_text_summaries() → LLM生成摘要
      │       generate_table_summaries()→ LLM总结表格
      │
      └─► store:
              vector_store.add_documents() → Milvus (dense embedding)
              [将来] BM25Retriever.index() → 内存 (sparse)
              file_repo.add_file()        → SQLite (元数据)
```

### 流 2：问答检索

```
用户提问: "Python是什么时候发布的？"
      │
      ▼
POST /chat {query, knowledge_base_name}
      │
      ▼
RetrievalChain.retrieve(query, kb_name)
      │
      ├─► Pre-retrieval:
      │       generate_multi_queries() → [原始Q, 变体Q1, 变体Q2, 变体Q3]
      │       generate_hyde_document() → 假想回答文本 (可选)
      │       route_query_to_file()    → 锁定具体文件 (可选)
      │
      ├─► Retrieval (每个查询并行):
      │       embedding.embed_query(q) → [0.12, 0.34, ...]
      │       ┌─ dense:  Milvus.search()    → [ScoredDoc x20]
      │       └─ sparse: BM25.search()      → [(idx, score) x20]
      │
      ├─► Post-retrieval:
      │       reciprocal_rank_fusion() → 合并多查询+密集+稀疏结果
      │       dedup by doc_id
      │       reranker.rank()          → CrossEncoder精准重排 top-5
      │
      └─► 返回 [ScoredDoc x5]
            │
            ▼
GenerateChain.generate(query, docs)
      │
      ├─► augment:  将 docs 格式化为提示词上下文
      └─► generate: LiteLLM → DeepSeek/OpenAI → "Python于1991年发布..."
            │
            ▼
      SSE: data: {"content": "Python"} data: {"content": "于"} ...
```

## 关键设计决策

### 1. 为什么用 LiteLLM 而不是直接调 OpenAI SDK？

LiteLLM 提供统一接口，换模型只需改配置：

```yaml
# 从 DeepSeek 换到 OpenAI — 只改一行
llm:
  model_name: gpt-4o  # 原来是 deepseek-chat
```

内置重试、降级、速率限制——这些我们自己写至少要 200 行。

### 2. 为什么用 pydantic-settings 而不是保留 ConfigWizard？

旧 ConfigWizard 377 行，来自 NVIDIA NeMo，项目实际只用不到 30%。

```
旧: 377 行 ConfigWizard (NVIDIA NeMo)
新: 205 行 pydantic-settings (标准库生态)
                     ↓
            IDE 自动补全 + 字段级校验
```

### 3. 为什么不经过 LangChain 调 Milvus？

LangChain 的 Milvus wrapper 有滞后性（跟不上 pymilvus 版本）、多一层序列化开销、不支持某些高级操作。

```python
# 旧: LangChain wrapper
from langchain.vectorstores import Milvus
vs = Milvus(embedding_function, collection_name="kb1", ...)

# 新: pymilvus 直连
from pymilvus import MilvusClient
client = MilvusClient(uri="http://127.0.0.1:19530")
```

### 4. 为什么用装饰器注册表而不是 if/elif 工厂？

```python
# 旧: 添加新loader要改2个地方
LOADER_MAPPING = {".pdf": CustomizedOcrPdfLoader}  # 1. 注册
def get_loader(name):                              # 2. 工厂函数
    if ".pdf" in name: ...

# 新: 添加新loader只改1个地方
@loader_registry.register(".epub")  # 装饰器自动注册
class EPUBLoader:
    ...
```

### 5. 为什么用 structlog 而不是 logging？

```python
# 旧: 非结构化文本, 难以搜索
logger.info(f"文档索引完成: {file}, chunks={len(chunks)}")

# 新: 结构化键值对, grep/jq 友好
logger.info("文档索引完成", file=file, chunks=len(chunks))
# → {"event": "文档索引完成", "file": "report.pdf", "chunks": 42, "timestamp": "..."}
```

## 扩展指南

### 添加新的文档加载器

```python
# src/rag0/indexing/loaders.py

from rag0.connectors.registry import loader_registry

@loader_registry.register(".epub")
class EPUBLoader:
    def __init__(self, file_path):
        self._path = file_path

    def load(self) -> list[Document]:
        # 你的 EPUB 解析逻辑
        ...
```

### 添加新的向量数据库

```python
# src/rag0/connectors/your_store.py

class YourVectorStore(VectorStoreInterface):
    def create_collection(self, name, dim): ...
    def add_documents(self, collection, docs, embeddings): ...
    def search(self, collection, query_emb, top_k, filters): ...
    def delete_by_filter(self, collection, filter_expr): ...
    def drop_collection(self, name): ...
    def collection_exists(self, name): ...
```

### 添加新的重排序器

```python
# src/rag0/retrieval/reranker.py

class YourReranker(RerankerInterface):
    def rank(self, query, documents, top_k) -> list[ScoredDocument]:
        # 你的排序逻辑
        ...
```

## 异常处理策略

```
用户请求
  │
  ▼
API 路由 (chat.py / knowledge.py)
  │  try/except 由 middleware 统一处理
  ▼
Chain 层 (indexing/retrieval/generation)
  │  抛出领域异常: DocumentLoadError, RetrievalError, ...
  ▼
Connector 层 (llm/embedding/vector_store)
  │  抛出连接异常: LLMConnectionError, VectorStoreConnectionError, ...
  │
  ▼
api/middleware.py → HTTP 响应
  ValidationError    → 400
  DocumentError      → 400
  ConnectionError    → 503
  RetrievalError     → 500
  Rag0Error          → 500
```

## 配置加载优先级

```
低优先级 ←──────────────────────────→ 高优先级

  默认值         .env          config.yaml      环境变量
(config.py)    (可选)          (可选)        (RAG0_ 前缀)
     │            │               │               │
     └────────────┴───────────────┴───────────────┘
                         │
                         ▼
                  RagConfig 单例
```

## 技术依赖图谱

```
rag0
├── liteLLM ─────────── openai, anthropic, deepseek, ...
├── sentence-transformers ─── torch, huggingface-hub
├── pymilvus ────────── milvus (vector DB)
├── rank-bm25 ───────── (sparse retrieval)
├── SQLAlchemy 2.0 ──── sqlite3
├── Alembic ─────────── (migrations)
├── FastAPI ─────────── uvicorn, sse-starlette
├── structlog ───────── (structured logging)
├── pydantic-settings ─ pyyaml
├── PyMuPDF ─────────── (PDF parsing)
├── python-docx ─────── (DOCX parsing)
├── RapidOCR ────────── (image OCR)
├── wired-table-rec ─── (table recognition)
├── diskcache ───────── (optional: disk cache)
├── Langfuse ────────── (optional: telemetry)
└── RAGAS ───────────── (optional: evaluation)
```
