# docu-mind

> Personal document assistant with real RAG — upload PDF/Word/Markdown, ask questions, get cited answers from your own documents.

[![CI](https://github.com/amirgh-devtech/docu-mind/actions/workflows/ci.yml/badge.svg)](https://github.com/amirgh-devtech/docu-mind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.133+-009688)
![pgvector](https://img.shields.io/badge/pgvector-%3E%3D0.8-336791)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 What it does

`docu-mind` is a self-hosted RAG service that:

- **Ingests** PDF, Word, and Markdown documents
- **Chunks** them with **Contextual Retrieval** (LLM-generated context per chunk)
- **Embeds** and stores in **PostgreSQL + pgvector**
- **Retrieves** using **Hybrid Search** (dense + keyword) with **RRF fusion**
- **Reranks** results with a cross-encoder
- **Answers** questions with **streaming responses** and **citations**

## 🧱 Tech stack

| Layer            | Technology                                        |
| ---------------- | ------------------------------------------------- |
| API              | FastAPI + Uvicorn                                 |
| LLM abstraction  | LiteLLM (OpenAI / Anthropic / Gemini / Groq)      |
| Embedding        | OpenAI text-embedding-3 (small/large)             |
| Reranker         | local cross-encoder (Eco) / Cohere (Pro)          |
| Storage          | PostgreSQL + pgvector (HNSW)                      |
| Keyword search   | PostgreSQL tsvector (GIN)                         |
| Document parsing | pymupdf (PDF), python-docx (Word), markdown-it-py |
| Container        | Docker + Docker Compose                           |
| Package mgmt     | uv                                                |
| Task runner      | just                                              |
| Evaluation       | ragas                                             |

## 🎛️ Operating modes

### Eco mode (default) — lowest cost

| Component | Choice                               | Cost                        |
| --------- | ------------------------------------ | --------------------------- |
| LLM       | Gemini 3.5 Flash-Lite                | $0.30 / $2.50 per 1M in/out |
| Embedding | OpenAI text-embedding-3-small (1536) | $0.02 / 1M                  |
| Reranker  | local ms-marco-MiniLM-L6-v2          | free                        |

### Pro mode — highest accuracy

| Component | Choice                                                         | Cost            |
| --------- | -------------------------------------------------------------- | --------------- |
| LLM       | Gemini 3.6 Flash + fallback to GPT-5.4 mini / Claude Haiku 4.5 | higher          |
| Embedding | OpenAI text-embedding-3-large (truncated to 1024)              | $0.13 / 1M      |
| Reranker  | Cohere Rerank 4 Pro                                            | $2 / 1K queries |

Switch mode via `DOCU_MIND_MODE=eco|pro` in `.env`.

## 🚧 Technical challenges addressed

- **Contextual Retrieval** — each chunk gets an LLM-generated summary prefix
- **Hybrid Search** — dense (pgvector) + sparse (tsvector) fused with RRF
- **Citation** — every answer includes source document + page/section
- **Context window management** — token budget per LLM call, truncation strategy
- **Evaluation** — ragas metrics: faithfulness, answer correctness, context precision

## 📁 Project layout

```
docu-mind/
├── src/docu_mind/          # application code
│   ├── api/                # FastAPI routes
│   ├── core/               # config, logging, llm abstraction
│   ├── db/                 # SQLAlchemy models & session
│   ├── schemas/            # Pydantic DTOs
│   ├── services/           # ingestion, retrieval, rag
│   └── workers/            # background jobs (future)
├── migrations/             # Alembic
├── tests/                  # unit + integration + eval
├── scripts/                # dev helpers
├── Dockerfile
├── docker-compose.yml
└── justfile
```

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/amirgh-devtech/docu-mind.git
cd docu-mind

# 2. Setup env
cp .env.example .env
# edit .env with your API keys

# 3. Start infra (Postgres + pgvector)
just up-infra

# 4. Install deps
just sync

# 5. Run migrations
just migrate

# 6. Run API
just api
```

## 📜 License

MIT
