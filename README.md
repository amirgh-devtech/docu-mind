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
| Embedding        | Gemini Embedding 001 (1536 dim, Matryoshka)       |
| Reranker         | local cross-encoder (Eco) / Cohere Rerank (Pro)   |
| Storage          | PostgreSQL 17 + pgvector 0.8 (HNSW)               |
| Keyword search   | PostgreSQL tsvector (GIN)                         |
| Document parsing | pymupdf (PDF), python-docx (Word), markdown-it-py |
| Container        | Docker + Docker Compose                           |
| Package mgmt     | uv                                                |
| Task runner      | just                                              |
| Evaluation       | ragas                                             |

## 🎛️ Operating modes

Switch mode via `DOCU_MIND_MODE=eco|pro` in `.env`.

### Eco mode (default) — lowest cost

| Component | Choice                          | Cost         |
| --------- | ------------------------------- | ------------ |
| LLM       | Gemini 3.5 Flash-Lite           | free tier    |
| Embedding | Gemini Embedding 001 (1536 dim) | free tier    |
| Reranker  | local cross-encoder (MiniLM-L6) | free (local) |

### Pro mode — highest accuracy

| Component | Choice                                                         | Cost            |
| --------- | -------------------------------------------------------------- | --------------- |
| LLM       | Gemini 3.6 Flash + fallback to GPT-5.4 mini / Claude Haiku 4.5 | pay-per-use     |
| Embedding | Gemini Embedding 001 (1536 dim)                                | pay-per-use     |
| Reranker  | Cohere Rerank v3.5                                             | $2 / 1K queries |

## 🚧 Technical challenges addressed

- **Contextual Retrieval** — each chunk gets an LLM-generated context prefix before embedding
- **Parent-Child chunking** — small child chunks are embedded; larger parents are returned for context
- **Hybrid Search** — dense (pgvector HNSW) + sparse (tsvector GIN) fused with Reciprocal Rank Fusion
- **Citation** — every answer includes source document + page/section + snippet
- **Context window management** — token budget per LLM call, truncation strategy
- **Evaluation** — ragas metrics: faithfulness, answer correctness, context precision

## 📁 Project layout

```
docu-mind/
├── src/docu_mind/ # application code
│ ├── api/ # FastAPI routes
│ ├── core/ # config, logging, LLM, embeddings
│ ├── db/ # SQLAlchemy models & session
│ ├── schemas/ # Pydantic DTOs
│ ├── services/ # ingestion, chunking, retrieval, rag
│ └── utils/
├── migrations/ # Alembic
├── tests/ # unit + integration + e2e + eval
├── scripts/ # dev helpers
├── Dockerfile
├── docker-compose.yml
└── justfile
```

## 🚀 Quick start

### Prerequisites

- **Python 3.12** — managed automatically by `uv`
- **Docker Desktop** — for PostgreSQL + pgvector
- **uv** — install: `winget install astral-sh.uv`
- **just** — install: `winget install Casey.Just`
- **GEMINI_API_KEY** — free from https://aistudio.google.com/app/apikey

### Setup

```bash
# 1. Clone
git clone https://github.com/amirgh-devtech/docu-mind.git
cd docu-mind

# 2. Setup env — copy and fill in GEMINI_API_KEY
cp .env.example .env

# Windows PowerShell:
# Copy-Item .env.example .env

# 3. Install dependencies
just sync

# 4. Start Postgres + pgvector
just up-pg

# 5. Run migrations (creates tables + pgvector extension)
just migrate

# 6. Run the API (with hot reload)
just api
```

Open Swagger UI at: http://localhost:8000/docs

### Quick verification

```bash
# Liveness
curl http://localhost:8000/api/v1/health

# → {"status":"ok","version":"0.1.0"}

# Readiness (checks DB + pgvector)
curl http://localhost:8000/api/v1/ready

# → {"status":"ready","version":"0.1.0","pgvector":"0.8.6"}

# Runtime config snapshot
curl http://localhost:8000/api/v1/config
```

### Everyday commands

```bash
just                          # list all recipes
just up-pg                    # start Postgres
just down-pg                  # stop Postgres
just migrate                  # apply migrations
just makemigration "add tags" # create a new migration
just api                      # run API (hot reload)
just lint                     # ruff + mypy
just fmt                      # auto-fix lint + format
just test                     # unit tests
just test-all                 # all tests
just psql                     # open psql shell
just clean                    # clean caches
```

### Stopping everything

```bash
# Ctrl+C in the API terminal, then:
just down-pg
```

## ⚙️ Configuration

All settings are loaded from `.env`. See `.env.example` for the full list.

Key variables:

| Variable              | Default               | Description                      |
| --------------------- | --------------------- | -------------------------------- |
| `DOCU_MIND_MODE`      | `eco`                 | `eco` (free) or `pro` (accurate) |
| `GEMINI_API_KEY`      | —                     | Required for LLM + embeddings    |
| `OPENAI_API_KEY`      | —                     | Optional (Pro mode fallback)     |
| `ANTHROPIC_API_KEY`   | —                     | Optional (Pro mode fallback)     |
| `COHERE_API_KEY`      | —                     | Required only in Pro mode        |
| `POSTGRES_*`          | localhost / docu_mind | Database connection              |
| `CHUNK_SIZE`          | `800`                 | Target chunk size (tokens)       |
| `CHUNK_OVERLAP`       | `150`                 | Chunk overlap (tokens)           |
| `CONTEXTUAL_CHUNKING` | `true`                | Enable Contextual Retrieval      |
| `RETRIEVAL_TOP_K`     | `20`                  | Candidates fetched per retriever |
| `RERANK_TOP_K`        | `5`                   | Final results after reranking    |

## 🗺️ Roadmap

- [x] Project skeleton + CI
- [x] Core (config, logging, LLM abstraction, embeddings)
- [x] Database layer + Alembic + pgvector
- [x] FastAPI app + routes + schemas
- [x] Ingestion pipeline (PDF/DOCX/MD → chunks → embeddings)
- [x] Retrieval (hybrid search + RRF + reranker)
- [x] RAG engine (prompt + citations + streaming)
- [ ] Docker Compose for full stack
- [ ] Tests (unit + integration + ragas evaluation)
- [ ] Deployment guide

## 🤝 Contributing

This is a personal portfolio project — PRs are welcome but may not be merged. Feel free to fork.

## 📜 License

MIT — see [LICENSE](LICENSE).
