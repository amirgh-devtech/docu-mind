# docu-mind

> Personal document assistant with real RAG — upload PDF/Word/Markdown, ask questions, get cited answers from your own documents.

[![CI](https://github.com/amirgh-devtech/docu-mind/actions/workflows/ci.yml/badge.svg)](https://github.com/amirgh-devtech/docu-mind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.133+-009688)
![pgvector](https://img.shields.io/badge/pgvector-%3E%3D0.8-336791)
![Redis](https://img.shields.io/badge/Redis-7.x-DC382D)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 What it does

`docu-mind` is a self-hosted RAG service that:

- **Ingests** PDF, Word, and Markdown documents with format-aware heading detection
- **Chunks** them with **Contextual Retrieval** (LLM-generated context prefix per chunk) and **Parent-Child** hierarchy
- **Embeds** and stores in **PostgreSQL + pgvector** (HNSW index)
- **Retrieves** using **Hybrid Search** (dense + keyword) fused with **weighted RRF**
- **Reranks** with a cross-encoder (sigmoid-normalized scores)
- **Answers** questions with **streaming responses**, **citations**, and **multi-turn history**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  FastAPI + Uvicorn                              │
├─────────────────────────────────────────────────┤
│  Ingestion:  PDF (multi-signal) | DOCX | MD     │
│       ↓                                         │
│  Chunking:   Recursive + Parent/Child           │
│       ↓                                         │
│  Contextual: LLM-generated prefix per chunk     │
│       ↓                                         │
│  Embedding:  Gemini 001 (1536-dim, Matryoshka)  │
│       ↓                                         │
│  Storage:    PostgreSQL + pgvector (HNSW)       │
│              + tsvector (GIN)                   │
├─────────────────────────────────────────────────┤
│  Retrieval:  Dense + Keyword + weighted RRF     │
│       ↓                                         │
│  Reranking:  CrossEncoder (sigmoid-normalized)  │
│       ↓                                         │
│  Expansion:  Parent-child swap                  │
│       ↓                                         │
│  RAG:        Context budget + LLM + Citations   │
│       ↓                                         │
│  Streaming:  SSE with token usage               │
├─────────────────────────────────────────────────┤
│  Cache:      Redis (query embeddings)           │
│  History:    Multi-turn conversations           │
│  Config:     env-driven (eco/pro modes)         │
└─────────────────────────────────────────────────┘
```

## ✨ Highlights

- **Contextual Retrieval** — LLM-generated context prefix per chunk (Anthropic technique, cuts retrieval failure by 35-49%)
- **Parent-Child Chunking** — children embedded for search, parents returned for context
- **Hybrid Search** — Dense (pgvector HNSW) + Keyword (Postgres tsvector) fused with **weighted Reciprocal Rank Fusion**
- **Multi-format Parsing** — PDF (font-size + bold + layout heuristics), DOCX (style detection), Markdown (heading hierarchy) — all normalized to a common schema
- **Redis Query Cache** — SHA-256 keyed, TTL-based, shared across workers
- **Streaming RAG** — Server-Sent Events with citations, token usage, and latency metrics
- **Multi-turn Chat** — conversation history with token budgeting
- **Provider Abstraction** — Gemini/OpenAI embeddings, LiteLLM for LLM, local/Cohere rerankers
- **Production-shaped** — retries with backoff, graceful degradation, structured logging, Prometheus-ready

## 🧱 Tech stack

| Layer            | Technology                                        |
| ---------------- | ------------------------------------------------- |
| API              | FastAPI + Uvicorn                                 |
| LLM abstraction  | LiteLLM (Gemini / OpenAI / Anthropic / Groq)      |
| Embedding        | Gemini Embedding 001 (1536 dim, Matryoshka)       |
| Reranker         | local CrossEncoder (Eco) / Cohere Rerank (Pro)    |
| Storage          | PostgreSQL 17 + pgvector 0.8 (HNSW)               |
| Keyword search   | PostgreSQL tsvector + GIN (websearch_to_tsquery)  |
| Cache            | Redis 7 (query embeddings) + in-memory fallback   |
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
| Reranker  | local CrossEncoder (MiniLM-L6)  | free (local) |

### Pro mode — highest accuracy

| Component | Choice                                                         | Cost            |
| --------- | -------------------------------------------------------------- | --------------- |
| LLM       | Gemini 3.6 Flash + fallback to GPT-5.4 mini / Claude Haiku 4.5 | pay-per-use     |
| Embedding | Gemini Embedding 001 (1536 dim)                                | pay-per-use     |
| Reranker  | Cohere Rerank v3.5                                             | $2 / 1K queries |

## 🚧 Technical challenges addressed

- **Contextual Retrieval** — each chunk gets an LLM-generated context prefix before embedding
- **Parent-Child chunking** — small child chunks are embedded; larger parents are returned for context
- **Hybrid Search** — dense (pgvector HNSW) + sparse (tsvector GIN) fused with **weighted RRF** (dense=0.7, keyword=0.3)
- **Multi-signal heading detection** — font size, bold, font family, ALL CAPS, numbered prefixes, layout
- **Citation** — every answer includes source document + page/section + snippet + relevance score
- **Context window management** — token budget per LLM call, truncation strategy
- **Idempotent ingestion** — SHA-256 content hash dedup
- **Evaluation** — ragas metrics: faithfulness, answer correctness, context precision

## 📁 Project layout

```
docu-mind/
├── src/docu_mind/          # application code
│   ├── api/                # FastAPI app + v1 routes
│   ├── core/               # config, logging, LLM, embeddings, cache
│   ├── db/                 # SQLAlchemy models & session
│   ├── schemas/            # Pydantic DTOs
│   ├── services/           # ingestion, chunking, retrieval, rag
│   └── utils/
├── migrations/             # Alembic
├── tests/                  # unit + integration + e2e + eval
├── scripts/                # dev helpers
├── Dockerfile
├── docker-compose.yml
└── justfile
```

## 🚀 Quick start

### Prerequisites

- **Python 3.12** — managed automatically by `uv`
- **Docker Desktop** — for PostgreSQL + pgvector + Redis
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
#   Copy-Item .env.example .env

# 3. Install dependencies
just sync

# 4. Start Postgres + pgvector + Redis
just up-infra

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
# → {"status":"ok","version":"1.0.0"}

# Readiness (checks DB + pgvector)
curl http://localhost:8000/api/v1/ready
# → {"status":"ready","version":"1.0.0","pgvector":"0.8.6"}

# Runtime config snapshot
curl http://localhost:8000/api/v1/config
```

### Upload and query

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@samples/example.pdf"

# Search (retrieval only)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is supervised learning?", "rerank": true}'

# Chat (RAG with citations)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is supervised learning?"}'

# Chat with streaming (SSE)
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is supervised learning?"}'
```

### Everyday commands

```bash
just                          # list all recipes
just up-infra                 # start Postgres + Redis
just down-infra               # stop Postgres + Redis
just up-pg                    # start Postgres only
just up-redis                 # start Redis only
just redis-cli                # open redis-cli
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
just down-infra
```

## ⚙️ Configuration

All settings are loaded from `.env`. See `.env.example` for the full list.

### Key variables

| Variable                         | Default                    | Description                        |
| -------------------------------- | -------------------------- | ---------------------------------- |
| `DOCU_MIND_MODE`                 | `eco`                      | `eco` (free) or `pro` (accurate)   |
| `GEMINI_API_KEY`                 | —                          | Required for LLM + embeddings      |
| `OPENAI_API_KEY`                 | —                          | Optional (Pro mode fallback)       |
| `ANTHROPIC_API_KEY`              | —                          | Optional (Pro mode fallback)       |
| `COHERE_API_KEY`                 | —                          | Required only in Pro mode          |
| `EMBEDDING_PROVIDER`             | `gemini`                   | `gemini` or `openai`               |
| `POSTGRES_*`                     | localhost / docu_mind      | Database connection                |
| `REDIS_URL`                      | `redis://localhost:6379/0` | Cache backend                      |
| `EMBEDDING_CACHE_BACKEND`        | `redis`                    | `redis` \| `memory` \| `none`      |
| `CHUNK_SIZE`                     | `800`                      | Target chunk size (chars)          |
| `CHUNK_OVERLAP`                  | `150`                      | Chunk overlap (chars)              |
| `CONTEXTUAL_CHUNKING`            | `true`                     | Enable Contextual Retrieval        |
| `RETRIEVAL_TOP_K`                | `20`                       | Candidates fetched per retriever   |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | `2`                        | Candidate pool multiplier          |
| `RERANK_TOP_K`                   | `5`                        | Final results after reranking      |
| `RRF_DENSE_WEIGHT`               | `0.7`                      | Dense retriever weight in RRF      |
| `RRF_KEYWORD_WEIGHT`             | `0.3`                      | Keyword retriever weight in RRF    |
| `KEYWORD_TS_CONFIG`              | `english`                  | Postgres text search config        |
| `KEYWORD_QUERY_MODE`             | `websearch`                | `websearch` \| `plain` \| `phrase` |
| `RAG_CONTEXT_TOKEN_BUDGET`       | `6000`                     | Max tokens for retrieved context   |
| `RAG_MAX_HISTORY_MESSAGES`       | `10`                       | Conversation history depth         |

## 🗺️ Roadmap

### ✅ v1.0.0 — Feature complete

- [x] Project skeleton + CI
- [x] Core (config, logging, LLM abstraction, embeddings)
- [x] Database layer + Alembic + pgvector (HNSW)
- [x] FastAPI app + routes + schemas
- [x] Ingestion pipeline (PDF/DOCX/MD → chunks → embeddings)
- [x] Multi-signal heading detection (PDF, DOCX)
- [x] Contextual Retrieval + Parent-Child chunking
- [x] Hybrid Search (dense + keyword) + weighted RRF
- [x] CrossEncoder reranking with sigmoid normalization
- [x] RAG engine (prompt + citations + streaming SSE)
- [x] Multi-turn conversation history
- [x] Redis-backed query embedding cache
- [x] Stream token usage reporting
- [x] Configurable candidate pool + RRF weights

### 🔜 Future work

- [ ] Docker Compose for full stack
- [ ] Tests (unit + integration + ragas evaluation)
- [ ] Deployment guide
- [ ] Larger reranker models (BGE-reranker-base, v2-m3)
- [ ] PDF heading hierarchy via layout analysis
- [ ] `pg_trgm` fuzzy matching for keyword retriever
- [ ] `paradedb/pg_search` for real BM25

## 🤝 Contributing

This is a personal portfolio project — PRs are welcome but may not be merged. Feel free to fork.

## 📜 License

MIT — see [LICENSE](LICENSE).
