# Refinement TODOs

Issues identified during step 8-9 testing. Updated after refinements v0.5.0.

## ✅ Completed (v0.5.0)

### ✅ 1. Keyword retriever was too weak

- **Fixed in:** `services/retrieval/keyword.py`
- **Solution:**
  - Switched to `websearch_to_tsquery` (Google-like OR semantics)
  - Added `english` ts_config with stemming (configurable via `KEYWORD_TS_CONFIG`)
  - Made query mode configurable (`websearch` | `plain` | `phrase`)
- **Impact:** Keyword retriever now returns results for semantic queries

### ✅ 2. DOCX parser lost heading_path

- **Fixed in:** `services/ingestion/docx.py`
- **Solution:**
  - Section-based parsing (heading + following paragraphs → one ParsedPage)
  - Multi-signal heading detection (style, bold, font size, layout)
  - Attach heading to _following_ paragraphs as `heading_path`
- **Impact:** DOCX chunks now have proper heading paths

### ✅ 3. Reranker first-load latency (was 3 min)

- **Fixed in:** `services/reranking/local.py`, `api/app.py`, `.env`
- **Solution:**
  - `HF_TOKEN` support (via `settings.hf_token`)
  - Module-level `_MODEL_CACHE` shared across instances
  - Preload at app startup (`lifespan`)
- **Impact:** Startup cost paid once; subsequent queries fast

### ✅ 4. PDF page-level chunking was too coarse

- **Fixed in:** `services/ingestion/pdf.py`
- **Solution:**
  - Comprehensive multi-signal heading detection:
    - Font size (ratio to body size, weighted by char count)
    - Bold flag
    - Font family difference
    - ALL CAPS
    - Numbered prefixes ("1.", "1.1", "Chapter 3")
    - Layout (short, no ending punctuation)
  - Adaptive body size per document
  - Header/footer noise filtering
  - Page number filtering
  - Merge tiny sections
- **Impact:** PDF chunks are now section-based (not page-based)

### ✅ 6. RRF weights were equal

- **Fixed in:** `services/retrieval/hybrid.py`, `core/config.py`
- **Solution:** Configurable weights (`RRF_DENSE_WEIGHT=0.7`, `RRF_KEYWORD_WEIGHT=0.3`)
- **Impact:** Dense signal is prioritized; keyword acts as a booster

### ✅ 7. No query embedding cache

- **Fixed in:** `core/cache.py`, `core/embeddings.py`
- **Solution:**
  - Redis-backed cache (`RedisCache`) with in-memory fallback (`MemoryCache`)
  - Configurable backend: `memory` | `redis` | `none`
  - SHA-256 key namespacing (`embed:query:<model>:<digest>`)
  - TTL-based expiration
- **Impact:** Repeated queries hit cache; ~200-500ms → ~1-5ms

### ✅ 8. `retrieval_top_k` produced hardcoded 30-candidate pool

- **Fixed in:** `services/retrieval/hybrid.py`, `core/config.py`
- **Solution:** Configurable `RETRIEVAL_CANDIDATE_MULTIPLIER` and `RETRIEVAL_CANDIDATE_MIN`
- **Impact:** Tunable per deployment

### ✅ 9. Conversation history not sent to LLM

- **Fixed in:** `services/rag/engine.py`, `core/config.py`
- **Solution:**
  - `_load_history()` loads last N messages (configurable `RAG_MAX_HISTORY_MESSAGES`)
  - Token-budgeted (half of context budget for history)
  - Prepended as `{role, content}` before current user message
- **Impact:** Multi-turn conversations work naturally

### ✅ 10. Stream mode didn't report token usage

- **Fixed in:** `core/llm.py`, `services/rag/engine.py`
- **Solution:**
  - `stream_options={"include_usage": True}` in `litellm.acompletion()`
  - `LLMClient.last_usage` captures final usage chunk
  - Persisted to DB and included in SSE `done` event
- **Impact:** Stream mode now reports accurate token usage

### ✅ 11. Reranker scores were raw logits (could be negative)

- **Fixed in:** `services/reranking/local.py`
- **Solution:** Apply sigmoid: `score = 1 / (1 + exp(-logit))`
- **Impact:** All scores in [0, 1]; interpretable as probabilities

---

## ⏭️ Deferred (post-v0.5.0)

### ⏭️ 5. Reranker model is too small

- **Symptom:** Score spread is narrow
- **Why deferred:** Larger model = larger Docker image + slower cold start
- **Fix options:**
  - [ ] Try `BAAI/bge-reranker-base` (~278M)
  - [ ] Try `BAAI/bge-reranker-v2-m3` (568M)
  - [ ] Or Cohere Rerank v3.5 (Pro mode)
- **Files:** `services/reranking/local.py`, `.env`

### 🟡 12. PDF heading levels flat when all headings same size

- **Symptom:** PDFs where all headings use the same font size produce flat heading paths (e.g. "Supervised Learning" instead of "Introduction > Supervised Learning")
- **Why deferred:** Requires layout analysis (indentation, y-position) — non-trivial
- **Fix:**
  - [ ] Use font size + indentation to infer hierarchy
  - [ ] Use numbered prefix depth when present
  - [ ] Or: cluster headings by size and rank clusters
- **Files:** `services/ingestion/pdf.py`, `_infer_level()`

### 🟢 13. `pg_trgm` fuzzy matching for keyword retriever

- **Symptom:** Typos in queries don't match keyword search
- **Why deferred:** Adds another Postgres extension
- **Fix:**
  - [ ] `CREATE EXTENSION pg_trgm`
  - [ ] Add `similarity()` as a third retriever signal
- **Files:** `services/retrieval/keyword.py`, migration

### 🟢 14. `paradedb/pg_search` for real BM25

- **Symptom:** `ts_rank_cd` is not BM25; ranking could be better
- **Why deferred:** External dependency, changes DB setup
- **Fix:**
  - [ ] Switch to `paradedb/pg_search` Docker image
  - [ ] Use real BM25 scoring
- **Files:** `docker-compose.yml`, `services/retrieval/keyword.py`

### 🟢 15. Query embedding cache for `embed_texts` (batch)

- **Symptom:** Only query embeddings are cached; batch ingestion isn't (intentional)
- **Why deferred:** Ingestion is one-shot; caching adds complexity
- **Fix:**
  - [ ] Optional cache for chunks too (mostly for re-ingest)
- **Files:** `core/embeddings.py`

---

## 📋 Notes

- **v0.5.0-refinements** completed 11 fixes
- Deferred items are documented for future iterations
- No open critical issues
