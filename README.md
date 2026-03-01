# Advance RAG

A production-grade, **text-first** Retrieval-Augmented Generation pipeline with LangGraph orchestration, FastAPI backend, optional MongoDB persistence, and a Next.js chat frontend.

**Repo:** [https://github.com/RaghavOG/advance-rag](https://github.com/RaghavOG/advance-rag) · **Author:** [Raghav Singla](https://github.com/RaghavOG)

---

## Architecture Overview

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  LangGraph  (graph/graph.py)                         │
│                                                      │
│  Layer 0 — normalize_user_prompt                     │
│  Layer 1 — detect_multi_query                        │
│                │                                     │
│         ┌──────┴────────────────────────┐            │
│         │ (per sub-query)               │            │
│         ▼                               │            │
│  Layer 2 — ambiguity_check              │            │
│           → clarification_node          │            │
│           → query_rewrite_expand        │ loop       │
│           → adaptive_top_k_decision     │ (multi-Q)  │
│           → retrieve_documents          │            │
│           → merge_retrieval_results     │            │
│                │                        │            │
│  Layer 3 — compress_context_node        │            │
│           → generate_answer_node        │            │
│           (failure nodes at each step)  │            │
│                │                        │            │
│         └──────┘                        │            │
│                                         │            │
│  Layer 4 — collect_sub_answers ─────────┘            │
│           → merge_final_answers                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Text-first, single embedding model | Keeps cost low; OCR/transcripts become text |
| Separate `text_index` / `image_index` / `audio_index` | Modality isolation without coupling |
| LangGraph for orchestration only | No multi-agent complexity; clean failure nodes |
| HyDE + multi-query rewriting | Significant recall improvement with modest extra tokens |
| Adaptive top-k | Factual queries stay cheap; explanatory queries get more context |
| `<<BEGIN/END COMPRESSED CONTEXT>>` tags | Reduces prompt-injection risk from retrieved content |
| MongoDB optional | Conversations persist when `MONGODB_URI` is set; in-memory fallback otherwise |
| `.env` from project root | Loaded via `python-dotenv` so config works regardless of CWD |

---

## Project Structure

```
advance-rag/
│
├── config/
│   └── settings.py          # Pydantic-settings — .env from project root, fail-fast validators
│
├── embeddings/
│   ├── base.py              # Abstract BaseEmbedder (LangChain Embeddings interface)
│   ├── openai.py            # OpenAI embeddings
│   ├── sentence_transformers.py  # Local SentenceTransformer embeddings
│   └── factory.py           # Singleton get_embedder() — only place embeddings are created
│
├── vectorstores/
│   └── factory.py           # Unified create_vectorstore() for Chroma / FAISS / Pinecone
│
├── ingestion/
│   ├── cleaning.py          # Text normalization
│   ├── loaders.py           # PDF, TXT, Markdown loaders (page/paragraph-wise)
│   ├── chunking.py          # Recursive character chunking
│   └── ingest.py            # ingest_document() / ingest_pdf() orchestrator
│
├── query/
│   ├── decompose.py         # Heuristic multi-query splitter + answer_user_prompt()
│   └── rewrite.py          # LLM query rewriting + HyDE document generation
│
├── retrieval/
│   └── retriever.py         # retrieve_text(): rewrites + HyDE + adaptive k + dedup
│
├── compression/
│   └── compressor.py       # LLM context compressor (tagged output)
│
├── generation/
│   └── answer.py            # Grounded, cited LLM answer generation
│
├── graph/
│   ├── state.py             # RAGState TypedDict — single state flowing through graph
│   ├── nodes.py             # All 15 LangGraph nodes
│   └── graph.py             # Graph wiring, conditional edges, compiled rag_graph
│
├── pipeline/
│   └── run.py               # answer_question(pdf_path, question) — delegates to graph
│
├── database/
│   ├── client.py            # MongoDB singleton; graceful fallback if URI missing
│   └── repository.py        # Conversation CRUD (save/get/delete/list)
│
├── backend/
│   ├── main.py              # FastAPI app — health dashboard, lifespan (LangSmith, MongoDB, graph warm-up)
│   ├── health.py            # Health checks: config, OpenAI, embedding, vector store, sample_docs, MongoDB, LangGraph, packages
│   ├── models.py            # Pydantic API request/response models
│   ├── store.py             # Conversation store (MongoDB-backed with in-memory fallback)
│   ├── pipeline_adapter.py  # LangGraph state → QueryResponse
│   └── routes/
│       └── query.py         # POST /api/query, POST /api/clarify, GET /api/conversation/{id}
│
├── frontend/                # Next.js chat UI (Framer Motion, structured RAG responses)
│
├── utils/
│   └── logger.py            # Structured logging (LOG_LEVEL, log_stage)
│
├── scripts/
│   ├── ingest_docs.py       # Ingest all files from sample_docs/ (or --docs-dir)
│   ├── run_demo.py         # Ingest + run demo questions through the graph
│   └── manual_test.py      # Ingest → retrieve → compress (no answer generation)
│
├── sample_docs/             # Default folder for ingestable documents (PDF, TXT, MD)
├── .env.example             # All environment variables documented
├── requirements.txt        # Python dependencies
├── Scripts.md               # One-line script reference
├── TODO.md                  # Feature status and backlog
└── README.md                # This file
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/RaghavOG/advance-rag.git
cd advance-rag
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY.
# For local dev, use VECTOR_STORE=chroma (no Pinecone index needed).
# Optional: MONGODB_URI for conversation persistence; LANGSMITH_* for tracing.
```

### 3. Add documents and ingest

Place PDF, TXT, or Markdown files in `sample_docs/`, then:

```bash
python scripts/ingest_docs.py
```

Or ingest a single file:

```bash
python scripts/ingest_docs.py --file sample_docs/rag_systems.md
```

### 4. Start the API and check health

```bash
uvicorn backend.main:app --reload --port 8000
```

- **Health dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — all services (config, OpenAI, embedding, vector store, sample_docs, MongoDB, LangGraph, packages)
- **JSON health:** [http://127.0.0.1:8000/health/json](http://127.0.0.1:8000/health/json)
- **API docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

On successful startup you’ll see: `✓ All services started successfully. API ready at http://127.0.0.1:8000`

### 5. Run a query (CLI or API)

**From Python:**

```python
from pipeline.run import answer_question

answer = answer_question("sample_docs/rag_systems.md", "What is HyDE?")
print(answer)
```

**From API:**

```bash
curl -X POST http://127.0.0.1:8000/api/query -H "Content-Type: application/json" -d "{\"prompt\": \"What is HyDE?\"}"
```

**Manual test (retrieve + compress only, no answer):**

```bash
python scripts/manual_test.py sample_docs/rag_systems.md "What is HyDE?"
```

More commands: see [Scripts.md](Scripts.md).

---

## API & Health

| Route | Description |
|-------|-------------|
| `GET /` | HTML health dashboard (all checks) |
| `GET /health` | JSON if `Accept: application/json`, else redirect to `/` |
| `GET /health/json` | Always JSON health report |
| `POST /api/query` | Run RAG pipeline; returns structured response (sub_answers, citations, clarification_required, etc.) |
| `POST /api/clarify` | Send clarification answer for a pending sub-query |
| `GET /api/conversation/{id}` | Get conversation state by ID |
| `GET /api/conversations` | List recent conversations (from MongoDB or in-memory) |
| `GET /docs` | Swagger UI |

Health checks: **config**, **openai_key**, **embedding**, **vector_store**, **docs_folder** (sample_docs/), **chroma_dir** (if Chroma), **mongodb**, **langgraph**, **packages**. If Pinecone returns 404, the dashboard explains that the index must be created in Pinecone Console (name, dimension, region).

---

## Configuration Reference

All settings live in `.env` (loaded from project root via `python-dotenv`). See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** when `EMBEDDING_PROVIDER=openai` |
| `VECTOR_STORE` | `chroma` | `chroma` \| `faiss` \| `pinecone` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `sentence-transformers` |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-large` | OpenAI embedding model (use matching `PINECONE_DIMENSION` if Pinecone) |
| `OPENAI_DEFAULT_MODEL` | `gpt-4.1-mini` | Default chat model |
| `ENABLE_HYDE` | `true` | Enable HyDE + query rewriting |
| `TOP_K_TEXT` | `5` | Retrieval top-k for text index |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.2` | Distance threshold for confidence gate |
| `GRAPH_MAX_RETRIES` | `2` | Max answer generation retries |
| `MAX_SUB_QUERIES` | `3` | Max independent questions per prompt |
| `MONGODB_URI` | — | Optional. When set, conversations persist; otherwise in-memory only |
| `MONGODB_DB_NAME` | `multimodal-rag` | Database name when using MongoDB |
| `LANGSMITH_TRACING` | `false` | Set `true` + `LANGSMITH_API_KEY` for LangSmith tracing |
| `LANGSMITH_PROJECT` | `advance-rag` | LangSmith project name |

**Pinecone:** If you use `VECTOR_STORE=pinecone`, create the index in [Pinecone Console](https://app.pinecone.io) with the same name as `PINECONE_INDEX_NAME`, dimension = `PINECONE_DIMENSION` (e.g. 3072 for `text-embedding-3-large`), metric = cosine, and region matching `PINECONE_ENVIRONMENT`. Otherwise the health check will report “Pinecone index not found (404)”.

**Documents:** Ingestable files live in `sample_docs/` by default (override with `--docs-dir` in scripts). Supported: `.pdf`, `.txt`, `.md`.

---

## LangGraph Node Reference

| # | Node | LLM | Description |
|---|------|-----|-------------|
| 1 | `normalize_user_prompt` | ❌ | Trim + normalize raw prompt |
| 2 | `detect_multi_query` | ❌ | Split prompt into independent sub-queries |
| 3 | `ambiguity_check` | ✅ cheap | Detect vague/underspecified questions |
| 4 | `clarification_node` | ❌ | Surface clarification question to caller |
| 5 | `query_rewrite_expand` | ✅ | Generate 2–4 retrieval-only rewrites |
| 6 | `adaptive_top_k_decision` | ❌ | Heuristic per-modality top-k |
| 7 | `retrieve_documents` | ❌ + embed | Vector search per rewrite + HyDE |
| 8 | `merge_retrieval_results` | ❌ | Dedup by `(doc_id, chunk_id)`, sort by score |
| 9 | `retrieval_failure_node` | ❌ | No-docs or low-confidence fallback |
| 10 | `compress_context_node` | ✅ | LLM-based context summarization |
| 11 | `compression_failure_node` | ❌ | Extractive fallback (first 400 chars/doc) |
| 12 | `generate_answer_node` | ✅ | Grounded answer with retry counter |
| 13 | `llm_timeout_failure_node` | ❌ | Exhausted retries fallback |
| 14 | `collect_sub_answers` | ❌ | Accumulate per-sub-query answers |
| 15 | `merge_final_answers` | ❌ | Format structured multi-Q output |

---

## Invariants (never break these)

1. **One question → one retrieval context** — sub-queries never share docs
2. **Multi-query ≠ query rewriting** — rewriting helps recall; splitting handles intent
3. **LLMs never silently control flow** — all routing is explicit in graph edges
4. **Failures are nodes, not exceptions** — every failure path has a user-facing message
5. **Clarification fires once, max** — `clarification_used` flag enforces this
6. **Embeddings created in exactly one place** — `embeddings/factory.py:get_embedder()`

---

## What's Not Built Yet

See [TODO.md](TODO.md) for the full backlog. Top priorities:

- **`llm/` retry + backoff layer** — no retry logic outside the graph counter yet
- **Ingestion idempotency** — re-running ingest duplicates chunks
- **Image ingestion** (`ingestion/image_loader.py`) — OCR via pytesseract → `image_index`
- **Audio ingestion** (`ingestion/audio_loader.py`) — transcription via faster-whisper → `audio_index`
- **Multimodal router** — classify query intent, conditionally add image/audio retrieval
- **Test suite** (`tests/`) — unit + integration tests
- **CLI** (`cli/`) — `ingest` and `query` commands

---

## Dependencies

**Core**

- `langchain`, `langchain-openai`, `langchain-community`, `langchain-core`
- `langgraph`
- `openai`
- `chromadb` / `faiss-cpu`
- `pypdf`
- `pydantic-settings`
- `python-dotenv` — load `.env` from project root
- `sentence-transformers` (for local embeddings)

**API & persistence**

- `fastapi`, `uvicorn`, `httpx`
- `pymongo[srv]` — MongoDB (conversation persistence)

**Optional**

- `pytesseract` + `Pillow` — image OCR
- `faster-whisper` — audio transcription
- `langsmith` — tracing
- Pinecone: `pinecone-client`, `langchain-pinecone` (when `VECTOR_STORE=pinecone`)

See `requirements.txt` for pinned versions.
