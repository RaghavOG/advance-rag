# Multimodal RAG Pipeline

A production-grade, **text-first** Retrieval-Augmented Generation pipeline with LangGraph orchestration.  

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

---

## Project Structure

```
multimodel-rag/
│
├── config/
│   └── settings.py          # Pydantic-settings — all env vars, fail-fast validators
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
│   ├── loaders.py           # PDF loader (pypdf, page-wise)
│   ├── chunking.py          # Recursive character chunking
│   └── ingest.py            # ingest_pdf() orchestrator
│
├── query/
│   ├── decompose.py         # Heuristic multi-query splitter + answer_user_prompt()
│   └── rewrite.py           # LLM query rewriting + HyDE document generation
│
├── retrieval/
│   └── retriever.py         # retrieve_text(): rewrites + HyDE + adaptive k + dedup
│
├── compression/
│   └── compressor.py        # LLM context compressor (tagged output)
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
├── scripts/
│   └── manual_test.py       # CLI: ingest → retrieve → compress (no answer generation)
│
├── .env.example             # All environment variables documented
├── requirements.txt         # Python dependencies
├── TODO.md                  # Feature status and backlog
└── README.md                # This file
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd multimodel-rag
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Run a query

```python
from pipeline.run import answer_question

answer = answer_question("path/to/your.pdf", "What is the main contribution?")
print(answer)
```

### 4. Manual validation (no answer generation)

```bash
python scripts/manual_test.py path/to/your.pdf "Your question here"
```

---

## Configuration Reference

All settings live in `.env` (see `.env.example` for the full list).  
Critical variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `VECTOR_STORE` | `chroma` | `chroma` \| `faiss` \| `pinecone` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `sentence-transformers` |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_DEFAULT_MODEL` | `gpt-4.1-mini` | Default chat model |
| `ENABLE_HYDE` | `true` | Enable HyDE + query rewriting |
| `TOP_K_TEXT` | `5` | Retrieval top-k for text index |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.2` | Distance threshold for confidence gate |
| `GRAPH_MAX_RETRIES` | `2` | Max answer generation retries |
| `MAX_SUB_QUERIES` | `3` | Max independent questions per prompt |

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

See [`TODO.md`](TODO.md) for the full backlog. Top priorities:

- **`llm/` retry + backoff layer** — no retry logic outside the graph counter yet
- **Ingestion idempotency** — re-running `ingest_pdf` duplicates chunks
- **Image ingestion** (`ingestion/image_loader.py`) — OCR via pytesseract → `image_index`
- **Audio ingestion** (`ingestion/audio_loader.py`) — transcription via faster-whisper → `audio_index`
- **Multimodal router** — classify query intent, conditionally add image/audio retrieval
- **Test suite** (`tests/`) — unit + integration tests
- **CLI** (`cli/`) — `ingest` and `query` commands

---

## Dependencies

Core:
- `langchain`, `langchain-openai`, `langchain-community`, `langchain-core`
- `langgraph`
- `openai`
- `chromadb` / `faiss-cpu`
- `pypdf`
- `pydantic-settings`
- `sentence-transformers` (for local embeddings)

Optional (for future modalities):
- `pytesseract` + `Pillow` — image OCR
- `faster-whisper` — audio transcription

See `requirements.txt` for pinned versions.
