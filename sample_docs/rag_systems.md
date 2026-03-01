# Retrieval-Augmented Generation (RAG) Systems

## Overview

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language model (LLM) generation. Instead of relying solely on knowledge encoded in the model's parameters during training, a RAG system retrieves relevant documents from an external knowledge source at inference time and conditions the LLM's response on that retrieved content.

RAG addresses the core limitations of standalone LLMs: knowledge cutoffs, hallucinations about specific facts, inability to cite sources, and lack of access to private or proprietary data.

## Core Pipeline

A basic RAG pipeline consists of the following stages:

**Ingestion**: Documents are loaded, cleaned, chunked into smaller segments, embedded using a vector embedding model, and stored in a vector database. Metadata (source, page, section) is stored alongside each chunk.

**Retrieval**: At query time, the user's question is embedded using the same model. The vector store is searched for semantically similar chunks using approximate nearest neighbor (ANN) search. The top-k most similar chunks are returned.

**Context Compression**: Retrieved chunks often contain noise. A compression step uses an LLM or extractive techniques to distill only the relevant portions, reducing the context window required for generation.

**Generation**: The compressed context and the original query are passed to an LLM, which generates a grounded answer. The system prompt instructs the LLM to use only the provided context and to cite its sources.

## Chunking Strategies

Chunking is one of the most impactful decisions in RAG system design. The chunk size governs the granularity of retrieval.

**Fixed-size chunking**: Split text every N characters with overlap. Simple and fast, but may split in the middle of sentences or concepts.

**Recursive character splitting**: Uses a hierarchy of separators (paragraph breaks, sentence breaks, word breaks) to create natural chunks. Preserves more semantic coherence than fixed-size.

**Semantic chunking**: Groups sentences with similar embeddings into the same chunk. Produces more coherent chunks but is computationally expensive.

**Parent-child chunking**: Index small child chunks for precision retrieval, but return their larger parent chunk to the LLM for more context. Balances retrieval precision with generation context richness.

## Query Enhancement Techniques

**Multi-query expansion**: The user's question is rewritten into multiple alternative phrasings by an LLM. Each rewrite is independently retrieved, and results are merged and deduplicated. This increases recall by capturing different vocabulary and framings of the same intent.

**HyDE (Hypothetical Document Embeddings)**: An LLM generates a hypothetical document that would answer the question. This hypothetical document is embedded and used as the retrieval query instead of the raw question. HyDE is particularly effective when the question phrasing is very different from the phrasing in the indexed documents.

**Step-back prompting**: The LLM is asked to first generate a more general background question, retrieve context for that, then proceed to answer the specific question. This improves reasoning on questions requiring background knowledge.

## Retrieval Quality Signals

Raw similarity scores from vector search are distance metrics (cosine, L2, dot product) and vary by backend and model. Lower distance does not always mean higher relevance. Key signals to monitor:

- **Retrieval recall**: Are the truly relevant documents in the top-k?
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of the first relevant document across queries.
- **NDCG**: Normalized Discounted Cumulative Gain — accounts for the rank of relevant documents.
- **Faithfulness**: Does the generated answer stay within the retrieved context?
- **Answer Relevance**: Does the generated answer address the question?

## Common Failure Modes

**Wrong chunks retrieved**: The embedding model maps the question and relevant document to different regions of the embedding space. Solutions: query rewriting, hybrid search (BM25 + dense), reranking.

**Chunk boundary truncation**: The answer spans two chunks, but only one is retrieved. Solutions: larger chunks, overlapping chunks, parent-child chunking.

**Context dilution**: Too many irrelevant chunks included in the prompt. Solutions: context compression, reranking, lower top-k.

**Hallucination on partial context**: The LLM fills gaps in incomplete context with plausible but incorrect facts. Solutions: stricter system prompts, answer validation passes, retrieval confidence gates.

**Ingestion idempotency failure**: Re-running ingestion on an already-indexed document creates duplicate chunks that dominate retrieval. Solutions: deterministic document IDs based on file hash, existence checks before indexing.

## Advanced Patterns

**Hybrid search**: Combines sparse retrieval (BM25, TF-IDF) with dense vector retrieval. Sparse methods excel at exact keyword matching; dense methods handle semantic similarity. Reciprocal Rank Fusion (RRF) is a common merging strategy.

**Reranking**: A cross-encoder model takes (query, document) pairs and produces a relevance score. More accurate than bi-encoder retrieval but computationally expensive. Applied as a post-retrieval step on the top-k candidates.

**Adaptive top-k**: Retrieval budget (k) adapts to query complexity. Factual single-word answers need top-3; broad analytical questions benefit from top-10 or more.

**Confidence thresholds**: Set a minimum similarity score threshold. If no retrieved document passes the threshold, surface a "no relevant information found" message rather than generating an answer from low-confidence context.

**Multi-hop retrieval**: Some questions require chaining: retrieve context for a sub-question, use that to form the next sub-question, retrieve again. LangGraph and agent frameworks are commonly used to implement this pattern.
