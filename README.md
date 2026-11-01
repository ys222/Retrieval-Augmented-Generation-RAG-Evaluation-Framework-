# RAG Evaluation Framework

A benchmarking pipeline that compares **sparse (BM25)**, **dense (E5)**, **hybrid**, and **graph-based (GraphRAG on Neo4j)** retrieval strategies for Retrieval-Augmented Generation (RAG) systems on the **Natural Questions** dataset. For every strategy it generates LLM answers, scores them with **RAGAS** (faithfulness and relevance), measures query latency, and reports the quality-versus-latency trade-off in a single comparison table.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Retrieval Strategies](#retrieval-strategies)
- [Evaluation Methodology](#evaluation-methodology)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Choosing a retrieval strategy is one of the most consequential decisions in a RAG system, yet the trade-offs between lexical, semantic, hybrid, and graph-based retrieval are rarely measured side by side under identical conditions. This project provides a reproducible harness that:

1. Indexes the same corpus once per retrieval backend.
2. Runs the same set of Natural Questions queries through every strategy.
3. Feeds the retrieved context to the same LLM with the same prompt.
4. Scores the resulting answers with RAGAS.
5. Records per-query latency.
6. Aggregates everything into a comparison table.

Because the corpus, queries, generator model, and prompt are held constant, differences in the final metrics are attributable to retrieval alone.

## Features

- **Four retrieval strategies** behind a common interface: BM25, dense (E5), hybrid, and GraphRAG (Neo4j).
- **Framework-agnostic orchestration** using LangChain and/or LlamaIndex.
- **Pluggable vector stores**: FAISS (local), Weaviate, or Pinecone.
- **LLM answer generation** through the OpenAI or Anthropic API.
- **Automated RAGAS scoring** on faithfulness and answer relevance.
- **Latency benchmarking** (mean, median, p95) per strategy.
- **Comparison table output** in Markdown and CSV for easy reporting.
- **Config-driven runs** so experiments are reproducible.

## Architecture

```
                         ┌──────────────────────────┐
                         │  Natural Questions (NQ)  │
                         │  corpus + eval queries   │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
        ┌──────────┐           ┌──────────────┐        ┌─────────────┐
        │   BM25   │           │  Dense (E5)  │        │  Graph build │
        │  index   │           │ FAISS/Weav./ │        │  (entities & │
        │          │           │  Pinecone    │        │  relations)  │
        └────┬─────┘           └──────┬───────┘        └──────┬──────┘
             │                        │                       │
             │                ┌───────┴───────┐               ▼
             │                │    Hybrid     │         ┌──────────┐
             │                │ (BM25 + dense │         │  Neo4j   │
             │                │    fusion)    │         │ GraphRAG │
             │                └───────┬───────┘         └────┬─────┘
             └────────────┬───────────┴──────────────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Retrieved context│
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  LLM generator  │  (OpenAI / Anthropic)
                 └────────┬────────┘
                          ▼
              ┌───────────────────────┐
              │   RAGAS evaluation    │  faithfulness, relevance
              │   + latency timing    │
              └───────────┬───────────┘
                          ▼
                 ┌─────────────────┐
                 │ Comparison table │
                 └─────────────────┘
```

## Retrieval Strategies

| Strategy | Signal | Index / Store | Strengths | Typical weaknesses |
|---|---|---|---|---|
| **BM25** | Lexical term overlap | In-memory inverted index | Fast, no embedding cost, strong on exact-term queries | Misses paraphrases and semantic matches |
| **Dense (E5)** | Semantic similarity of embeddings | FAISS / Weaviate / Pinecone | Handles paraphrase and intent well | Can miss rare entities and exact identifiers |
| **Hybrid** | BM25 + dense score fusion | Both indexes | Combines lexical precision with semantic recall | Higher latency and tuning of fusion weights |
| **GraphRAG** | Entity and relation traversal | Neo4j | Multi-hop and relational questions | Costly graph construction; added query complexity |

### Hybrid fusion

Hybrid retrieval merges the ranked lists from BM25 and dense search using either **Reciprocal Rank Fusion (RRF)** or a **weighted score combination**. For RRF the fused score of document *d* is:

```
RRF(d) = Σ_r  1 / (k + rank_r(d))
```

where `rank_r(d)` is the rank of *d* in retriever *r* and `k` is a smoothing constant (commonly 60).

### GraphRAG

Documents are processed to extract entities and relationships, which are stored in Neo4j as nodes and edges linked back to their source passages. At query time, entities are identified in the question, the neighborhood around them is traversed, and the connected passages are returned as context.

## Evaluation Methodology

For each query, and for each retrieval strategy:

1. **Retrieve** the top-*k* passages.
2. **Generate** an answer with the same LLM, prompt template, and decoding parameters.
3. **Time** the retrieval step (and, optionally, the end-to-end pipeline).
4. **Score** the answer with RAGAS.

### Metrics

| Metric | Source | What it measures |
|---|---|---|
| **Faithfulness** | RAGAS | Whether claims in the answer are supported by the retrieved context (hallucination control) |
| **Answer relevance** | RAGAS | Whether the answer directly addresses the question |
| **Query latency** | Wall-clock timer | Mean, median, and p95 retrieval time per query |

### Fairness controls

- Identical corpus, chunking, and evaluation queries across all strategies.
- Identical generator model, prompt, temperature, and max tokens.
- Fixed random seeds where applicable.
- Warm-up queries excluded from latency measurements.

## Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain, LlamaIndex
- **Sparse retrieval:** BM25 (`rank_bm25`)
- **Dense retrieval:** `sentence-transformers` with E5 embeddings
- **Vector stores:** FAISS, Weaviate, Pinecone
- **Graph store:** Neo4j
- **Evaluation:** RAGAS
- **LLM providers:** OpenAI API, Anthropic API
- **Dataset:** Natural Questions

## Project Structure

```
rag-evaluation-framework/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   └── default.yaml          # experiment configuration
├── data/
│   ├── raw/                  # downloaded Natural Questions data
│   └── processed/            # chunked corpus and eval queries
├── src/
│   ├── data/
│   │   └── natural_questions.py   # dataset loading and chunking
│   ├── retrievers/
│   │   ├── base.py                # common Retriever interface
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── graph_rag.py
│   ├── generation/
│   │   └── generator.py           # LLM answer generation
│   ├── evaluation/
│   │   ├── ragas_eval.py          # RAGAS scoring
│   │   └── latency.py             # latency measurement
│   ├── reporting/
│   │   └── table.py               # comparison table builder
│   └── pipeline.py                # end-to-end orchestration
├── scripts/
│   ├── build_indexes.py
│   ├── build_graph.py
│   └── run_benchmark.py
├── results/                  # generated outputs
└── tests/
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- An OpenAI and/or Anthropic API key
- A running Neo4j instance (for GraphRAG)
- Optional: a Weaviate or Pinecone account/instance (FAISS works locally with no setup)

### Installation

```bash
git clone https://github.com/<your-username>/rag-evaluation-framework.git
cd rag-evaluation-framework

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

```dotenv
# LLM providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Neo4j (GraphRAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Vector store (only if not using FAISS)
PINECONE_API_KEY=your_pinecone_key
WEAVIATE_URL=http://localhost:8080
```

### Start Neo4j (Docker)

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

## Usage

### 1. Prepare data and build indexes

```bash
python scripts/build_indexes.py --config configs/default.yaml
```

This downloads and chunks the Natural Questions corpus, then builds the BM25 index and the dense (E5) vector index.

### 2. Build the knowledge graph (GraphRAG only)

```bash
python scripts/build_graph.py --config configs/default.yaml
```

### 3. Run the benchmark

```bash
python scripts/run_benchmark.py \
  --config configs/default.yaml \
  --strategies bm25 dense hybrid graph \
  --num-queries 200
```

### 4. View the results

Outputs are written to `results/`:

- `results/comparison.md` – the comparison table
- `results/comparison.csv` – the same table in CSV form
- `results/per_query.jsonl` – per-query retrieved context, generated answer, scores, and latency

## Configuration

All experiment settings live in `configs/default.yaml`:

```yaml
dataset:
  name: natural_questions
  num_queries: 200
  chunk_size: 512
  chunk_overlap: 64

retrieval:
  top_k: 5
  dense:
    model: intfloat/e5-base-v2
    store: faiss            # faiss | weaviate | pinecone
  hybrid:
    fusion: rrf             # rrf | weighted
    rrf_k: 60
    alpha: 0.5              # used when fusion = weighted
  graph:
    max_hops: 2

generation:
  provider: openai          # openai | anthropic
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 256

evaluation:
  metrics: [faithfulness, answer_relevancy]
  warmup_queries: 5
```

## Results

Running the benchmark produces a comparison table of the following form (values are generated by your run and depend on corpus size, models, and configuration):

| Strategy | Faithfulness | Answer Relevance | Mean Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|
| BM25 | *generated* | *generated* | *generated* | *generated* |
| Dense (E5) | *generated* | *generated* | *generated* | *generated* |
| Hybrid | *generated* | *generated* | *generated* | *generated* |
| GraphRAG | *generated* | *generated* | *generated* | *generated* |

### How to read the table

- **Quality vs. latency:** a strategy that gains a few points of faithfulness at several times the latency may not be the right production choice.
- **Faithfulness gaps** typically point to retrieval returning off-topic or incomplete context.
- **Relevance gaps** typically point to retrieval missing the passage that actually answers the question.
- **GraphRAG** tends to pay off on multi-hop, entity-centric questions; on simple factoid lookups its extra cost often is not justified.

## Extending the Framework

**Add a new retriever.** Implement the `Retriever` interface in `src/retrievers/base.py`, then register it in the strategy registry used by `pipeline.py`:

```python
from src.retrievers.base import Retriever, RetrievedChunk

class MyRetriever(Retriever):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        ...
```

**Swap the vector store.** Change `retrieval.dense.store` in the config to `faiss`, `weaviate`, or `pinecone`.

**Swap the generator.** Change `generation.provider` and `generation.model`. The prompt and decoding settings stay fixed so comparisons remain valid.

**Add metrics.** Append additional RAGAS metric names (for example, context precision or context recall) to `evaluation.metrics`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `Neo4jError: authentication failure` | Wrong credentials in `.env` | Verify `NEO4J_USER` and `NEO4J_PASSWORD` match the container's `NEO4J_AUTH` |
| RAGAS scoring is slow or rate-limited | Many LLM judge calls per query | Lower `num_queries`, add retry/backoff, or use a faster judge model |
| Out-of-memory when embedding | Large batch size or corpus | Reduce the embedding batch size or the corpus subset |
| Inconsistent latency numbers | Cold caches or background load | Keep `warmup_queries` > 0 and run on an otherwise idle machine |

## License

Distributed under the MIT License. See `LICENSE` for details.
