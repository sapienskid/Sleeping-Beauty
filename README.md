# Sleeping Beauty — Local Knowledge Search & RAG Hub

A zero-Docker, ultra-fast local knowledge search and retrieval-augmented generation (RAG) assistant. Index any local folder or vault (Markdown notes, research papers, codebases, PDFs) and search or chat with your knowledge using **EmbeddingGemma-2**, **LanceDB**, and **1–2B micro LLMs** on your local GPU.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SLEEPING BEAUTY LOCAL HUB                            │
├────────────────────────────────────────────────────────────────────────┤
│  Web Dashboard :8000   │   Terminal CLI (./sb)   │  OpenAI Proxy /v1   │
├────────────────────────┴─────────────────────────┴─────────────────────┤
│                          CORE PYTHON ENGINE                            │
│  ├── Document Ingest (Markdown, Code, TXT, PDF via PyMuPDF)            │
│  ├── Header-Aware Semantic Chunking (350-450 tokens context budget)    │
│  ├── EmbeddingGemma-2 (768d with MRL & task instruction prefixes)      │
│  ├── LanceDB (In-process, Tantivy BM25 + Dense Vector Hybrid Search)   │
│  └── Grounded Micro-RAG Engine (Strict XML delimiters, zero-halluc)    │
├────────────────────────────────────────────────────────────────────────┤
│                          LOCAL LLM RUNTIME                             │
│  llama-server (Host GPU :10000)                                        │
│  • Micro LLMs: Qwen 2.5 1.5B / DeepSeek-R1-1.5B / Gemma 2 2B           │
│  • Built-in llama.cpp web UI also available at :10000                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Why This Stack?

1. **Zero Docker Overhead:** Replaced heavy containers (Docling, Qdrant, Open WebUI, Edge-TTS) with in-process, high-performance native libraries. Saves ~2GB of host RAM and eliminates container latency.
2. **True Hybrid Search (BM25 + Vectors):** LanceDB embeds Tantivy full-text keyword search alongside dense vector ANN, fused with **Reciprocal Rank Fusion (RRF)**. Exact code symbols, names, and dates are retrieved alongside conceptual matches.
3. **Optimized for 1–2B Micro LLMs:**
   * Runs at **Q8_0 or FP16** with 100% GPU offload on a 4GB GTX 1650 (~80–120 tokens/sec).
   * Strict context budgeting (Top 3–4 chunks, ~1,500 total tokens) prevents "lost in the middle" errors.
   * Explicit XML `<context>` delimiters and low temperature (`0.05`) ensure strictly grounded answers with citations.
4. **Any-Vault / Dynamic Ingestion:** Point to any folder or Obsidian vault at any time (`./sb index /path/to/vault` or via Web UI).

---

## Quick Start

### 1. Requirements

* Linux (x86_64)
* NVIDIA GPU (tested on GTX 1650 4GB) with CUDA drivers
* `uv` Python package manager

### 2. Environment Setup

```bash
# Clone and enter repo
cd Sleeping-Beauty

# Setup dependencies (managed automatically via uv and ./sb)
cp .env.example .env
./sb --help
```

### 3. Download a 1–2B Micro Model

Run the download script to fetch one of the recommended models:

```bash
# Top pick for fast RAG & code (Qwen 2.5 1.5B Instruct Q8_0):
./scripts/download-hf-model.sh Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --file qwen2.5-1.5b-instruct-q8_0.gguf

# Or the micro reasoning model (DeepSeek-R1 Distill 1.5B with chain-of-thought):
./scripts/download-hf-model.sh unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
  --file DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf

# Or Google Gemma 2 2B Instruct:
./scripts/download-hf-model.sh bartowski/gemma-2-2b-it-GGUF \
  --file gemma-2-2b-it-Q8_0.gguf
```

### 4. Build & Run llama-server

If you haven't built `llama.cpp` yet:
```bash
./scripts/build-llama-cpp-gpu.sh
```

Run as a systemd service (or run directly):
```bash
sudo ./scripts/install-llama-systemd.sh
```

The `llama-server` is now live at `http://localhost:10000` (including its own built-in web UI).

### 5. Start the Sleeping Beauty Dashboard

```bash
./sb serve
```

Open **`http://localhost:8000`** in your browser.

---

## CLI Usage

### Index Any Directory or Vault
```bash
# Index an Obsidian vault or notes directory
./sb index "/home/sapiens/Second Brain/Notes"

# Index a code repository
./sb index ~/Development/MyProject

# Index with custom chunking
./sb index ~/Documents/Papers --chunk-size 450 --overlap 90
```

### Hybrid Search (BM25 + Dense Vectors)
```bash
./sb search "vector projection FAISS"
./sb search "Gemma 2 MRL embeddings" --top-k 5
./sb search "nepali dictation" --vault "Notes"
```

### Ask Knowledge (RAG Q&A)
```bash
./sb ask "What is the core architecture of the Nepali dictation engine?"
./sb ask "How does Matryoshka Representation Learning work in EmbeddingGemma?"
```

### View Stats
```bash
./sb stats
```

---

## Web Dashboard Features (`http://localhost:8000`)

* **Ask Knowledge (RAG):** Live streaming chat with source citations. Automatically renders `<think>...</think>` collapsible thought trees for reasoning models like DeepSeek-R1.
* **Hybrid Search:** Live search bar showing similarity scores, file paths, and exact highlighted text snippets.
* **Manage Vaults:** Ingest any directory path with progress feedback, view indexed collections, or delete collections.
* **OpenAI-Compatible Proxy (`/v1/chat/completions`):** Connect external tools (Zed editor, Obsidian plugins, curl) directly to your augmented local knowledge base.

---

## Hardware Budget (Tuned for 4GB GTX 1650)

| Component | VRAM Usage | Notes |
| :--- | :--- | :--- |
| `llama-server` (1.5B Q8_0) | ~1.6 GB | 100% layers offloaded to GPU (`-ngl 99`, `-fa on`) |
| `EmbeddingGemma-2` | ~0.7 – 1.0 GB | Runs in FP16 on GPU (or auto-falls back to CPU) |
| `LanceDB` + `Tantivy` | 0 GB VRAM | Runs in system RAM (sub-5ms retrieval) |
| **Total VRAM Headroom** | **~1.4 GB free** | Safe margin for KV cache and OS display |
