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
│  llama-server Router (Host GPU :10000, --models-max 1)                 │
│  • Qwen3.5-2B (Default)                                                │
│  • Qwen3.5-2B-WSP & Qwen3.5-0.8B-WSP (Weight-Space Programming)        │
│  • Switch models directly in Web UI (:8000) or built-in UI (:10000)    │
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

### 3. Download 1–2B Models (Qwen3.5-2B + Weight-Space Programming)

Download **Qwen3.5-2B** (default) and optional **Weight-Space Programming (WSP)** variants into `/home/sapiens/models`:

```bash
# Official Qwen3.5-2B GGUF (default)
hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf \
  --local-dir /home/sapiens/models

# Qwen3.5-2B improved via Weight-Space Programming (loop-ablated direct answers)
hf download NikitaLazarev/Qwen3.5-2B-WSP-GGUF Qwen3.5-2B-UD-Q4_K_XL-wsp.gguf \
  --local-dir /home/sapiens/models

# RikiyaT / MIT Sparks of Weight-Space Programming Qwen3.5-0.8B (MATH-500 & IFEval boost)
hf download prithivMLmods/Qwen3.5-0.8B-weight-space-programming-GGUF \
  Qwen3.5-0.8B-weight-space-programming.Q8_0.gguf \
  --local-dir /home/sapiens/models
```

### 4. Build & Run as Systemd Daemon

If you haven't built `llama.cpp` yet:
```bash
./scripts/build-llama-cpp-gpu.sh
```

Install and start all three Sleeping Beauty services (`sleeping-beauty-llm` on `:10000`, `sleeping-beauty-embed` on `:10001`, and `sleeping-beauty` on `:8000`) via systemd:
```bash
# Recommended: user systemd services (no sudo required)
./scripts/install-systemd.sh

# Or system-wide services (/etc/systemd/system)
# sudo ./scripts/install-systemd.sh --system
```

Manage the daemon cleanly with `systemctl`:
```bash
systemctl --user start sleeping-beauty
systemctl --user stop sleeping-beauty
systemctl --user restart sleeping-beauty
systemctl --user status sleeping-beauty sleeping-beauty-llm sleeping-beauty-embed
journalctl --user -u sleeping-beauty -u sleeping-beauty-llm -u sleeping-beauty-embed -f
```

Open **`http://localhost:8000`** in your browser (or run manually in foreground with `./sb start-all && ./sb serve`).

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
