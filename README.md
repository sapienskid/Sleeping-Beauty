# Sleeping Beauty Local AI

A fully local RAG-powered AI assistant. Upload your documents, build knowledge bases, and use them to write emails, create social media posts in your style, do research, and more -- all private, all offline.

**Stack:** llama.cpp (host) + Open WebUI + Qdrant + Docling + Qwen3 Embeddings + Edge TTS

## Quick start

### 1. Configure

```bash
cp .env.example .env
# Edit .env -- set HF_TOKEN at minimum
```

### 2. Build llama.cpp

```bash
./scripts/build-llama-cpp-gpu.sh
```

### 3. Download a model

```bash
./scripts/download-hf-model.sh bartowski/Llama-3.1-8B-Instruct-GGUF \
  --file Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### 4. Install llama.cpp as a service

```bash
sudo ./scripts/install-llama-systemd.sh
```

### 5. Start the stack

```bash
docker compose up -d --pull always
```

Done. Open http://localhost:3000.

## Endpoints

| Service | URL |
|---------|-----|
| Open WebUI | http://localhost:3000 |
| llama.cpp API | http://localhost:10000/v1 |
| Qdrant | http://localhost:6333 |
| Docling UI | http://localhost:5001/ui |
| Embeddings | http://localhost:8081/v1/embeddings |
| Edge TTS | http://localhost:5050/v1 |
| Edge TTS voices | http://localhost:5050/v1/audio/voices |

## How to use for writing & research

1. **Build knowledge bases**: Upload your past emails, blog posts, social media content, or any documents via Open WebUI. They get chunked, embedded, and stored in Qdrant automatically.

2. **Write in your style**: Reference your knowledge base in conversations. Ask the AI to draft emails or posts based on your uploaded writing samples.

3. **Research**: Enable web fetch (`ENABLE_RAG_LOCAL_WEB_FETCH=true`, already set) to let the AI pull web content into your RAG pipeline during conversations.

4. **Iterate**: Upload new documents anytime. The vector store grows with your knowledge.

## Voice conversations (English practice)

The stack includes text-to-speech (Edge TTS) and speech-to-text (Whisper) for hands-free voice conversations — great for practicing spoken English.

**How to use:**

1. Open http://localhost:3000 and start a chat.
2. Click the **headphone icon** (top right of the chat) to enter **Voice Call Mode**.
3. Speak naturally — Whisper transcribes your speech, the LLM responds, and Edge TTS reads the reply aloud.
4. The call auto-listens after ~3 seconds of silence. Tap or speak to interrupt.

**Changing the voice:**

Browse available voices at http://localhost:5050/v1/audio/voices. To switch, update `AUDIO_TTS_VOICE` and `EDGE_TTS_DEFAULT_VOICE` in `.env`, then restart:

```bash
docker compose up -d
```

**If you see `401 UNAUTHORIZED` from `http://edge-tts:5050/v1/audio/speech`:**

`travisvn/openai-edge-tts` requires a bearer token by default. Ensure these match in `.env`:

- `AUDIO_TTS_OPENAI_API_KEY`
- `EDGE_TTS_API_KEY`

Then restart:

```bash
docker compose up -d edge-tts open-webui
```

Optional direct check:

```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Authorization: Bearer sk-edge-tts" \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Voice test","voice":"en-US-AndrewNeural"}' \
  -o /tmp/voice-test.mp3
```

**STT note:** Whisper runs on CPU inside the Open WebUI container (no VRAM used). The `base` model is fast enough for real-time voice clips. Upgrade to `small` in `.env` (`WHISPER_MODEL=small`) if accuracy matters more than latency.

**Open WebUI realtime call mode checks (if still silent):**

1. Open chat settings and verify TTS engine is set to `openai`.
2. Confirm TTS base URL is `http://edge-tts:5050/v1` and API key matches `.env`.
3. Allow browser microphone access and autoplay for `http://localhost:3000`.
4. In browser devtools, confirm `POST /api/v1/audio/speech` returns `200` (not `401/500`).

## Supported document formats

Docling (using the full base image) handles:

| Format | Types |
|--------|-------|
| Office | PDF, DOCX, XLSX, PPTX |
| Web | HTML, XHTML, Markdown |
| Text | CSV, LaTeX, AsciiDoc |
| Images | PNG, JPEG, TIFF, BMP, WEBP |
| XML | USPTO patents, JATS articles, XBRL financial reports |

**EPUB**: Convert to HTML first (e.g. `ebook-convert book.epub book.html` via Calibre), then upload the HTML.

## Key `.env` knobs

Most things work out of the box. These are the ones worth knowing:

| Variable | Default | What it does |
|----------|---------|-------------|
| `LLAMA_N_GPU_LAYERS` | 20 | GPU layers offloaded. Lower if OOM. |
| `LLAMA_MODELS_MAX` | 1 | Only one model loaded at a time. Old model is unloaded before the new one loads. |
| `LLAMA_CACHE_RAM` | 1400 | Max model cache in MiB. Prevents VRAM exhaustion when switching models. |
| `LLAMA_CONTEXT_SIZE` | 4096 | Context window. Lower to 2048 if tight on VRAM. |
| `RAG_TOP_K` | 6 | How many chunks are retrieved per query. |
| `CHUNK_SIZE` | 800 | Characters per chunk. |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks for context continuity. |
| `RAG_HYBRID_BM25_WEIGHT` | 0.30 | Balance between keyword (BM25) and semantic search. |
| `AUDIO_TTS_VOICE` | `en-US-AndrewNeural` | TTS voice. See localhost:5050/v1/audio/voices for options. |
| `EDGE_TTS_API_KEY` | `sk-edge-tts` | Must match `AUDIO_TTS_OPENAI_API_KEY` to avoid TTS 401 errors. |
| `WHISPER_MODEL` | `base` | STT model size. `tiny`/`base`/`small`/`medium`/`large`. |
| `DOCLING_NUM_THREADS` | 6 | CPU threads for document processing. |

## Managing llama.cpp

```bash
sudo systemctl status llama-cpp     # check status
sudo journalctl -u llama-cpp -f     # live logs
sudo systemctl restart llama-cpp    # restart after .env changes
```

After changing llama.cpp settings in `.env`, re-run the install script to update the systemd env:

```bash
sudo ./scripts/install-llama-systemd.sh
```

## Switching models

All GGUF files in `~/models/` are automatically available in Open WebUI's model dropdown. Just pick a different model and start chatting -- llama.cpp's router will unload the old model and load the new one.

`LLAMA_MODELS_MAX=1` ensures only one model is loaded at a time, so the old model's VRAM is freed before the new one loads. `LLAMA_CACHE_RAM=1400` caps the total model cache to ~1.4 GB to stay within the GTX 1650's budget (the embedding model takes ~1.3 GB).

If a model switch hangs (rare, usually an OOM), restart the service:

```bash
sudo systemctl restart llama-cpp
```

## Stop

```bash
docker compose down
```

## Architecture

```
Browser :3000
    |
Open WebUI (Docker)
    |--- LLM ---------> llama.cpp (host :10000, GPU)
    |--- Embeddings ---> TEI / Qwen3-0.6B (Docker :8081, GPU)
    |--- Documents ----> Docling (Docker :5001, CPU)
    |--- Vectors ------> Qdrant (Docker :6333)
    |--- TTS ----------> Edge TTS (Docker :5050, CPU)
    |--- STT ----------> Built-in Whisper (CPU, inside Open WebUI)
```

## Requirements

- NVIDIA GPU with CUDA (tested on GTX 1650 4GB)
- NVIDIA Container Toolkit (for GPU passthrough to Docker)
- Docker with Compose
- ~15GB disk for images + model weights
