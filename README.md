# Enterprise AI Study Companion

AI-powered B2B knowledge & exam-preparation engine. Upload PDFs (course notes, manuals, policies) and generate **grounded** study material — a concise summary, key concepts, formulas and 3 self-assessment Q&As — where every item cites the exact source page.

**Stack:** FastAPI · Supabase (PostgreSQL + pgvector) · `all-MiniLM-L6-v2` local embeddings · Groq LLM · PyMuPDF · EasyOCR · vanilla JS frontend

## How it works

1. **Ingest** — `POST /api/v1/upload-pdf` extracts text (OCR for scanned pages), strips noise (headers/footers, page numbers, TOC leaders), splits into ~500-char sentence-aware chunks, embeds them and stores them in pgvector.
2. **Retrieve** — the query is embedded and matched via the `match_document_chunks` RPC (cosine similarity, optional subject/topic filters).
3. **Generate** — `POST /api/v1/generate-study-material` sends numbered passages to the LLM with strict grounding rules and JSON output.

### Anti-hallucination guardrails
- No relevant passages → the LLM is never called; the response is `grounded: false`.
- The prompt restricts the model to the provided passages, requires citations, and treats document text as data, not instructions.
- Output is schema-validated (exactly 3 questions), with one corrective retry.
- Citations to non-existent passages are removed; concepts/formulas left without a valid citation are dropped.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your keys
```

Run [`schema.sql`](schema.sql) in the Supabase SQL Editor. Use the **service_role** key in `.env` (not the anon key).

```bash
uvicorn main:app
```

- Web UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## Configuration

See [`.env.example`](.env.example). Notable options: `GROQ_MODEL` (default `openai/gpt-oss-120b`), `CHUNK_SIZE`, `RAG_TOP_K`, `RAG_MATCH_THRESHOLD`, `OCR_ENABLED`, `OCR_LANGUAGES` (e.g. `en,az`), `CORS_ORIGINS`.

> OCR runs on CPU and takes ~15–20 s per scanned page. Set `OCR_ENABLED=false` for faster startup if you only use text PDFs.

## Project structure

```
main.py                  FastAPI app, CORS, static frontend
schema.sql               pgvector table, index, similarity-search RPC
app/api/endpoints.py     HTTP endpoints
app/core/                settings, clients (Supabase, Groq, embedder), error handling
app/models/schemas.py    request/response models
app/services/            pdf_service (ingest), ocr_service, rag_service
frontend/                web UI (HTML/CSS/JS)
```
