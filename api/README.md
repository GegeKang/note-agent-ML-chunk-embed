# ML Service — HTTP API

FastAPI controller that exposes the `ml/` and `backend/` packages as
stateless HTTP endpoints. No existing files are modified.

## Install

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt          # the existing ML deps
pip install -r api/requirements.txt      # fastapi + uvicorn + python-multipart
python -m spacy download en_core_web_sm  # required for /ml/chunk quality
```

## Run

```bash
# from repo root (note-agent-ML/)
export GROQ_API_KEY=gsk_...              # or OPENAI_API_KEY — needed for /ml/extract + /ml/notes/process
export ML_INTERNAL_KEY=change-me         # optional shared secret; empty disables auth
uvicorn api.server:app --host 0.0.0.0 --port 9000
```

OpenAPI docs: `http://localhost:9000/docs`

### Docker (optional)

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt \
    && python -m spacy download en_core_web_sm
COPY ml ./ml
COPY backend ./backend
COPY api ./api
EXPOSE 9000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "9000"]
```

## Auth

If `ML_INTERNAL_KEY` is set, every request must include:

```
X-Internal-Key: <key>
```

If unset, endpoints are open (useful for local dev).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness check |
| POST | `/ml/chunk` | Sentence-aware windowed chunking |
| POST | `/ml/embed` | Sentence-transformers vectors (384-dim) |
| POST | `/ml/extract` | LLM object + link extraction |
| POST | `/ml/resolve` | Duplicate detection with cosine similarity |
| POST | `/ml/search` | Hybrid vector + keyword search with RRF |
| POST | `/ml/insights/contradictions` | Graph-based contradiction detection |
| POST | `/ml/insights/stale-threads` | Stale question/task/idea detection |
| POST | `/ml/notes/process` | Full end-to-end pipeline (chunk → embed → extract → insights) |
| POST | `/ml/extract-text` | PDF/DOCX/image/audio → plain text |
| GET  | `/ml/feedback/pending` | List pending human-review items |
| POST | `/ml/feedback/review/{id}` | Submit a human correction |
| GET  | `/ml/feedback/stats` | Review counts |

Full schemas: see `/docs`.

## Why stateless?

The backend holds the authoritative DB (Postgres + pgvector). The ML
service just processes payloads in/out — no shared DB, no workspace
leakage, easy to horizontally scale. The two exceptions:

- `/ml/extract` / `/ml/notes/process` write to a local SQLite
  `feedback.db` (via `ml/feedback.py`) so that HITL corrections can be
  replayed as few-shot examples on the next extract call.
- `/ml/extract-text` writes an uploaded file to `tmp` during processing
  and deletes it afterward.

## Embedding model

Pinned to `sentence-transformers/all-MiniLM-L6-v2` (384-dim) to match the
backend's pgvector column dimension. Override via `EMBEDDING_MODEL` env
var — but the backend's `Vector(384)` columns must be changed in lockstep.

## Typical wiring from the backend

```python
# Backend's app/ml/pipeline.py becomes a thin HTTP client:
import httpx

ML_URL = os.getenv("ML_SERVICE_URL", "http://ml:9000")
ML_KEY = os.getenv("ML_INTERNAL_KEY", "")
HEADERS = {"X-Internal-Key": ML_KEY} if ML_KEY else {}

async def process_note(note_id, workspace_id, text):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{ML_URL}/ml/notes/process",
            headers=HEADERS,
            json={"text": text, "note_id": str(note_id), "workspace_id": str(workspace_id)},
        )
        r.raise_for_status()
        return r.json()  # {spans, embeddings, objects, links, mentions, contradictions, ...}
```
