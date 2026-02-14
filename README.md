# RagBookGui Monorepo

Monorepo with:
- `frontend`: Vite + React + TypeScript
- `backend`: FastAPI (Python)

## Project structure

```text
.
├── backend
│   ├── app
│   │   └── main.py
│   └── requirements.txt
├── frontend
│   ├── src
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
└── package.json
```

## Run locally

### 1) One-time setup

```bash
pnpm install
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Configure backend API key

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
OPENAI_API_KEY=your_key_here
```

Do not commit `backend/.env` (it is ignored in `.gitignore`).

### 3) Run backend with uvicorn

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend
```

### 4) Run both apps

```bash
pnpm run dev
```

This starts:
- Backend at `http://localhost:8000`
- Frontend at `http://localhost:5173`

### 5) Rebuild index (paragraph-aware chunking)

```bash
pnpm run backend:reindex
```

This reindexes `~/RagBooks/Books` into `~/RagBooks/ChromaDB` using:
- paragraph-aware chunks split on blank lines
- max chunk size around 2800 chars
- 1-paragraph overlap between adjacent chunks
- collection recreation (`ragbooks` and `ragbooks_chapter_summaries`) before indexing so old and new chunks are not mixed
- one chapter-level summary per chapter (<= 8 bullets) for two-stage retrieval
- timeline extraction artifacts written to `~/RagBooks/Artifacts/timeline.db`

You can still run each one separately with:
- `pnpm run backend:dev`
- `pnpm run frontend:dev`

## CORS

CORS is enabled in `backend/app/main.py` for:
- `http://localhost:5173`
- `http://127.0.0.1:5173`

## API

### `GET /api/health`

Returns backend health status.

### `POST /api/ask`

Request:

```json
{
  "question": "Your question here",
  "books": ["Optional", "Book Filter"],
  "pov": "Optional POV Filter",
  "rerank_sources": true,
  "summaries_first": false
}
```

Response:

```json
{
  "answer": "Model answer",
  "stage1_chapters": [
    {
      "book": "Book Name",
      "chapter": 4,
      "chapter_file": "04_4.txt",
      "pov": "Character",
      "date": "Monday, November 9th",
      "score": 8.4,
      "summary_snippet": "- Key chapter summary points..."
    }
  ],
  "sources": [
    {
      "source": "source name/path/url",
      "citation": "Book | Ch # | POV | Date | File | Chunk",
      "snippet": "first ~600 chars",
      "score": 8.7,
      "metadata": {},
      "distance": 0.1234
    }
  ]
}
```

Backend behavior for `POST /api/ask`:
- Loads Chroma `PersistentClient` from `~/RagBooks/ChromaDB`
- Uses collections `ragbooks` and `ragbooks_chapter_summaries`
- Embeds question with OpenAI
- Optional Stage 1: selects top chapter summaries first (global question or `summaries_first=true`)
- Stage 2: retrieves evidence passages from `ragbooks`, constrained to Stage 1 chapters when used
- Builds context
- Retrieves a larger candidate set from Chroma
- Reranks candidates (LLM by default, embedding fallback)
- Calls OpenAI chat model
- Returns `{answer, sources}`

Optional environment variables:
- `CHROMA_PATH` (default: `~/RagBooks/ChromaDB`)
- `CHROMA_COLLECTION` (default: `ragbooks`)
- `CHROMA_SUMMARY_COLLECTION` (default: `ragbooks_chapter_summaries`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_RERANK_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_SUMMARY_MODEL` (default: `gpt-4o-mini`)
- `RAG_CANDIDATE_COUNT` (default: `60`)
- `RAG_TOP_K` (default: `12`)
- `RAG_RERANK_MODE` (default: `llm`, options: `llm`, `embedding`)
- `RAG_STAGE1_CHAPTER_COUNT` (default: `10`)

If `OPENAI_API_KEY` is missing, `/api/ask` fails gracefully with a clear error.

### `GET /api/timeline/search`

Query timeline artifacts extracted during indexing.

Supported query params:
- `character` (substring match against `characters_present`)
- `date` (substring match against `date_raw`)
- `location` (substring match against `locations`)
- `books` (comma-separated book names for scope)
- `limit` (default `100`, max `500`)

Example:

```bash
curl "http://localhost:8000/api/timeline/search?character=Jared&location=Boston&books=Faded"
```

Response:

```json
{
  "results": [
    {
      "book": "Faded",
      "chapter": "4",
      "chapter_file": "04_4.txt",
      "pov": "Jared Gatlin",
      "date_raw": "Monday, November 9th",
      "key_events": ["..."],
      "locations": ["..."],
      "characters_present": ["..."],
      "chapter_ref": "Faded/04_4.txt"
    }
  ]
}
```

### `GET /api/timeline/options`

Returns distinct option lists for searchable timeline filters.

Supported query params:
- `books` (comma-separated book names for scope)
- `limit` (default `500`, max `1000`)

Response:

```json
{
  "characters": ["Jared Gatlin", "..."],
  "dates": ["Monday, November 9th", "..."],
  "locations": ["Boston", "..."]
}
```

## Timeline Artifacts

Timeline extraction runs during `pnpm run backend:reindex` and stores one structured record per chapter in:
- `~/RagBooks/Artifacts/timeline.db`

Schema fields:
- `book`
- `chapter`
- `pov`
- `date_raw`
- `key_events` (JSON array)
- `locations` (JSON array)
- `characters_present` (JSON array)
- `chapter_ref`

Regeneration behavior:
- Timeline artifacts are rebuilt each reindex run (`reset=True`), so they stay aligned with current chapter files.

Safety:
- Artifacts are not committed; `.gitignore` includes `Artifacts/`.

## Troubleshooting

If `/api/ask` fails with Chroma errors like:
- `no such column: collections.topic`
- `'_type'`

your local Chroma DB schema and installed `chromadb` client are out of sync.

Fix:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```
