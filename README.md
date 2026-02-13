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
  "question": "Your question here"
}
```

Response:

```json
{
  "answer": "Model answer",
  "sources": [
    {
      "source": "source name/path/url",
      "metadata": {},
      "distance": 0.1234
    }
  ]
}
```

Backend behavior for `POST /api/ask`:
- Loads Chroma `PersistentClient` from `~/RagBooks/ChromaDB`
- Uses collection `ragbooks`
- Embeds question with OpenAI
- Queries top 10 docs + metadatas
- Builds context
- Calls OpenAI chat model
- Returns `{answer, sources}`

Optional environment variables:
- `CHROMA_PATH` (default: `~/RagBooks/ChromaDB`)
- `CHROMA_COLLECTION` (default: `ragbooks`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)

If `OPENAI_API_KEY` is missing, `/api/ask` fails gracefully with a clear error.

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
