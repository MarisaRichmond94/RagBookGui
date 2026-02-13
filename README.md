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

### 2) Run both apps

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

The frontend calls `GET http://localhost:8000/api/health` to confirm connectivity.
