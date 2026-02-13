import os
from typing import Any
from typing import List
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import MissingOpenAIAPIKeyError
from app.rag import RagRuntimeError
from app.rag import ask_rag

app = FastAPI(title="RagBookGui API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "fastapi"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    books: Optional[List[str]] = None
    pov: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class FilterOptionsResponse(BaseModel):
    books: list[str]
    povs: list[str]


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@app.get("/api/filter-options", response_model=FilterOptionsResponse)
def filter_options() -> FilterOptionsResponse:
    return FilterOptionsResponse(
        books=_csv_env("ALLOWED_BOOKS"),
        povs=_csv_env("ALLOWED_POVS"),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = ask_rag(question)
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RagRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime integration failures
        raise HTTPException(status_code=500, detail=f"ask failed: {exc}") from exc

    return AskResponse(**result)
