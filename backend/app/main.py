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
from app.timeline import get_timeline_options
from app.timeline import search_timeline

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
    rerank_sources: bool = True
    summaries_first: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    stage1_chapters: list[dict[str, Any]] = []


class FilterOptionsResponse(BaseModel):
    books: list[str]
    povs: list[str]


class TimelineEntry(BaseModel):
    book: str
    chapter: Optional[str] = None
    chapter_file: str
    pov: Optional[str] = None
    date_raw: Optional[str] = None
    key_events: list[str]
    locations: list[str]
    characters_present: list[str]
    chapter_ref: str


class TimelineSearchResponse(BaseModel):
    results: list[TimelineEntry]


class TimelineOptionsResponse(BaseModel):
    characters: list[str]
    dates: list[str]
    locations: list[str]


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@app.get("/api/filter-options", response_model=FilterOptionsResponse)
def filter_options() -> FilterOptionsResponse:
    return FilterOptionsResponse(
        books=_csv_env("ALLOWED_BOOKS"),
        povs=_csv_env("ALLOWED_POVS"),
    )


@app.get("/api/timeline/search", response_model=TimelineSearchResponse)
def timeline_search(
    character: Optional[str] = None,
    date: Optional[str] = None,
    location: Optional[str] = None,
    books: Optional[str] = None,
    limit: int = 100,
) -> TimelineSearchResponse:
    scoped_books = [item.strip() for item in books.split(",") if item.strip()] if books else None
    safe_limit = max(1, min(limit, 500))

    rows = search_timeline(
        character=character,
        date_substring=date,
        location=location,
        books=scoped_books,
        limit=safe_limit,
    )
    return TimelineSearchResponse(results=[TimelineEntry(**row) for row in rows])


@app.get("/api/timeline/options", response_model=TimelineOptionsResponse)
def timeline_options(
    books: Optional[str] = None,
    limit: int = 500,
) -> TimelineOptionsResponse:
    scoped_books = [item.strip() for item in books.split(",") if item.strip()] if books else None
    safe_limit = max(1, min(limit, 1000))
    options = get_timeline_options(books=scoped_books, limit=safe_limit)
    explicit_characters = _csv_env("ALLOWED_CHARACTERS")
    explicit_locations = _csv_env("ALLOWED_LOCATIONS")
    if explicit_characters:
        options["characters"] = explicit_characters[:safe_limit]
    if explicit_locations:
        options["locations"] = explicit_locations[:safe_limit]
    return TimelineOptionsResponse(**options)


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = ask_rag(
            question,
            books=payload.books,
            pov=payload.pov,
            rerank_sources=payload.rerank_sources,
            summaries_first=payload.summaries_first,
        )
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RagRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime integration failures
        raise HTTPException(status_code=500, detail=f"ask failed: {exc}") from exc

    return AskResponse(**result)
