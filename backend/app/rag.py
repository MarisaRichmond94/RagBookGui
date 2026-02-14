from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any
from typing import Optional

from dotenv import load_dotenv

DEFAULT_CHROMA_PATH = Path.home() / "RagBooks" / "ChromaDB"
DEFAULT_COLLECTION_NAME = "ragbooks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
CHROMA_DEFAULT_COLLECTION_CONFIG_JSON = (
    '{"hnsw_configuration": {"space": "l2", "ef_construction": 100, "ef_search": 10, '
    '"num_threads": 12, "M": 16, "resize_factor": 1.2, "batch_size": 100, '
    '"sync_threshold": 1000, "_type": "HNSWConfigurationInternal"}, '
    '"_type": "CollectionConfigurationInternal"}'
)

# Load backend/.env for local development.
load_dotenv(dotenv_path=BACKEND_ENV_PATH)


class MissingOpenAIAPIKeyError(Exception):
    pass


class RagRuntimeError(Exception):
    pass


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def _normalize_books(books: Optional[list[str]]) -> list[str]:
    if not books:
        return []
    return [book.strip() for book in books if book and book.strip()]


def _normalize_pov(pov: Optional[str]) -> Optional[str]:
    if not pov:
        return None
    normalized = pov.strip()
    return normalized or None


def _metadata_matches_filters(metadata: dict[str, Any] | None, books: list[str], pov: Optional[str]) -> bool:
    data = metadata or {}
    if books:
        book_value = data.get("book")
        if not isinstance(book_value, str) or book_value not in books:
            return False
    if pov:
        pov_value = data.get("pov")
        if not isinstance(pov_value, str) or pov_value != pov:
            return False
    return True


def _build_where_filter(books: list[str], pov: Optional[str]) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if books:
        clauses.append({"book": {"$in": books}})
    if pov:
        clauses.append({"pov": pov})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _merge_query_results(results: list[dict[str, Any]], n_results: int) -> dict[str, list[list[Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    insertion_order = 0

    for result in results:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for idx, doc_id in enumerate(ids):
            key = str(doc_id)
            candidate_distance = distances[idx] if idx < len(distances) else None
            candidate = {
                "id": key,
                "document": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
                "distance": candidate_distance,
                "order": insertion_order,
            }
            insertion_order += 1

            existing = merged.get(key)
            if not existing:
                merged[key] = candidate
                continue

            existing_distance = existing.get("distance")
            if existing_distance is None and candidate_distance is not None:
                merged[key] = candidate
            elif (
                existing_distance is not None
                and candidate_distance is not None
                and float(candidate_distance) < float(existing_distance)
            ):
                merged[key] = candidate

    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            float(row["distance"]) if row["distance"] is not None else float("inf"),
            row["order"],
        )
    )
    rows = rows[:n_results]

    return {
        "ids": [[row["id"] for row in rows]],
        "documents": [[row["document"] for row in rows]],
        "metadatas": [[row["metadata"] for row in rows]],
        "distances": [[row["distance"] for row in rows]],
    }


def _query_with_filters(
    collection: Any,
    query_vector: list[float],
    books: list[str],
    pov: Optional[str],
    n_results: int = 10,
) -> dict[str, Any]:
    include = ["documents", "metadatas", "distances"]
    where_filter = _build_where_filter(books, pov)

    try:
        if where_filter:
            return collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=include,
                where=where_filter,
            )
        return collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=include,
        )
    except Exception:
        # Fallback for clients/servers where $in support may be inconsistent.
        if books and len(books) > 1:
            per_book_results: list[dict[str, Any]] = []
            for book in books:
                per_book_where: dict[str, Any] = {"book": book}
                if pov:
                    per_book_where = {"$and": [per_book_where, {"pov": pov}]}
                per_book_results.append(
                    collection.query(
                        query_embeddings=[query_vector],
                        n_results=n_results,
                        include=include,
                        where=per_book_where,
                    )
                )
            return _merge_query_results(per_book_results, n_results)
        raise


def _fallback_keyword_query(
    collection: Any,
    question: str,
    books: list[str],
    pov: Optional[str],
    n_results: int = 10,
) -> dict[str, list[list[Any]]]:
    count = collection.count()
    if count == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    rows = collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    ids = rows.get("ids") or []
    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    question_tokens = _tokenize(question)

    scored: list[tuple[int, int]] = []
    for idx, doc in enumerate(docs):
        metadata = metas[idx] if idx < len(metas) else None
        if not _metadata_matches_filters(metadata, books, pov):
            continue
        doc_tokens = _tokenize(doc or "")
        overlap = len(question_tokens & doc_tokens)
        scored.append((overlap, idx))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_indices = [idx for _, idx in scored[:n_results]]

    top_ids = [ids[i] if i < len(ids) else f"row-{i}" for i in top_indices]
    top_docs = [docs[i] for i in top_indices]
    top_metas = [metas[i] if i < len(metas) else None for i in top_indices]
    # Keep distances shape compatible with query() output; lexical fallback has no vector distance.
    top_distances: list[None] = [None for _ in top_indices]
    return {"ids": [top_ids], "documents": [top_docs], "metadatas": [top_metas], "distances": [top_distances]}


def _is_missing_type_error(exc: Exception) -> bool:
    message = str(exc)
    if isinstance(exc, KeyError) and exc.args and exc.args[0] == "_type":
        return True
    return "'_type'" in message or '"_type"' in message


def _repair_legacy_collection_config(chroma_path: Path, collection_name: str) -> tuple[bool, str | None]:
    sqlite_path = chroma_path / "chroma.sqlite3"
    if not sqlite_path.exists():
        return (False, f"Chroma SQLite file not found at {sqlite_path}")

    try:
        with sqlite3.connect(str(sqlite_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE collections
                SET config_json_str = ?
                WHERE name = ?
                  AND (config_json_str = '{}' OR config_json_str IS NULL OR config_json_str = '')
                """,
                (CHROMA_DEFAULT_COLLECTION_CONFIG_JSON, collection_name),
            )
            updated = cursor.rowcount
            conn.commit()
        return (updated > 0, None)
    except sqlite3.Error as exc:
        return (False, str(exc))


def _chroma_schema_hint(exc: Exception) -> str | None:
    message = str(exc)
    if "collections.topic" in message:
        return (
            "Chroma schema/client mismatch: your local DB is newer than the installed "
            "chromadb client. Reinstall backend dependencies from backend/requirements.txt."
        )
    if _is_missing_type_error(exc):
        return (
            "Chroma could not parse collection config metadata ('_type'). "
            "This backend attempted an automatic repair for legacy '{}' configs. "
            "If the error persists, reinstall backend dependencies from "
            "backend/requirements.txt."
        )
    if "'dict' object has no attribute 'dimensionality'" in message:
        return (
            "Chroma vector index metadata is incompatible with the installed client. "
            "The backend will fall back to keyword retrieval, or you can rebuild the "
            "collection/index for full vector search behavior."
        )
    return None


def _source_name(metadata: dict[str, Any] | None, index: int) -> str:
    if not metadata:
        return f"Document {index}"

    for key in ("source", "file", "path", "url", "title", "document_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"Document {index}"


def _build_citation(metadata: dict[str, Any], source: str, rank: int) -> str:
    book = str(metadata.get("book") or source or f"Document {rank}")
    chapter = str(metadata.get("chapter") or metadata.get("chapter_file") or "?")
    pov = str(metadata.get("pov") or "Unknown")
    date = str(metadata.get("date") or "Unknown")
    chapter_file = str(metadata.get("chapter_file") or "Unknown")
    chunk_index = str(metadata.get("chunk_index") if metadata.get("chunk_index") is not None else rank - 1)
    return f"{book} | Ch {chapter} | POV {pov} | Date {date} | File {chapter_file} | Chunk {chunk_index}"


def ask_rag(question: str, books: Optional[list[str]] = None, pov: Optional[str] = None) -> dict[str, Any]:
    books = _normalize_books(books)
    pov = _normalize_pov(pov)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingOpenAIAPIKeyError(
            "OPENAI_API_KEY is missing. Add it to backend/.env or your environment."
        )

    try:
        import chromadb
        from openai import OpenAI
    except Exception as exc:
        raise RagRuntimeError(
            "Backend dependencies for RAG are not available. "
            "Run `pip install -r backend/requirements.txt`."
        ) from exc

    chroma_path = Path(os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_PATH))).expanduser()
    collection_name = os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    chroma = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = chroma.get_collection(name=collection_name)
    except Exception as exc:
        if _is_missing_type_error(exc):
            repaired, repair_error = _repair_legacy_collection_config(chroma_path, collection_name)
            if repaired:
                collection = chroma.get_collection(name=collection_name)
            else:
                hint = _chroma_schema_hint(exc)
                if hint:
                    if repair_error:
                        hint = f"{hint} Auto-repair failed: {repair_error}."
                    raise RagRuntimeError(hint) from exc
                raise
        else:
            hint = _chroma_schema_hint(exc)
            if hint:
                raise RagRuntimeError(hint) from exc
            raise RagRuntimeError(
                f"Failed to load Chroma collection '{collection_name}' from '{chroma_path}': {exc}"
            ) from exc

    client = OpenAI(api_key=api_key)
    embedding = client.embeddings.create(model=embedding_model, input=question)
    query_vector = embedding.data[0].embedding

    try:
        result = _query_with_filters(
            collection=collection,
            query_vector=query_vector,
            books=books,
            pov=pov,
            n_results=10,
        )
    except Exception as exc:
        if "'dict' object has no attribute 'dimensionality'" in str(exc):
            result = _fallback_keyword_query(collection, question, books=books, pov=pov, n_results=200)
        else:
            hint = _chroma_schema_hint(exc)
            if hint:
                raise RagRuntimeError(hint) from exc
            raise

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for idx, doc in enumerate(documents, start=1):
        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else None
        if not _metadata_matches_filters(metadata, books, pov):
            continue

        doc_id = ids[idx - 1] if idx - 1 < len(ids) else f"doc-{idx}"
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        source = _source_name(metadata, idx)
        context_blocks.append(f"[{idx}] {source}\n{doc}")
        data = metadata or {}

        sources.append(
            {
                "id": doc_id,
                "rank": idx,
                "source": source,
                "metadata": data,
                "distance": distance,
                "text": doc or "",
                "citation": _build_citation(data, source, idx),
                "snippet": (doc or "")[:600],
            }
        )

    if not context_blocks:
        return {
            "answer": "I could not find relevant context for that question with the selected filters.",
            "sources": [],
        }

    context = "\n\n".join(context_blocks)
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "If the context does not contain the answer, say you do not know."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
    )

    answer = completion.choices[0].message.content or "No answer returned."
    return {"answer": answer, "sources": sources}
