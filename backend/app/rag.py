from __future__ import annotations

import json
import math
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
DEFAULT_RERANK_MODEL = "gpt-4o-mini"
DEFAULT_CANDIDATE_COUNT = 60
DEFAULT_TOP_K = 12
DEFAULT_RERANK_MODE = "llm"
BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
CHROMA_DEFAULT_COLLECTION_CONFIG_JSON = (
    '{"hnsw_configuration": {"space": "l2", "ef_construction": 100, "ef_search": 10, '
    '"num_threads": 12, "M": 16, "resize_factor": 1.2, "batch_size": 100, '
    '"sync_threshold": 1000, "_type": "HNSWConfigurationInternal"}, '
    '"_type": "CollectionConfigurationInternal"}'
)

load_dotenv(dotenv_path=BACKEND_ENV_PATH)


class MissingOpenAIAPIKeyError(Exception):
    pass


class RagRuntimeError(Exception):
    pass


def _get_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


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
    n_results: int,
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
        # Fallback for environments where $in support may be inconsistent.
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
    n_results: int,
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
    date = str(metadata.get("date") or metadata.get("chapter_date") or "Unknown")
    chapter_file = str(metadata.get("chapter_file") or "Unknown")
    chunk_index = str(metadata.get("chunk_index") if metadata.get("chunk_index") is not None else rank - 1)
    return f"{book} | Ch {chapter} | POV {pov} | Date {date} | File {chapter_file} | Chunk {chunk_index}"


def _try_parse_rerank_response(text: str) -> tuple[float, str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return (0.0, "Could not parse reranker output.")
        parsed = json.loads(text[start : end + 1])

    score = parsed.get("score", 0)
    rationale = parsed.get("rationale", "")
    try:
        numeric = float(score)
    except Exception:
        numeric = 0.0
    numeric = max(0.0, min(10.0, numeric))
    return (numeric, str(rationale).strip() or "No rationale provided.")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_candidates_llm(
    openai_client: Any,
    rerank_model: str,
    question: str,
    candidates: list[dict[str, Any]],
) -> None:
    for candidate in candidates:
        snippet = (candidate.get("document") or "")[:1400]
        citation = candidate.get("citation", "")
        response = openai_client.chat.completions.create(
            model=rerank_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval reranker. Score source relevance from 0 to 10. "
                        "Respond with strict JSON: {\"score\": <number>, \"rationale\": <one sentence>}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Candidate citation:\n{citation}\n\n"
                        f"Candidate text:\n{snippet}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        score, rationale = _try_parse_rerank_response(content)
        candidate["score"] = score
        candidate["rerank_rationale"] = rationale


def _get_stored_embeddings(collection: Any, ids: list[str]) -> dict[str, list[float]]:
    try:
        rows = collection.get(ids=ids, include=["embeddings"])
    except Exception:
        return {}
    out: dict[str, list[float]] = {}
    row_ids = rows.get("ids") or []
    row_embeddings = rows.get("embeddings") or []
    for idx, doc_id in enumerate(row_ids):
        emb = row_embeddings[idx] if idx < len(row_embeddings) else None
        if isinstance(emb, list):
            out[str(doc_id)] = emb
    return out


def _score_candidates_embedding(
    openai_client: Any,
    embedding_model: str,
    collection: Any,
    question_embedding: list[float],
    candidates: list[dict[str, Any]],
) -> None:
    ids = [str(candidate["id"]) for candidate in candidates]
    stored_embeddings = _get_stored_embeddings(collection, ids)

    for candidate in candidates:
        doc_id = str(candidate["id"])
        candidate_embedding = stored_embeddings.get(doc_id)
        if not candidate_embedding:
            text = candidate.get("document") or ""
            emb = openai_client.embeddings.create(model=embedding_model, input=text)
            candidate_embedding = emb.data[0].embedding

        similarity = _cosine_similarity(question_embedding, candidate_embedding)
        # Normalize cosine range [-1,1] to score range [0,10].
        score = max(0.0, min(10.0, (similarity + 1.0) * 5.0))
        candidate["score"] = score
        candidate["rerank_rationale"] = "Embedding cosine similarity."


def select_top_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda item: (
            -float(item[1].get("score", 0.0)),
            item[0],
        )
    )
    return [candidate for _, candidate in indexed[:top_k]]


def _rerank_candidates(
    openai_client: Any,
    collection: Any,
    question: str,
    question_embedding: list[float],
    candidates: list[dict[str, Any]],
    rerank_mode: str,
    embedding_model: str,
    rerank_model: str,
) -> list[dict[str, Any]]:
    mode = (rerank_mode or DEFAULT_RERANK_MODE).lower()
    if mode not in {"llm", "embedding"}:
        mode = DEFAULT_RERANK_MODE

    try:
        if mode == "embedding":
            _score_candidates_embedding(
                openai_client=openai_client,
                embedding_model=embedding_model,
                collection=collection,
                question_embedding=question_embedding,
                candidates=candidates,
            )
        else:
            _score_candidates_llm(
                openai_client=openai_client,
                rerank_model=rerank_model,
                question=question,
                candidates=candidates,
            )
    except Exception:
        # Guaranteed fallback mode.
        _score_candidates_embedding(
            openai_client=openai_client,
            embedding_model=embedding_model,
            collection=collection,
            question_embedding=question_embedding,
            candidates=candidates,
        )

    return candidates


def ask_rag(
    question: str,
    books: Optional[list[str]] = None,
    pov: Optional[str] = None,
    rerank_sources: bool = True,
) -> dict[str, Any]:
    books = _normalize_books(books)
    pov = _normalize_pov(pov)
    candidate_count = _get_int_env("RAG_CANDIDATE_COUNT", DEFAULT_CANDIDATE_COUNT, minimum=10, maximum=120)
    top_k = _get_int_env("RAG_TOP_K", DEFAULT_TOP_K, minimum=3, maximum=30)
    rerank_mode = os.getenv("RAG_RERANK_MODE", DEFAULT_RERANK_MODE).strip().lower()

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
    rerank_model = os.getenv("OPENAI_RERANK_MODEL", DEFAULT_RERANK_MODEL)

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
    question_embedding = embedding.data[0].embedding

    try:
        result = _query_with_filters(
            collection=collection,
            query_vector=question_embedding,
            books=books,
            pov=pov,
            n_results=candidate_count,
        )
    except Exception as exc:
        if "'dict' object has no attribute 'dimensionality'" in str(exc):
            result = _fallback_keyword_query(
                collection=collection,
                question=question,
                books=books,
                pov=pov,
                n_results=max(candidate_count, 120),
            )
        else:
            hint = _chroma_schema_hint(exc)
            if hint:
                raise RagRuntimeError(hint) from exc
            raise

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    candidates: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents, start=1):
        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else None
        if not _metadata_matches_filters(metadata, books, pov):
            continue

        data = metadata or {}
        doc_id = ids[idx - 1] if idx - 1 < len(ids) else f"doc-{idx}"
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        source = _source_name(metadata, idx)
        candidates.append(
            {
                "id": str(doc_id),
                "rank": idx,
                "source": source,
                "metadata": data,
                "distance": distance,
                "document": doc or "",
                "text": doc or "",
                "citation": _build_citation(data, source, idx),
                "snippet": (doc or "")[:600],
                "score": 0.0,
                "rerank_rationale": "",
            }
        )

    if not candidates:
        return {
            "answer": "I could not find relevant context for that question with the selected filters.",
            "sources": [],
        }

    if rerank_sources:
        reranked = _rerank_candidates(
            openai_client=client,
            collection=collection,
            question=question,
            question_embedding=question_embedding,
            candidates=candidates,
            rerank_mode=rerank_mode,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
        )
        selected = select_top_candidates(reranked, top_k=top_k)
    else:
        for i, candidate in enumerate(candidates, start=1):
            candidate["score"] = max(0.0, 10.0 - (i - 1) * 0.2)
            candidate["rerank_rationale"] = "Chroma retrieval order."
        selected = candidates[:top_k]

    context_blocks = [f"[{i}] {item['citation']}\n{item['document']}" for i, item in enumerate(selected, start=1)]
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
    sources = [
        {
            "id": item["id"],
            "rank": i,
            "source": item["source"],
            "metadata": item["metadata"],
            "distance": item["distance"],
            "text": item["text"],
            "citation": item["citation"],
            "snippet": item["snippet"],
            "score": round(float(item.get("score", 0.0)), 3),
            "rationale": item.get("rerank_rationale", ""),
        }
        for i, item in enumerate(selected, start=1)
    ]
    return {"answer": answer, "sources": sources}

