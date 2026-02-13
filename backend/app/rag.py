from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

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


def _fallback_keyword_query(collection: Any, question: str, n_results: int = 10) -> dict[str, list[list[Any]]]:
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    rows = collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    question_tokens = _tokenize(question)

    scored: list[tuple[int, int]] = []
    for idx, doc in enumerate(docs):
        doc_tokens = _tokenize(doc or "")
        overlap = len(question_tokens & doc_tokens)
        scored.append((overlap, idx))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_indices = [idx for _, idx in scored[:n_results]]

    top_docs = [docs[i] for i in top_indices]
    top_metas = [metas[i] if i < len(metas) else None for i in top_indices]
    # Keep distances shape compatible with query() output; lexical fallback has no vector distance.
    top_distances: list[None] = [None for _ in top_indices]
    return {"documents": [top_docs], "metadatas": [top_metas], "distances": [top_distances]}


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


def ask_rag(question: str) -> dict[str, Any]:
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
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        if "'dict' object has no attribute 'dimensionality'" in str(exc):
            result = _fallback_keyword_query(collection, question, n_results=10)
        else:
            hint = _chroma_schema_hint(exc)
            if hint:
                raise RagRuntimeError(hint) from exc
            raise

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, doc in enumerate(documents, start=1):
        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else None
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        source = _source_name(metadata, idx)
        context_blocks.append(f"[{idx}] {source}\n{doc}")

        if source not in seen:
            seen.add(source)
            sources.append(
                {
                    "source": source,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

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
