from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(os.getenv("RAGBOOKS_ROOT", str(Path.home() / "RagBooks"))).expanduser()
BOOKS_DIR = PROJECT_ROOT / "Books"
CHROMA_DIR = PROJECT_ROOT / "ChromaDB"
BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ragbooks")
SUMMARY_COLLECTION_NAME = os.getenv("CHROMA_SUMMARY_COLLECTION", "ragbooks_chapter_summaries")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
MAX_CHARS = 2800
OVERLAP_PARAGRAPHS = 1

load_dotenv(dotenv_path=BACKEND_ENV_PATH)


def _first_non_empty(meta: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            continue
        if value is not None:
            return value
    return None


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                continue
            cleaned[key] = normalized
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    return paragraphs


def chunk_text_by_paragraphs(
    text: str, max_chars: int = MAX_CHARS, overlap_paragraphs: int = OVERLAP_PARAGRAPHS
) -> list[str]:
    paragraphs = split_paragraphs(text)
    chunks: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            chunks.append("\n\n".join(buffer).strip())

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush_buffer()
            buffer = []
            for i in range(0, len(paragraph), max_chars):
                piece = paragraph[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        candidate = "\n\n".join([*buffer, paragraph]).strip()
        if buffer and len(candidate) > max_chars:
            overlap = buffer[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            flush_buffer()
            buffer = [*overlap, paragraph]
        else:
            buffer.append(paragraph)

    flush_buffer()
    return chunks


def load_meta_for_chapter(chapter_txt: Path) -> dict[str, Any]:
    meta_path = chapter_txt.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def generate_chapter_summary(openai_client: OpenAI, chapter_text: str) -> str:
    prompt_text = chapter_text[:12000]
    response = openai_client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a concise, structured chapter summary in 8 bullets or fewer. "
                    "Cover key characters present, major events, important reveals, and timeline markers."
                ),
            },
            {
                "role": "user",
                "content": f"Chapter text:\n{prompt_text}",
            },
        ],
    )
    summary = (response.choices[0].message.content or "").strip()
    return summary


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to rebuild the index.")

    if not BOOKS_DIR.exists():
        raise SystemExit(f"Books folder not found: {BOOKS_DIR}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recreate collections each run so old and new indexing artifacts never mix.
    existing_names = {collection.name for collection in chroma_client.list_collections()}
    if COLLECTION_NAME in existing_names:
        chroma_client.delete_collection(COLLECTION_NAME)
    if SUMMARY_COLLECTION_NAME in existing_names:
        chroma_client.delete_collection(SUMMARY_COLLECTION_NAME)

    collection = chroma_client.create_collection(COLLECTION_NAME)
    summary_collection = chroma_client.create_collection(SUMMARY_COLLECTION_NAME)

    chapter_files = sorted(BOOKS_DIR.glob("*/*.txt"))
    if not chapter_files:
        raise SystemExit(f"No chapter .txt files found under: {BOOKS_DIR}")

    openai_client = OpenAI(api_key=api_key)
    total_chunks = 0
    total_summaries = 0
    indexed_chapters = 0

    for chapter_txt in chapter_files:
        if chapter_txt.name.startswith("00_"):
            continue

        text = chapter_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        indexed_chapters += 1
        book_name = chapter_txt.parent.name
        chapter_meta = load_meta_for_chapter(chapter_txt)

        base_meta: dict[str, Any] = {
            "book": book_name,
            "chapter_file": chapter_txt.name,
            "chapter_dir": str(chapter_txt.parent),
            "chapter": _first_non_empty(chapter_meta, ["chapter", "chapter_number", "chapter_num"]),
            "chapter_heading": _first_non_empty(
                chapter_meta, ["chapter_heading", "chapter_title", "heading", "title"]
            ),
            "pov": _first_non_empty(chapter_meta, ["pov", "point_of_view"]),
            "date": _first_non_empty(chapter_meta, ["date", "chapter_date", "day", "timeline_date"]),
        }
        clean_base_meta = sanitize_metadata(base_meta)

        chapter_summary = generate_chapter_summary(openai_client, text)
        summary_embedding = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=chapter_summary
        ).data[0].embedding
        summary_doc_id = f"{book_name}::{chapter_txt.name}::summary"
        summary_collection.add(
            ids=[summary_doc_id],
            documents=[chapter_summary],
            embeddings=[summary_embedding],
            metadatas=[clean_base_meta],
        )
        total_summaries += 1

        chunks = chunk_text_by_paragraphs(text, max_chars=MAX_CHARS, overlap_paragraphs=OVERLAP_PARAGRAPHS)
        for idx, chunk in enumerate(chunks):
            embedding = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=chunk).data[0].embedding

            metadata = sanitize_metadata({**clean_base_meta, "chunk_index": idx})
            doc_id = f"{book_name}::{chapter_txt.name}::{idx}"

            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            total_chunks += 1

    print(f"Indexed chapters: {indexed_chapters}")
    print(f"Chapter summaries: {total_summaries}")
    print(f"Indexed chunks:   {total_chunks}")
    print(f"Chroma path:      {CHROMA_DIR}")
    print(f"Chunk collection: {COLLECTION_NAME}")
    print(f"Summary collection: {SUMMARY_COLLECTION_NAME}")
    print("Rebuild complete.")


if __name__ == "__main__":
    main()
