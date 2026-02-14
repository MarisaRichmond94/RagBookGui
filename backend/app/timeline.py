from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

PROJECT_ROOT = Path(os.getenv("RAGBOOKS_ROOT", str(Path.home() / "RagBooks"))).expanduser()
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts"
TIMELINE_DB_PATH = ARTIFACTS_DIR / "timeline.db"
TIMELINE_OPTIONS_PATH = ARTIFACTS_DIR / "timeline_options.json"
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "ChromaDB"))).expanduser()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ragbooks")

EVENT_CUE_VERBS = (
    "revealed",
    "discovered",
    "decided",
    "found",
    "killed",
    "attacked",
    "confessed",
    "learned",
    "escaped",
    "arrived",
    "left",
    "met",
    "realized",
    "promised",
    "admitted",
    "betrayed",
)

NAME_STOPWORDS = {
    "The",
    "A",
    "An",
    "And",
    "But",
    "If",
    "When",
    "While",
    "He",
    "She",
    "They",
    "We",
    "I",
    "You",
    "It",
    "His",
    "Her",
    "Their",
    "About",
    "Chapter",
    "Later",
    "Then",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "After",
    "All",
    "Already",
    "Are",
    "Because",
    "Before",
    "Come",
    "Did",
    "Don",
    "Even",
    "Everyone",
    "Everything",
    "Nothing",
    "Something",
    "There",
    "Then",
    "This",
    "That",
    "These",
    "Those",
}

LOCATION_STOPWORDS = {
    "I",
    "He",
    "She",
    "They",
    "We",
    "It",
    "The",
    "A",
    "An",
    "And",
    "But",
    "Night",
    "Morning",
    "Evening",
    "Noon",
    "Dawn",
    "Dusk",
    "Dr",
    "Coach",
    "Chief",
    "Principal",
    "Dad",
    "Mom",
    "Grandpa",
    "Man",
    "English",
    "Spanish",
    "Korean",
    "Global",
    "October",
    "November",
    "December",
    "That",
}

CHARACTER_TITLE_STOPWORDS = {"Mr", "Mrs", "Ms", "Dr", "Coach", "Chief", "Principal"}
LOCATIONISH_TOKENS = {
    "High",
    "School",
    "Falls",
    "Coast",
    "Hills",
    "City",
    "Lake",
    "River",
    "Street",
    "Road",
    "Park",
    "Global",
    "University",
    "College",
}


def _is_plausible_character(name: str) -> bool:
    parts = [part for part in name.split() if part]
    if not parts or len(parts) > 2:
        return False
    if any(part in NAME_STOPWORDS for part in parts):
        return False
    if parts[0] in CHARACTER_TITLE_STOPWORDS:
        return False
    if any(part in LOCATIONISH_TOKENS for part in parts):
        return False
    if any(not re.match(r"^[A-Z][a-z]{1,}$", part) for part in parts):
        return False
    return True


def _normalize_list(items: list[str], max_items: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = " ".join(item.split()).strip(" .,:;!?")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _extract_locations_from_text(text: str) -> list[str]:
    matches = re.findall(
        r"(?:in|at|to|from|near|inside|outside)\s+([A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+){0,2})",
        text,
    )
    cleaned: list[str] = []
    for match in matches:
        parts = [part for part in match.split() if part]
        if not parts or len(parts) > 3:
            continue
        if parts[0] in LOCATION_STOPWORDS:
            continue
        if any(not re.match(r"^[A-Z][A-Za-z']+$", part) for part in parts):
            continue
        cleaned.append(" ".join(parts))
    return _normalize_list(cleaned, max_items=30)


def _extract_character_candidates(text: str) -> list[str]:
    full_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    matches = list(full_matches)
    return _normalize_list(
        [name for name in matches if _is_plausible_character(name)],
        max_items=120,
    )


def _normalize_book_scope(books: Optional[list[str]]) -> list[str]:
    if not books:
        return []
    return [book.strip() for book in books if book and book.strip()]


def extract_timeline_fields(text: str, pov: Optional[str] = None) -> dict[str, list[str]]:
    normalized = text.replace("\r\n", "\n").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]

    key_events: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(verb in lower for verb in EVENT_CUE_VERBS):
            key_events.append(sentence[:220])
        if len(key_events) >= 8:
            break

    if not key_events:
        key_events = [s[:220] for s in sentences[:8]]

    locations = _extract_locations_from_text(normalized)[:10]
    candidates = _extract_character_candidates(normalized)
    if pov:
        candidates.insert(0, pov)
    characters_present = _normalize_list(candidates, max_items=14)

    return {
        "key_events": _normalize_list(key_events, max_items=8),
        "locations": locations,
        "characters_present": characters_present,
    }


def init_timeline_db(reset: bool = False) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(TIMELINE_DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline_chapters (
                id TEXT PRIMARY KEY,
                book TEXT NOT NULL,
                chapter TEXT,
                chapter_file TEXT NOT NULL,
                pov TEXT,
                date_raw TEXT,
                key_events_json TEXT NOT NULL,
                locations_json TEXT NOT NULL,
                characters_present_json TEXT NOT NULL,
                chapter_ref TEXT NOT NULL
            )
            """
        )
        if reset:
            cursor.execute("DELETE FROM timeline_chapters")
        conn.commit()


def upsert_timeline_chapter(
    *,
    book: str,
    chapter: Any,
    chapter_file: str,
    pov: Optional[str],
    date_raw: Optional[str],
    key_events: list[str],
    locations: list[str],
    characters_present: list[str],
) -> None:
    chapter_ref = f"{book}/{chapter_file}"
    row_id = f"{book}::{chapter_file}"

    with sqlite3.connect(str(TIMELINE_DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO timeline_chapters (
                id, book, chapter, chapter_file, pov, date_raw,
                key_events_json, locations_json, characters_present_json, chapter_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                book = excluded.book,
                chapter = excluded.chapter,
                chapter_file = excluded.chapter_file,
                pov = excluded.pov,
                date_raw = excluded.date_raw,
                key_events_json = excluded.key_events_json,
                locations_json = excluded.locations_json,
                characters_present_json = excluded.characters_present_json,
                chapter_ref = excluded.chapter_ref
            """,
            (
                row_id,
                book,
                str(chapter) if chapter is not None else None,
                chapter_file,
                pov,
                date_raw,
                json.dumps(key_events),
                json.dumps(locations),
                json.dumps(characters_present),
                chapter_ref,
            ),
        )
        conn.commit()


def _parse_json_array(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []
    except Exception:
        return []


def search_timeline(
    *,
    character: Optional[str] = None,
    date_substring: Optional[str] = None,
    location: Optional[str] = None,
    books: Optional[list[str]] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not TIMELINE_DB_PATH.exists():
        return []

    query = "SELECT book, chapter, chapter_file, pov, date_raw, key_events_json, locations_json, characters_present_json, chapter_ref FROM timeline_chapters"
    params: list[Any] = []
    where: list[str] = []

    if books:
        placeholders = ",".join("?" for _ in books)
        where.append(f"book IN ({placeholders})")
        params.extend(books)
    if date_substring:
        where.append("LOWER(COALESCE(date_raw, '')) LIKE ?")
        params.append(f"%{date_substring.lower()}%")

    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY book, chapter_file LIMIT ?"
    params.append(limit)

    out: list[dict[str, Any]] = []
    with sqlite3.connect(str(TIMELINE_DB_PATH)) as conn:
        cursor = conn.cursor()
        rows = cursor.execute(query, params).fetchall()
        for row in rows:
            key_events = _parse_json_array(row[5])
            locations = _parse_json_array(row[6])
            characters = _parse_json_array(row[7])

            if character and not any(character.lower() in c.lower() for c in characters):
                continue
            if location and not any(location.lower() in loc.lower() for loc in locations):
                continue

            out.append(
                {
                    "book": row[0],
                    "chapter": row[1],
                    "chapter_file": row[2],
                    "pov": row[3],
                    "date_raw": row[4],
                    "key_events": key_events,
                    "locations": locations,
                    "characters_present": characters,
                    "chapter_ref": row[8],
                }
            )

    return out


def _normalize_option_values(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = " ".join(str(raw).split()).strip(" .,:;!?")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return sorted(out, key=lambda item: item.lower())


def _resolve_character_names(values: list[str], limit: int) -> list[str]:
    normalized = _normalize_option_values(values, limit=5000)
    full_names_by_first: dict[str, set[str]] = {}
    for value in normalized:
        parts = value.split()
        if len(parts) >= 2:
            full_names_by_first.setdefault(parts[0].lower(), set()).add(" ".join(parts[:2]))

    resolved: list[str] = []
    for value in normalized:
        parts = value.split()
        if len(parts) == 1:
            candidates = full_names_by_first.get(parts[0].lower(), set())
            if len(candidates) == 1:
                resolved.append(next(iter(candidates)))
                continue
        resolved.append(value)
    return _normalize_option_values(resolved, limit)


def _timeline_options_from_sqlite(
    *, books: Optional[list[str]] = None, limit: int = 500
) -> dict[str, list[str]]:
    if not TIMELINE_DB_PATH.exists():
        return {"characters": [], "dates": [], "locations": []}

    scoped_books = _normalize_book_scope(books)
    query = (
        "SELECT date_raw, locations_json, characters_present_json "
        "FROM timeline_chapters"
    )
    params: list[Any] = []
    if scoped_books:
        placeholders = ",".join("?" for _ in scoped_books)
        query += f" WHERE book IN ({placeholders})"
        params.extend(scoped_books)

    dates_raw: list[str] = []
    locations_raw: list[str] = []
    characters_raw: list[str] = []

    with sqlite3.connect(str(TIMELINE_DB_PATH)) as conn:
        cursor = conn.cursor()
        rows = cursor.execute(query, params).fetchall()
        for row in rows:
            if row[0]:
                dates_raw.append(str(row[0]))
            locations_raw.extend(_parse_json_array(row[1] or "[]"))
            characters_raw.extend(_parse_json_array(row[2] or "[]"))

    safe_limit = max(1, min(limit, 1000))
    filtered_character_values = [
        " ".join(str(value).split()).strip()
        for value in characters_raw
        if _is_plausible_character(" ".join(str(value).split()).strip())
    ]
    return {
        "characters": _resolve_character_names(filtered_character_values, safe_limit),
        "dates": _normalize_option_values(dates_raw, safe_limit),
        "locations": _normalize_option_values(locations_raw, safe_limit),
    }


def _persist_timeline_options(options: dict[str, list[str]]) -> None:
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "characters": options.get("characters", []),
            "dates": options.get("dates", []),
            "locations": options.get("locations", []),
        }
        TIMELINE_OPTIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Cache persistence is best-effort; callers still receive computed options.
        return


def _timeline_options_from_cached_json(limit: int = 500) -> dict[str, list[str]]:
    if not TIMELINE_OPTIONS_PATH.exists():
        return {"characters": [], "dates": [], "locations": []}
    try:
        payload = json.loads(TIMELINE_OPTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"characters": [], "dates": [], "locations": []}
    safe_limit = max(1, min(limit, 1000))
    return {
        "characters": _resolve_character_names(
            [str(v) for v in (payload.get("characters") or [])], safe_limit
        ),
        "dates": _normalize_option_values([str(v) for v in (payload.get("dates") or [])], safe_limit),
        "locations": _normalize_option_values(
            [str(v) for v in (payload.get("locations") or [])], safe_limit
        ),
    }


def build_timeline_options_from_chroma(
    *, books: Optional[list[str]] = None, limit: int = 500, persist: bool = True
) -> dict[str, list[str]]:
    scoped_books = set(_normalize_book_scope(books))
    if not CHROMA_PATH.exists():
        return {"characters": [], "dates": [], "locations": []}

    try:
        import chromadb
    except Exception:
        return {"characters": [], "dates": [], "locations": []}

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return {"characters": [], "dates": [], "locations": []}

    total = collection.count()
    if total <= 0:
        return {"characters": [], "dates": [], "locations": []}

    batch_size = 250
    chapter_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        row_count = max(len(documents), len(metadatas))

        for idx in range(row_count):
            meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            doc = documents[idx] if idx < len(documents) and isinstance(documents[idx], str) else ""

            book_value = meta.get("book")
            if scoped_books:
                if not isinstance(book_value, str) or book_value not in scoped_books:
                    continue

            if not isinstance(book_value, str) or not book_value.strip():
                continue

            chapter_file = meta.get("chapter_file")
            if not isinstance(chapter_file, str) or not chapter_file.strip():
                continue

            key = (book_value, chapter_file)
            row = chapter_rows.setdefault(key, {"date": None, "pov": None, "chunks": []})

            date_value = meta.get("date")
            if row["date"] is None and isinstance(date_value, str) and date_value.strip():
                row["date"] = date_value.strip()

            pov_value = meta.get("pov")
            if row["pov"] is None and isinstance(pov_value, str) and pov_value.strip():
                row["pov"] = pov_value.strip()

            chunk_index = meta.get("chunk_index")
            try:
                order = int(chunk_index) if chunk_index is not None else len(row["chunks"])
            except Exception:
                order = len(row["chunks"])
            row["chunks"].append((order, doc))

    safe_limit = max(1, min(limit, 1000))
    dates_raw: list[str] = []
    locations_raw: list[str] = []
    characters_raw: list[str] = []

    for row in chapter_rows.values():
        chapter_date = row.get("date")
        chapter_pov = row.get("pov")
        chunks: list[tuple[int, str]] = row.get("chunks") or []

        if isinstance(chapter_date, str) and chapter_date:
            dates_raw.append(chapter_date)
        if isinstance(chapter_pov, str) and chapter_pov:
            characters_raw.append(chapter_pov)

        chapter_text = "\n\n".join(
            text for _, text in sorted(chunks, key=lambda item: item[0]) if isinstance(text, str)
        )
        if chapter_text:
            extracted = extract_timeline_fields(
                chapter_text,
                pov=chapter_pov if isinstance(chapter_pov, str) else None,
            )
            locations_raw.extend(extracted.get("locations", []))

    character_counts: Counter[str] = Counter(
        " ".join(str(value).split()).strip()
        for value in characters_raw
        if _is_plausible_character(" ".join(str(value).split()).strip())
    )
    filtered_characters = [name for name, count in character_counts.items() if count >= 2]
    canonical_characters = _resolve_character_names(filtered_characters, safe_limit)

    character_tokens = {token.lower() for name in canonical_characters for token in name.split()}
    location_counts: Counter[str] = Counter(
        " ".join(str(value).split()).strip() for value in locations_raw if str(value).strip()
    )
    frequent_locations = [name for name, count in location_counts.items() if count >= 2]
    filtered_locations = [
        location
        for location in _normalize_option_values(frequent_locations, safe_limit * 3)
        if (" " in location) or (location.lower() not in character_tokens)
    ]
    options = {
        "characters": canonical_characters,
        "dates": _normalize_option_values(dates_raw, safe_limit),
        "locations": _normalize_option_values(filtered_locations, safe_limit),
    }
    if persist and not scoped_books:
        _persist_timeline_options(options)
    return options


def get_timeline_options(
    *, books: Optional[list[str]] = None, limit: int = 500
) -> dict[str, list[str]]:
    scoped_books = _normalize_book_scope(books)
    sqlite_options = _timeline_options_from_sqlite(books=scoped_books, limit=limit)
    if any(sqlite_options.values()):
        return sqlite_options

    if scoped_books:
        return build_timeline_options_from_chroma(books=scoped_books, limit=limit, persist=False)

    cached_options = _timeline_options_from_cached_json(limit=limit)
    if any(cached_options.values()):
        return cached_options

    return build_timeline_options_from_chroma(limit=limit, persist=True)
