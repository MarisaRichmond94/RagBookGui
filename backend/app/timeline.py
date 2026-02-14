from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
from typing import Optional

PROJECT_ROOT = Path(os.getenv("RAGBOOKS_ROOT", str(Path.home() / "RagBooks"))).expanduser()
ARTIFACTS_DIR = PROJECT_ROOT / "Artifacts"
TIMELINE_DB_PATH = ARTIFACTS_DIR / "timeline.db"

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
}


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

    location_matches = re.findall(
        r"(?:in|at|to|from|near|inside|outside)\s+([A-Z][A-Za-z' -]{2,40})",
        normalized,
    )
    locations = _normalize_list(location_matches, max_items=10)

    name_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", normalized)
    candidates = [name for name in name_matches if name.split()[0] not in NAME_STOPWORDS]
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


def get_timeline_options(
    *, books: Optional[list[str]] = None, limit: int = 500
) -> dict[str, list[str]]:
    if not TIMELINE_DB_PATH.exists():
        return {"characters": [], "dates": [], "locations": []}

    query = (
        "SELECT date_raw, locations_json, characters_present_json "
        "FROM timeline_chapters"
    )
    params: list[Any] = []
    if books:
        placeholders = ",".join("?" for _ in books)
        query += f" WHERE book IN ({placeholders})"
        params.extend(books)

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
    return {
        "characters": _normalize_option_values(characters_raw, safe_limit),
        "dates": _normalize_option_values(dates_raw, safe_limit),
        "locations": _normalize_option_values(locations_raw, safe_limit),
    }
