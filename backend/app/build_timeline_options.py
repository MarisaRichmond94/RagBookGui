from __future__ import annotations

import argparse
import json

from app.timeline import build_timeline_options_from_chroma


def main() -> None:
    parser = argparse.ArgumentParser(description="Build timeline filter options from Chroma.")
    parser.add_argument(
        "--books",
        type=str,
        default="",
        help="Optional comma-separated list of books to scope extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max values per option category (default: 500).",
    )
    args = parser.parse_args()

    books = [item.strip() for item in args.books.split(",") if item.strip()] or None
    options = build_timeline_options_from_chroma(
        books=books,
        limit=max(1, min(args.limit, 1000)),
        persist=books is None,
    )
    print(json.dumps(options, indent=2))


if __name__ == "__main__":
    main()
