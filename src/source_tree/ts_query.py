"""Run bounded queries against the serialized Tree-sitter CST index."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


SOURCE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = SOURCE_ROOT / "generated" / "ts_source_tree" / "facts.sqlite"


def query_nodes(
    database: Path,
    *,
    node_id: str | None = None,
    parent_id: str | None = None,
    file: str | None = None,
    file_prefix: str | None = None,
    node_type: str | None = None,
    field_name: str | None = None,
    text: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")

    clauses: list[str] = []
    parameters: list[Any] = []
    for column, value in (
        ("id", node_id),
        ("parent_id", parent_id),
        ("file", file),
        ("node_type", node_type),
        ("field_name", field_name),
        ("text", text),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)

    if file_prefix is not None:
        clauses.append("file LIKE ? ESCAPE '\\'")
        escaped_prefix = (
            file_prefix.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        parameters.append(escaped_prefix + "%")

    if not clauses:
        raise ValueError("at least one query filter is required")

    sql = "SELECT * FROM nodes WHERE " + " AND ".join(clauses)
    sql += " ORDER BY file, start_line, start_column, id LIMIT ?"
    parameters.append(limit + 1)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(sql, parameters)]
    finally:
        connection.close()

    return {
        "nodes": rows[:limit],
        "count": min(len(rows), limit),
        "truncated": len(rows) > limit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--id", dest="node_id")
    parser.add_argument("--parent-id")
    parser.add_argument("--file")
    parser.add_argument("--file-prefix")
    parser.add_argument("--node-type")
    parser.add_argument("--field-name")
    parser.add_argument("--text")
    parser.add_argument("--limit", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = query_nodes(
            args.database,
            node_id=args.node_id,
            parent_id=args.parent_id,
            file=args.file,
            file_prefix=args.file_prefix,
            node_type=args.node_type,
            field_name=args.field_name,
            text=args.text,
            limit=args.limit,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
