"""Build the disposable SQLite index used to query serialized CST nodes."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


class FactStoreError(RuntimeError):
    """Raised when CST outputs cannot be converted into a valid index."""


_REQUIRED_FIELDS = {
    "id",
    "file",
    "node_type",
    "parent_id",
    "field_name",
    "start_line",
    "start_column",
    "end_line",
    "end_column",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactStoreError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise FactStoreError("manifest must use schema_version 1")
    if manifest.get("complete") is not True:
        raise FactStoreError("manifest is not complete")
    return manifest


def _load_nodes(path: Path) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                node = json.loads(line)
                if not isinstance(node, dict) or not _REQUIRED_FIELDS <= node.keys():
                    raise FactStoreError(f"invalid CST node at {path}:{line_number}")
                nodes.append(node)
    except json.JSONDecodeError as exc:
        raise FactStoreError(f"invalid JSONL in {path}: {exc}") from exc
    except OSError as exc:
        raise FactStoreError(f"cannot read CST nodes {path}: {exc}") from exc
    return nodes


def build_fact_database(
    manifest_path: Path,
    facts_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    nodes = _load_nodes(facts_path)
    expected = manifest.get("totals", {}).get("nodes")
    if expected != len(nodes):
        raise FactStoreError(f"manifest expects {expected} nodes; found {len(nodes)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.executescript(
                """
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    file TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    parent_id TEXT,
                    field_name TEXT,
                    start_line INTEGER NOT NULL,
                    start_column INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    end_column INTEGER NOT NULL,
                    text TEXT
                );
                CREATE INDEX nodes_file ON nodes(file);
                CREATE INDEX nodes_type ON nodes(node_type);
                CREATE INDEX nodes_parent ON nodes(parent_id);
                CREATE INDEX nodes_field ON nodes(field_name);
                CREATE INDEX nodes_text ON nodes(text);
                """
            )
            connection.executemany(
                """
                INSERT INTO nodes VALUES (
                    :id, :file, :node_type, :parent_id, :field_name,
                    :start_line, :start_column, :end_line, :end_column, :text
                )
                """,
                ({**node, "text": node.get("text")} for node in nodes),
            )
            missing_parents = connection.execute(
                """
                SELECT COUNT(*) FROM nodes child
                LEFT JOIN nodes parent ON parent.id = child.parent_id
                WHERE child.parent_id IS NOT NULL AND parent.id IS NULL
                """
            ).fetchone()[0]
            if missing_parents:
                raise FactStoreError(f"found {missing_parents} missing parent nodes")
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_path, output_path)
    except sqlite3.Error as exc:
        raise FactStoreError(f"cannot build SQLite index: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "database": str(output_path.resolve()),
        "file_count": manifest["totals"]["files"],
        "node_count": len(nodes),
    }
