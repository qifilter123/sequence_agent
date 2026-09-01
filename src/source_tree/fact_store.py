"""Build and query a bounded SQLite index over deterministic source facts.

``facts.jsonl`` remains the audit-friendly source of truth.  The SQLite file is
a disposable, read-only query index for agents and can always be rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATABASE_SCHEMA_VERSION = 1
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = {"1.0"}
SUPPORTED_FACT_KINDS = ("module", "import", "symbol", "call", "assignment")
DEFAULT_QUERY_LIMIT = 100
MAX_QUERY_LIMIT = 500


SCHEMA_SQL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE source_files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    line_count INTEGER NOT NULL CHECK (line_count >= 0),
    sha256 TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    fact_counts_json TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    node_type TEXT NOT NULL,
    scope TEXT,
    parent TEXT,
    name TEXT,
    qualified_name TEXT,
    symbol_kind TEXT,
    target TEXT,
    callee TEXT,
    caller TEXT,
    import_module TEXT,
    start_line INTEGER NOT NULL CHECK (start_line >= 1),
    start_column INTEGER NOT NULL CHECK (start_column >= 0),
    end_line INTEGER NOT NULL CHECK (end_line >= 1),
    end_column INTEGER NOT NULL CHECK (end_column >= 0),
    raw_json TEXT NOT NULL,
    FOREIGN KEY (file_path) REFERENCES source_files(path)
);

CREATE TABLE fact_arguments (
    fact_id TEXT NOT NULL,
    argument_ordinal INTEGER NOT NULL CHECK (argument_ordinal >= 0),
    position INTEGER,
    keyword TEXT,
    value_syntax TEXT,
    value_text TEXT NOT NULL,
    PRIMARY KEY (fact_id, argument_ordinal),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX idx_facts_file_kind
    ON facts(file_path, kind);
CREATE INDEX idx_facts_scope_kind
    ON facts(scope, kind);
CREATE INDEX idx_facts_qualified_name
    ON facts(qualified_name);
CREATE INDEX idx_facts_name_kind
    ON facts(name, kind);
CREATE INDEX idx_facts_target
    ON facts(target);
CREATE INDEX idx_facts_callee
    ON facts(callee);
CREATE INDEX idx_facts_caller
    ON facts(caller);
CREATE INDEX idx_facts_import_module
    ON facts(import_module);
CREATE INDEX idx_fact_arguments_value
    ON fact_arguments(value_text, fact_id);
"""


class FactStoreError(RuntimeError):
    """Raised when the fact index cannot be built or queried safely."""


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.resolve(strict=True).read_bytes()
    except OSError as exc:
        raise FactStoreError(f"cannot read {label} {path}: {exc}") from exc


def _load_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactStoreError(f"invalid UTF-8 JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FactStoreError(f"{label} root must be a JSON object")
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactStoreError(f"{label} must be a non-negative integer")
    return value


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise FactStoreError(f"{label} must be {qualifier}")
    return value


def _optional_index_text(value: Any) -> str | None:
    # Long extractor values are deterministic summary objects.  They remain in
    # raw_json but are not indexed as if their preview were the complete value.
    return value if isinstance(value, str) else None


def _validate_fact_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise FactStoreError(f"{label} must be an object")
    unknown = sorted(set(value) - set(SUPPORTED_FACT_KINDS))
    if unknown:
        raise FactStoreError(f"{label} contains unsupported fact kinds: {unknown}")
    return {
        kind: _require_non_negative_int(value.get(kind, 0), f"{label}.{kind}")
        for kind in SUPPORTED_FACT_KINDS
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    if manifest.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise FactStoreError(
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}"
        )
    if manifest.get("complete") is not True:
        raise FactStoreError("manifest complete must be true")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FactStoreError("manifest.files must be a non-empty array")

    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    expected_totals: Counter[str] = Counter()
    for index, raw in enumerate(raw_files):
        label = f"manifest.files[{index}]"
        if not isinstance(raw, dict):
            raise FactStoreError(f"{label} must be an object")
        path = _require_string(raw.get("path"), f"{label}.path")
        if path in seen_paths:
            raise FactStoreError(f"duplicate manifest file path: {path}")
        seen_paths.add(path)
        parse = raw.get("parse")
        if not isinstance(parse, dict) or parse.get("status") != "ok":
            raise FactStoreError(f"{path}: parse.status must be ok")
        if _require_non_negative_int(parse.get("error_nodes"), f"{path}.parse.error_nodes"):
            raise FactStoreError(f"{path}: parse.error_nodes must be zero")
        if _require_non_negative_int(parse.get("missing_nodes"), f"{path}.parse.missing_nodes"):
            raise FactStoreError(f"{path}: parse.missing_nodes must be zero")
        sha256 = _require_string(raw.get("sha256"), f"{path}.sha256")
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise FactStoreError(f"{path}.sha256 must be a lowercase SHA-256 hex digest")
        counts = _validate_fact_counts(raw.get("facts"), f"{path}.facts")
        if counts["module"] != 1:
            raise FactStoreError(f"{path}.facts.module must be exactly one")
        expected_totals.update(counts)
        files.append(
            {
                "path": path,
                "language": _require_string(raw.get("language"), f"{path}.language"),
                "bytes": _require_non_negative_int(raw.get("bytes"), f"{path}.bytes"),
                "lines": _require_non_negative_int(raw.get("lines"), f"{path}.lines"),
                "sha256": sha256,
                "parse_status": "ok",
                "facts": counts,
            }
        )

    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise FactStoreError("manifest.totals must be an object")
    if _require_non_negative_int(totals.get("files"), "manifest.totals.files") != len(files):
        raise FactStoreError("manifest.totals.files does not match manifest.files")
    if _require_non_negative_int(
        totals.get("parse_error_files"), "manifest.totals.parse_error_files"
    ) != 0:
        raise FactStoreError("manifest.totals.parse_error_files must be zero")
    declared_totals = _validate_fact_counts(totals.get("facts"), "manifest.totals.facts")
    if declared_totals != {kind: expected_totals[kind] for kind in SUPPORTED_FACT_KINDS}:
        raise FactStoreError("manifest total fact counts do not match per-file counts")
    return files, expected_totals


def _validate_position(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise FactStoreError(f"{label} must be [line, column]")
    line = _require_non_negative_int(value[0], f"{label}[0]")
    column = _require_non_negative_int(value[1], f"{label}[1]")
    if line < 1:
        raise FactStoreError(f"{label} line must be one-based")
    return line, column


def _is_text_or_summary(value: Any) -> bool:
    return isinstance(value, str) or isinstance(value, dict)


def _validate_fact_shape(fact: Mapping[str, Any], kind: str, label: str) -> None:
    if kind == "module":
        return
    if kind == "import":
        if not _is_text_or_summary(fact.get("statement")):
            raise FactStoreError(f"{label}.statement must be text or a bounded summary")
        if not isinstance(fact.get("names"), list):
            raise FactStoreError(f"{label}.names must be an array")
        return
    if kind == "symbol":
        _require_string(fact.get("name"), f"{label}.name")
        _require_string(fact.get("qualified_name"), f"{label}.qualified_name")
        symbol_kind = _require_string(fact.get("symbol_kind"), f"{label}.symbol_kind")
        if symbol_kind not in {"class", "function", "method"}:
            raise FactStoreError(f"{label}.symbol_kind is unsupported: {symbol_kind}")
        parent = fact.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise FactStoreError(f"{label}.parent must be a string or null")
        for key in ("bases", "decorators", "parameters"):
            if key in fact and not isinstance(fact[key], list):
                raise FactStoreError(f"{label}.{key} must be an array")
        return
    if kind == "call":
        if not _is_text_or_summary(fact.get("callee")):
            raise FactStoreError(f"{label}.callee must be text or a bounded summary")
        _require_string(fact.get("caller"), f"{label}.caller")
        if not isinstance(fact.get("arguments"), list):
            raise FactStoreError(f"{label}.arguments must be an array")
        return
    if kind == "assignment":
        _require_string(fact.get("scope"), f"{label}.scope")
        if not _is_text_or_summary(fact.get("target")):
            raise FactStoreError(f"{label}.target must be text or a bounded summary")
        value = fact.get("value")
        if value is not None and not isinstance(value, dict):
            raise FactStoreError(f"{label}.value must be an object or null")


def _fact_row(
    fact: Mapping[str, Any],
    *,
    line_number: int,
    known_files: set[str],
) -> tuple[Any, ...]:
    label = f"facts.jsonl line {line_number}"
    fact_id = _require_string(fact.get("id"), f"{label}.id")
    file_path = _require_string(fact.get("file"), f"{label}.file")
    if file_path not in known_files:
        raise FactStoreError(f"{label}: file is absent from manifest: {file_path}")
    kind = _require_string(fact.get("kind"), f"{label}.kind")
    if kind not in SUPPORTED_FACT_KINDS:
        raise FactStoreError(f"{label}: unsupported fact kind: {kind}")
    node_type = _require_string(fact.get("node_type"), f"{label}.node_type")
    if not fact_id.startswith(f"{file_path}:"):
        raise FactStoreError(f"{label}.id must be namespaced by its file path")
    if kind == "module" and fact_id != f"{file_path}:module":
        raise FactStoreError(f"{label}: module id must be <file>:module")
    _validate_fact_shape(fact, kind, label)
    span = fact.get("span")
    if not isinstance(span, dict):
        raise FactStoreError(f"{label}.span must be an object")
    start_line, start_column = _validate_position(span.get("start"), f"{label}.span.start")
    end_line, end_column = _validate_position(span.get("end"), f"{label}.span.end")
    if (end_line, end_column) < (start_line, start_column):
        raise FactStoreError(f"{label}.span end precedes start")

    return (
        fact_id,
        file_path,
        kind,
        node_type,
        _optional_index_text(fact.get("scope")),
        _optional_index_text(fact.get("parent")),
        _optional_index_text(fact.get("name")),
        _optional_index_text(fact.get("qualified_name")),
        _optional_index_text(fact.get("symbol_kind")),
        _optional_index_text(fact.get("target")),
        _optional_index_text(fact.get("callee")),
        _optional_index_text(fact.get("caller")),
        _optional_index_text(fact.get("module")),
        start_line,
        start_column,
        end_line,
        end_column,
        _compact_json(fact),
    )


INSERT_FACT_SQL = """
INSERT INTO facts (
    id, file_path, kind, node_type, scope, parent, name, qualified_name,
    symbol_kind, target, callee, caller, import_module,
    start_line, start_column, end_line, end_column, raw_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _argument_rows(fact: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    """Extract exact scalar argument terms without changing the source fact."""

    arguments = fact.get("arguments")
    if not isinstance(arguments, list):
        return []
    rows: list[tuple[Any, ...]] = []
    for ordinal, argument in enumerate(arguments):
        if not isinstance(argument, dict):
            continue
        value = argument.get("value")
        if not isinstance(value, dict):
            continue
        value_text: str | None = None
        for key in ("value", "name", "callee", "text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                value_text = candidate
                break
        if value_text is None:
            continue
        position = argument.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            position = None
        rows.append(
            (
                fact["id"],
                ordinal,
                position,
                _optional_index_text(argument.get("keyword")),
                _optional_index_text(value.get("syntax")),
                value_text,
            )
        )
    return rows


def _iter_fact_rows(
    facts_data: bytes,
    known_files: set[str],
) -> Iterable[tuple[int, dict[str, Any], tuple[Any, ...]]]:
    try:
        text = facts_data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FactStoreError(f"facts.jsonl is not valid UTF-8: {exc}") from exc
    if not text:
        raise FactStoreError("facts.jsonl must not be empty")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise FactStoreError(f"facts.jsonl line {line_number} is blank")
        try:
            fact = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FactStoreError(f"invalid JSON on facts.jsonl line {line_number}: {exc}") from exc
        if not isinstance(fact, dict):
            raise FactStoreError(f"facts.jsonl line {line_number} must be a JSON object")
        yield line_number, fact, _fact_row(
            fact, line_number=line_number, known_files=known_files
        )


def _metadata(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM metadata"))


def build_fact_database(
    manifest_path: Path,
    facts_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate manifest/facts and atomically rebuild the SQLite query index."""

    manifest_resolved = manifest_path.resolve(strict=True)
    facts_resolved = facts_path.resolve(strict=True)
    output = output_path.resolve()
    if manifest_resolved == facts_resolved:
        raise FactStoreError("manifest and facts must be different files")
    if output in {manifest_resolved, facts_resolved}:
        raise FactStoreError("SQLite output must not overwrite manifest or facts")
    manifest_data = _read_bytes(manifest_path, "manifest")
    facts_data = _read_bytes(facts_path, "facts")
    manifest = _load_json_object(manifest_data, "manifest")
    files, expected_totals = _validate_manifest(manifest)
    known_files = {item["path"] for item in files}

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    actual_counts: Counter[str] = Counter()
    file_counts: Counter[tuple[str, str]] = Counter()
    seen_ids: set[str] = set()

    try:
        conn = sqlite3.connect(temporary)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.executescript(SCHEMA_SQL)
            with conn:
                conn.executemany(
                    """
                    INSERT INTO source_files (
                        path, language, byte_count, line_count, sha256,
                        parse_status, fact_counts_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item["path"],
                            item["language"],
                            item["bytes"],
                            item["lines"],
                            item["sha256"],
                            item["parse_status"],
                            _compact_json(item["facts"]),
                        )
                        for item in files
                    ],
                )
                for line_number, fact, row in _iter_fact_rows(facts_data, known_files):
                    fact_id = row[0]
                    if fact_id in seen_ids:
                        raise FactStoreError(
                            f"duplicate fact id on facts.jsonl line {line_number}: {fact_id}"
                        )
                    seen_ids.add(fact_id)
                    actual_counts[fact["kind"]] += 1
                    file_counts[(fact["file"], fact["kind"])] += 1
                    conn.execute(INSERT_FACT_SQL, row)
                    argument_rows = _argument_rows(fact)
                    if argument_rows:
                        conn.executemany(
                            """
                            INSERT INTO fact_arguments (
                                fact_id, argument_ordinal, position, keyword,
                                value_syntax, value_text
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            argument_rows,
                        )

                for item in files:
                    actual = {
                        kind: file_counts[(item["path"], kind)]
                        for kind in SUPPORTED_FACT_KINDS
                    }
                    if actual != item["facts"]:
                        raise FactStoreError(
                            f"{item['path']}: JSONL fact counts do not match manifest; "
                            f"expected {item['facts']}, got {actual}"
                        )
                if any(actual_counts[kind] != expected_totals[kind] for kind in SUPPORTED_FACT_KINDS):
                    raise FactStoreError("JSONL total fact counts do not match manifest")

                metadata = {
                    "database_schema_version": str(DATABASE_SCHEMA_VERSION),
                    "manifest_schema_version": str(manifest["schema_version"]),
                    "manifest_sha256": _sha256(manifest_data),
                    "facts_sha256": _sha256(facts_data),
                    "fact_count": str(sum(actual_counts.values())),
                    "file_count": str(len(files)),
                }
                conn.executemany(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    sorted(metadata.items()),
                )
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check != ("ok",):
                raise FactStoreError(f"SQLite quick_check failed: {quick_check}")
        finally:
            conn.close()
        os.replace(temporary, output)
    except (OSError, sqlite3.Error) as exc:
        raise FactStoreError(f"cannot build SQLite fact index: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "complete": True,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "files": len(files),
        "facts": {
            kind: actual_counts[kind] for kind in SUPPORTED_FACT_KINDS
        },
        "total_facts": sum(actual_counts.values()),
    }


@dataclass(frozen=True)
class QueryRequest:
    """Constrained, AND-combined fact filters accepted from an agent."""

    file: str | None = None
    file_prefix: str | None = None
    kinds: tuple[str, ...] = field(default_factory=tuple)
    node_type: str | None = None
    scope: str | None = None
    parent: str | None = None
    name: str | None = None
    qualified_name: str | None = None
    symbol_kind: str | None = None
    target: str | None = None
    callee: str | None = None
    caller: str | None = None
    module: str | None = None
    argument: str | None = None
    limit: int = DEFAULT_QUERY_LIMIT
    offset: int = 0

    def validate(self) -> None:
        selectors = (
            self.file,
            self.file_prefix,
            self.kinds,
            self.node_type,
            self.scope,
            self.parent,
            self.name,
            self.qualified_name,
            self.symbol_kind,
            self.target,
            self.callee,
            self.caller,
            self.module,
            self.argument,
        )
        if not any(selectors):
            raise FactStoreError("at least one fact filter is required")
        if self.file is not None and self.file_prefix is not None:
            raise FactStoreError("file and file_prefix cannot be combined")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise FactStoreError("limit must be an integer")
        if self.limit < 1 or self.limit > MAX_QUERY_LIMIT:
            raise FactStoreError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise FactStoreError("offset must be a non-negative integer")
        unknown = sorted(set(self.kinds) - set(SUPPORTED_FACT_KINDS))
        if unknown:
            raise FactStoreError(f"unsupported fact kinds: {unknown}")
        for field_name in (
            "file",
            "file_prefix",
            "node_type",
            "scope",
            "parent",
            "name",
            "qualified_name",
            "symbol_kind",
            "target",
            "callee",
            "caller",
            "module",
            "argument",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise FactStoreError(f"{field_name} must be a non-empty string")

    def public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in (
            "file",
            "file_prefix",
            "node_type",
            "scope",
            "parent",
            "name",
            "qualified_name",
            "symbol_kind",
            "target",
            "callee",
            "caller",
            "module",
            "argument",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.kinds:
            result["kinds"] = list(self.kinds)
        result["limit"] = self.limit
        result["offset"] = self.offset
        return result


def _read_only_connection(database_path: Path) -> sqlite3.Connection:
    try:
        resolved = database_path.resolve(strict=True)
        conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        return conn
    except (OSError, sqlite3.Error) as exc:
        raise FactStoreError(f"cannot open SQLite fact index read-only: {exc}") from exc


def _validate_index_freshness(
    conn: sqlite3.Connection,
    manifest_path: Path,
) -> dict[str, str]:
    try:
        metadata = _metadata(conn)
    except sqlite3.Error as exc:
        raise FactStoreError(f"invalid SQLite fact index metadata: {exc}") from exc
    required = {
        "database_schema_version",
        "manifest_schema_version",
        "manifest_sha256",
        "facts_sha256",
        "fact_count",
        "file_count",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise FactStoreError(f"SQLite fact index metadata is incomplete: {missing}")
    if metadata["database_schema_version"] != str(DATABASE_SCHEMA_VERSION):
        raise FactStoreError("unsupported SQLite fact index schema version")
    manifest_data = _read_bytes(manifest_path, "manifest")
    if _sha256(manifest_data) != metadata["manifest_sha256"]:
        raise FactStoreError("SQLite fact index is stale; rebuild it from the current manifest/facts")
    return metadata


def _prefix_upper_bound(prefix: str) -> str | None:
    # Transform prefix matching into an index-friendly half-open range without
    # exposing SQL wildcard behavior to the query language.
    points = [ord(ch) for ch in prefix]
    for index in range(len(points) - 1, -1, -1):
        if points[index] < 0x10FFFF:
            points[index] += 1
            return "".join(chr(point) for point in points[: index + 1])
    return None


def _where_clause(request: QueryRequest) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    exact_columns = {
        "file": "file_path",
        "node_type": "node_type",
        "scope": "scope",
        "parent": "parent",
        "name": "name",
        "qualified_name": "qualified_name",
        "symbol_kind": "symbol_kind",
        "target": "target",
        "callee": "callee",
        "caller": "caller",
        "module": "import_module",
    }
    for field_name, column in exact_columns.items():
        value = getattr(request, field_name)
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    if request.file_prefix is not None:
        clauses.append("file_path >= ?")
        parameters.append(request.file_prefix)
        upper = _prefix_upper_bound(request.file_prefix)
        if upper is not None:
            clauses.append("file_path < ?")
            parameters.append(upper)
    if request.kinds:
        placeholders = ",".join("?" for _ in request.kinds)
        clauses.append(f"kind IN ({placeholders})")
        parameters.extend(request.kinds)
    if request.argument is not None:
        clauses.append(
            "EXISTS ("
            "SELECT 1 FROM fact_arguments "
            "WHERE fact_arguments.fact_id = facts.id "
            "AND fact_arguments.value_text = ?"
            ")"
        )
        parameters.append(request.argument)
    return " AND ".join(clauses), parameters


def query_facts(
    database_path: Path,
    manifest_path: Path,
    request: QueryRequest,
) -> dict[str, Any]:
    """Return a bounded, stable, exact-match query response."""

    request.validate()
    conn = _read_only_connection(database_path)
    try:
        metadata = _validate_index_freshness(conn, manifest_path)
        where_sql, parameters = _where_clause(request)
        try:
            matched_row = conn.execute(
                f"SELECT COUNT(*) FROM facts WHERE {where_sql}", parameters
            ).fetchone()
            if matched_row is None:
                raise FactStoreError("fact count query returned no row")
            matched = int(matched_row[0])
            rows = conn.execute(
                f"""
                SELECT raw_json
                FROM facts
                WHERE {where_sql}
                ORDER BY file_path, start_line, start_column,
                         end_line, end_column, kind, id
                LIMIT ? OFFSET ?
                """,
                [*parameters, request.limit, request.offset],
            ).fetchall()
        except sqlite3.Error as exc:
            raise FactStoreError(f"SQLite fact query failed: {exc}") from exc
    finally:
        conn.close()

    facts = [json.loads(row[0]) for row in rows]
    returned = len(facts)
    truncated = request.offset + returned < matched
    response: dict[str, Any] = {
        "schema_version": 1,
        "database_schema_version": int(metadata["database_schema_version"]),
        "query": request.public_dict(),
        "matched": matched,
        "returned": returned,
        "truncated": truncated,
        "facts": facts,
    }
    if truncated:
        response["next_offset"] = request.offset + returned
    return response
