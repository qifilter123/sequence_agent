# Deterministic Source Tree for LLM Agents

This package parses allowlisted source files with Tree-sitter and emits a
compact, deterministic fact index. Tree-sitter builds a concrete syntax tree
without executing imports, requiring a working application runtime, or asking
an LLM to infer structure.

Tree-sitter does not resolve runtime dispatch, dynamic imports, monkey patching,
or the actual target of every call. Those relationships must be confirmed by
executable DAG definitions or narrowly selected source evidence; otherwise they
remain unresolved.

SQLite provides a disposable, read-only query index over the generated facts.
It lets an agent retrieve bounded facts without loading the full JSONL file into
its context.

## Build source facts

Install the Tree-sitter dependencies listed by this package, then run from the
repository `src` folder:

```powershell
python -m pip install -r .\source_tree\requirements-source-tree.txt

python -m source_tree.build_source_tree `
  --project-root .. `
  --config .\source_tree\config\source_tree.json `
  --output .\source_tree\generated
```

The builder fails closed for empty include matches, unsupported extensions,
escaping paths, invalid UTF-8, missing grammars, ambiguous language mappings,
and Tree-sitter `ERROR` or `MISSING` nodes. Outputs contain stable relative
paths and source locations, with no timestamps or absolute project paths.

## Generated files

```text
src/source_tree/generated/
├── manifest.json
├── facts.jsonl
└── facts.sqlite
```

- `manifest.json` records input identities, parse completeness, and counts.
- `facts.jsonl` remains the deterministic, audit-friendly source of truth.
- `facts.sqlite` is derived from those two files and can always be rebuilt.

`manifest.json` must report `complete: true` before either the SQLite indexer or
the context agent consumes the facts.

Python includes the `sqlite3` module; no SQLite server or extra Python package
is required.  Verify the bundled version with:

```powershell
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

## Build the index

After `build_source_tree` succeeds, run this from the repository `src` folder:

```powershell
python -m source_tree.build_fact_index
```

Explicit paths are also supported:

```powershell
python -m source_tree.build_fact_index `
  --manifest .\source_tree\generated\manifest.json `
  --facts .\source_tree\generated\facts.jsonl `
  --output .\source_tree\generated\facts.sqlite
```

The loader fails closed for an incomplete manifest, partial parse, malformed or
duplicate fact, unknown file/kind, invalid span, or any fact-count mismatch. It
builds a temporary database and atomically replaces `facts.sqlite` only after
validation and SQLite `quick_check` succeed.

## Query

Queries use exact matching, combine filters with `AND`, require at least one
filter, default to 100 rows, and permit at most 500 rows.

```powershell
python -m source_tree.query `
  --file src/model_diagnostic/cfg_base.py `
  --kind assignment `
  --scope CFG2
```

```powershell
python -m source_tree.query --qualified-name CFG2
```

```powershell
python -m source_tree.query `
  --callee execute `
  --caller CFG2.get_tunable_parameters
```

Supported filters are:

```text
file, file-prefix, kind, node-type, scope, parent, name, qualified-name,
symbol-kind, target, callee, caller, module, argument
```

Repeat `--kind` to select several fact kinds. `file-prefix` is a literal prefix,
not a SQL wildcard. Arbitrary SQL, regex, custom ordering, and unrestricted
full scans are not exposed.

`--argument` performs exact matching against scalar call arguments. It is
especially useful for deterministic registry lookups:

```powershell
python -m source_tree.query `
  --callee FEATURE_DAG_REGISTRY.register `
  --argument stack_fields
```

The response contains the normalized query, total matches, returned count,
truncation state, optional next offset, and the original fact objects:

```json
{
  "schema_version": 1,
  "database_schema_version": 1,
  "query": {
    "kinds": ["assignment"],
    "scope": "CFG2",
    "limit": 100,
    "offset": 0
  },
  "matched": 8,
  "returned": 8,
  "truncated": false,
  "facts": []
}
```

If `truncated` is true, narrow the filters before requesting another page. An
agent must not automatically page through the complete fact store.

The query service opens SQLite in read-only/query-only mode and rejects an index
whose stored manifest hash differs from the current `manifest.json`.

## Future MCP wrapper

Both CLIs delegate to `source_tree.fact_store`. A future MCP tool should import
`query_facts()` and expose the same constrained request/response schema. The MCP
adapter must not accept arbitrary SQL and does not need to duplicate storage or
query logic.
