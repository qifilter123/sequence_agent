---
name: dag-schema
description: Define and interpret the generic JSON schema for DAG source graphs, including node types, fields, link types, relation directions, and structural invariants.
---

# DAG Source Graph Schema

Use this domain-independent schema to represent a directed acyclic graph and its referenced source symbols.

The graph is one JSON object with exactly two top-level fields:

```json
{"nodes": [], "links": []}
```

Nodes and links are flat records. Represent ownership and dependencies with links, not nested graph records.

## Node Fields

Each node permits only these fields:

| Field | Rule |
| --- | --- |
| `id` | Required and unique. DAG graph nodes use `<dag_id>.<uuid-v4>`; CODE nodes use `code.<uuid-v4>`. |
| `file_type` | Required node type from the table below. |
| `label` | Required except for unnamed INPUT_FIELD list items. Preserve case. |
| `value` | Allowed only on PARAM. May contain any standard JSON value. |
| `source_file` | Required path relative to the configured source root. |
| `source_location` | Required one-based line in `L<number>` format. |

Do not add fields or placeholder values. Arrays and objects in PARAM.value are data, not nested graph nodes.

## Node Types

| Type | Represents | Label |
| --- | --- | --- |
| `DAG` | One DAG definition | DAG identifier |
| `DAG_NODE` | One executable or build step | DAG node identifier |
| `OPERATION` | Operation selected by a DAG_NODE | Declared operation name |
| `PARAM` | Fixed parameter | Parameter name |
| `CFG_PARAM` | Binding to a configuration source symbol | Parameter name |
| `DERIVED_PARAM` | Parameter supplied by another DAG_NODE | Parameter name |
| `INPUT_LIST` | Named list input container | Input name |
| `INPUT_MAP` | Named map input container | Input name |
| `INPUT_FIELD` | One input binding | Map key; omit for unnamed list items |
| `OUTPUT_MAP` | DAG output container | `outputs` |
| `OUTPUT_FIELD` | One named DAG output | Output name |
| `CODE` | Referenced source symbol | Simple declared symbol name |

A CODE label contains only the symbol name, such as `build_step` or `timeout`. Represent source ownership with `CODE contains CODE`; do not encode it as `Settings.timeout` in the label.

## Link Fields

Each link contains exactly:

```json
{"source": "<node-id>", "target": "<node-id>", "relation": "contains"}
```

Both endpoints must exist. Duplicate `(source, relation, target)` triples are invalid.

## Relations

| Relation | Meaning |
| --- | --- |
| `contains` | The source structurally owns the target. |
| `uses` | The source consumes a value supplied by the target. |
| `references` | The source declares or selects the target implementation. |

Allowed endpoint combinations:

| Relation | Source | Target |
| --- | --- | --- |
| `contains` | DAG | DAG_NODE, OUTPUT_MAP |
| `contains` | DAG_NODE | OPERATION, PARAM, CFG_PARAM, DERIVED_PARAM, INPUT_LIST, INPUT_MAP, INPUT_FIELD |
| `contains` | INPUT_LIST | INPUT_FIELD |
| `contains` | INPUT_MAP | INPUT_FIELD |
| `contains` | OUTPUT_MAP | OUTPUT_FIELD |
| `contains` | CODE | CODE |
| `uses` | INPUT_FIELD | DAG_NODE |
| `uses` | DERIVED_PARAM | DAG_NODE |
| `uses` | OUTPUT_FIELD | DAG_NODE |
| `uses` | CFG_PARAM | CODE |
| `references` | OPERATION | CODE |

An OPERATION or CFG_PARAM may link to multiple CODE nodes. The same CODE node may be reused by multiple source nodes.

## Invariants

- IDs are unique, and every link endpoint exists.
- Every DAG_NODE belongs to one DAG and contains exactly one OPERATION.
- Every DAG contains exactly one OUTPUT_MAP.
- Every non-DAG, non-CODE node has exactly one `contains` parent.
- A CODE node may stand alone or be contained by another CODE node.
- `contains` is acyclic.
- DAG_NODE data dependencies are acyclic within a DAG.
- PARAM is the only node type that carries `value`.

## Example

```json
{
  "nodes": [
    {"id":"example.10000000-0000-4000-8000-000000000001","file_type":"DAG","label":"example","source_file":"config/example.yaml","source_location":"L1"},
    {"id":"example.10000000-0000-4000-8000-000000000002","file_type":"DAG_NODE","label":"stage","source_file":"config/example.yaml","source_location":"L3"},
    {"id":"example.10000000-0000-4000-8000-000000000003","file_type":"OPERATION","label":"build_stage","source_file":"config/example.yaml","source_location":"L4"},
    {"id":"example.10000000-0000-4000-8000-000000000004","file_type":"CFG_PARAM","label":"threshold","source_file":"config/example.yaml","source_location":"L7"},
    {"id":"example.10000000-0000-4000-8000-000000000005","file_type":"PARAM","label":"options","value":["fast",{"retries":1}],"source_file":"config/example.yaml","source_location":"L9"},
    {"id":"example.10000000-0000-4000-8000-000000000006","file_type":"OUTPUT_MAP","label":"outputs","source_file":"config/example.yaml","source_location":"L11"},
    {"id":"example.10000000-0000-4000-8000-000000000007","file_type":"OUTPUT_FIELD","label":"result","source_file":"config/example.yaml","source_location":"L12"},
    {"id":"code.20000000-0000-4000-8000-000000000001","file_type":"CODE","label":"build_stage","source_file":"operations.py","source_location":"L20"},
    {"id":"code.20000000-0000-4000-8000-000000000002","file_type":"CODE","label":"Settings","source_file":"settings.py","source_location":"L5"},
    {"id":"code.20000000-0000-4000-8000-000000000003","file_type":"CODE","label":"threshold","source_file":"settings.py","source_location":"L7"}
  ],
  "links": [
    {"source":"example.10000000-0000-4000-8000-000000000001","target":"example.10000000-0000-4000-8000-000000000002","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000001","target":"example.10000000-0000-4000-8000-000000000006","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000002","target":"example.10000000-0000-4000-8000-000000000003","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000002","target":"example.10000000-0000-4000-8000-000000000004","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000002","target":"example.10000000-0000-4000-8000-000000000005","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000003","target":"code.20000000-0000-4000-8000-000000000001","relation":"references"},
    {"source":"example.10000000-0000-4000-8000-000000000004","target":"code.20000000-0000-4000-8000-000000000003","relation":"uses"},
    {"source":"example.10000000-0000-4000-8000-000000000006","target":"example.10000000-0000-4000-8000-000000000007","relation":"contains"},
    {"source":"example.10000000-0000-4000-8000-000000000007","target":"example.10000000-0000-4000-8000-000000000002","relation":"uses"},
    {"source":"code.20000000-0000-4000-8000-000000000002","target":"code.20000000-0000-4000-8000-000000000003","relation":"contains"}
  ]
}
```

The `source-graph-mcp` server supports queries over source graphs that follow this schema.
