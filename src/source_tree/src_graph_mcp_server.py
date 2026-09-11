"""Serve read-only queries over a generated DAG graph through MCP."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .ts_query import DEFAULT_DATABASE as DEFAULT_TS_DATABASE
    from .ts_query import query_nodes
except ImportError:  # Allows direct execution during development.
    from ts_query import DEFAULT_DATABASE as DEFAULT_TS_DATABASE  # type: ignore[no-redef]
    from ts_query import query_nodes  # type: ignore[no-redef]


DAG_GRAPH_LOCATION_ENV = "DAG_GRAPH_LOCATION"
GRAPH_FILENAME = "dag_graph.json"
_TS_QUERY_LIMIT = 50

_INVERSE_RELATIONS = {
    "uses": "is_used_by",
    "contains": "is_contained_by",
    "references": "is_referenced_by",
}

_TOOL_NAMES = [
    "query_by_label",
    "query_by_id",
    "query_sources_by_target_id",
    "query_targets_by_source_id",
    "ts_query_text",
]


class GraphServerInitializationError(RuntimeError):
    """Raised when the configured graph cannot initialize the MCP server."""


@dataclass(frozen=True)
class _GraphIndex:
    graph_path: Path
    nodes_by_id: dict[str, dict[str, Any]]
    nodes_by_label: dict[str, tuple[dict[str, Any], ...]]
    sources_by_target: dict[str, tuple[tuple[str, str], ...]]
    targets_by_source: dict[str, tuple[tuple[str, str], ...]]
    link_count: int


_graph_index: _GraphIndex | None = None


def _require_non_empty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_graph(graph_path: Path) -> _GraphIndex:
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GraphServerInitializationError(
            f"Cannot read DAG graph {graph_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GraphServerInitializationError(
            f"DAG graph is not valid JSON: {graph_path}: {exc}"
        ) from exc

    if not isinstance(graph, dict) or set(graph) != {"nodes", "links"}:
        raise GraphServerInitializationError(
            "DAG graph must be an object with exactly 'nodes' and 'links'"
        )
    nodes = graph["nodes"]
    links = graph["links"]
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise GraphServerInitializationError(
            "DAG graph nodes and links must both be arrays"
        )

    nodes_by_id: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[dict[str, Any]]] = {}
    for position, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise GraphServerInitializationError(
                f"nodes[{position}] must be an object"
            )
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise GraphServerInitializationError(
                f"nodes[{position}].id must be a non-empty string"
            )
        if node_id in nodes_by_id:
            raise GraphServerInitializationError(f"Duplicate node id: {node_id}")
        stored_node = dict(node)
        nodes_by_id[node_id] = stored_node

        label = node.get("label")
        if label is not None:
            if not isinstance(label, str) or not label:
                raise GraphServerInitializationError(
                    f"Node {node_id} has an invalid label"
                )
            labels.setdefault(label, []).append(stored_node)

    sources: dict[str, list[tuple[str, str]]] = {}
    targets: dict[str, list[tuple[str, str]]] = {}
    seen_links: set[tuple[str, str, str]] = set()
    for position, link in enumerate(links):
        if not isinstance(link, dict) or set(link) != {
            "source",
            "target",
            "relation",
        }:
            raise GraphServerInitializationError(
                f"links[{position}] must contain exactly source, target, relation"
            )
        source = link["source"]
        target = link["target"]
        relation = link["relation"]
        if not all(
            isinstance(value, str) and value
            for value in (source, target, relation)
        ):
            raise GraphServerInitializationError(
                f"links[{position}] values must be non-empty strings"
            )
        if source not in nodes_by_id:
            raise GraphServerInitializationError(
                f"links[{position}].source does not exist: {source}"
            )
        if target not in nodes_by_id:
            raise GraphServerInitializationError(
                f"links[{position}].target does not exist: {target}"
            )
        if relation not in _INVERSE_RELATIONS:
            raise GraphServerInitializationError(
                f"links[{position}] has unsupported relation: {relation}"
            )
        triple = (source, relation, target)
        if triple in seen_links:
            raise GraphServerInitializationError(f"Duplicate link: {triple}")
        seen_links.add(triple)
        sources.setdefault(target, []).append((source, relation))
        targets.setdefault(source, []).append((target, relation))

    return _GraphIndex(
        graph_path=graph_path,
        nodes_by_id=nodes_by_id,
        nodes_by_label={
            label: tuple(matching_nodes)
            for label, matching_nodes in labels.items()
        },
        sources_by_target={
            node_id: tuple(incoming)
            for node_id, incoming in sources.items()
        },
        targets_by_source={
            node_id: tuple(outgoing)
            for node_id, outgoing in targets.items()
        },
        link_count=len(links),
    )


def _load_graph_from_environment() -> _GraphIndex:
    configured = os.environ.get(DAG_GRAPH_LOCATION_ENV)
    if not configured:
        raise GraphServerInitializationError(
            f"Environment variable {DAG_GRAPH_LOCATION_ENV} is required"
        )

    location = Path(configured).expanduser()
    if not location.is_absolute():
        raise GraphServerInitializationError(
            f"{DAG_GRAPH_LOCATION_ENV} must be an absolute directory: {configured}"
        )
    location = location.resolve()
    if not location.is_dir():
        raise GraphServerInitializationError(
            f"{DAG_GRAPH_LOCATION_ENV} is not a directory: {location}"
        )

    graph_path = location / GRAPH_FILENAME
    if not graph_path.is_file():
        raise GraphServerInitializationError(
            f"DAG graph file does not exist: {graph_path}"
        )
    return _load_graph(graph_path)


def _initialize_graph() -> _GraphIndex:
    global _graph_index
    _graph_index = _load_graph_from_environment()
    return _graph_index


def _index() -> _GraphIndex:
    return _graph_index if _graph_index is not None else _initialize_graph()


def query_by_label(label: str) -> str:
    """Return all nodes whose label exactly matches ``label`` as a JSON array.

    Matching is case-sensitive and does not perform partial or normalized
    matching. The array follows graph node order. No match returns ``[]``.
    Returned nodes are unchanged graph records and do not include relation.
    """

    label = _require_non_empty(label, name="label")
    return _json([dict(node) for node in _index().nodes_by_label.get(label, ())])


def query_by_id(id: str) -> str:
    """Return the unique node matching ``id`` as a JSON object.

    Node IDs are exact and case-sensitive. No match returns ``{}``. The
    returned node is an unchanged graph record and does not include relation.
    """

    node_id = _require_non_empty(id, name="id")
    node = _index().nodes_by_id.get(node_id)
    return _json(dict(node) if node is not None else {})


def query_sources_by_target_id(id: str) -> str:
    """Return nodes with links whose target is ``id`` as a JSON array.

    Each result is a copy of the source node plus the original forward link
    relation: ``uses``, ``contains``, or ``references``. For example, if a
    DAG_NODE contains the queried INPUT_MAP, the returned DAG_NODE has
    ``"relation":"contains"``. An unknown ID or no incoming links returns
    ``[]``.
    """

    target_id = _require_non_empty(id, name="id")
    index = _index()
    results: list[dict[str, Any]] = []
    for source_id, relation in index.sources_by_target.get(target_id, ()):
        node = dict(index.nodes_by_id[source_id])
        node["relation"] = relation
        results.append(node)
    return _json(results)


def query_targets_by_source_id(id: str) -> str:
    """Return nodes with links whose source is ``id`` as a JSON array.

    Each result is a copy of the target node plus the inverse relation from the
    returned target's perspective: ``uses`` becomes ``is_used_by``,
    ``contains`` becomes ``is_contained_by``, and ``references`` becomes
    ``is_referenced_by``. For example, an INPUT_MAP contained by the queried
    DAG_NODE has ``"relation":"is_contained_by"``. An unknown ID or no
    outgoing links returns ``[]``.
    """

    source_id = _require_non_empty(id, name="id")
    index = _index()
    results: list[dict[str, Any]] = []
    for target_id, relation in index.targets_by_source.get(source_id, ()):
        node = dict(index.nodes_by_id[target_id])
        node["relation"] = _INVERSE_RELATIONS[relation]
        results.append(node)
    return _json(results)


def ts_query_text(text: str) -> str:
    """Return CODE candidates for exact Tree-sitter identifier matches.

    The query is case-sensitive and fixed to ``node_type="identifier"``. It
    searches all identifier occurrences, including declarations and usages,
    and returns a JSON array of compact CODE candidate objects. Each candidate
    contains ``file_type``, ``label``, ``source_file``, and one-based
    ``source_location``. It has no graph ID because this read-only query does
    not insert the candidate into ``dag_graph.json``. Duplicate file/line
    locations are removed. No match returns ``[]``. More than 50 raw matches
    raises an error instead of returning an incomplete list.
    """

    text = _require_non_empty(text, name="text")
    database = DEFAULT_TS_DATABASE.resolve()
    if not database.is_file():
        raise RuntimeError(f"Tree-sitter fact database does not exist: {database}")

    result = query_nodes(
        database,
        node_type="identifier",
        text=text,
        limit=_TS_QUERY_LIMIT,
    )
    if result["truncated"]:
        raise RuntimeError(
            f"Tree-sitter query for {text!r} exceeds {_TS_QUERY_LIMIT} matches"
        )

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for node in result["nodes"]:
        source_file = node.get("file")
        start_line = node.get("start_line")
        if (
            not isinstance(source_file, str)
            or not source_file
            or not isinstance(start_line, int)
            or start_line < 1
        ):
            raise RuntimeError("Tree-sitter query returned an invalid source location")
        key = (source_file, start_line)
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            {
                "file_type": "CODE",
                "label": text,
                "source_file": source_file,
                "source_location": f"L{start_line}",
            }
        )

    return _json(matches)


def create_server() -> Any:
    """Load the configured graph and create the read-only MCP server."""

    _index()
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP
        except ImportError as exc:
            raise RuntimeError(
                "The MCP Python package is required to run src_graph_mcp_server."
            ) from exc

    server = FastMCP("source-tree-graph")
    server.tool(name="query_by_label")(query_by_label)
    server.tool(name="query_by_id")(query_by_id)
    server.tool(name="query_sources_by_target_id")(query_sources_by_target_id)
    server.tool(name="query_targets_by_source_id")(query_targets_by_source_id)
    server.tool(name="ts_query_text")(ts_query_text)
    return server


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only source_tree DAG graph MCP server."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load and validate the configured graph without starting MCP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    try:
        index = _initialize_graph()
        if arguments.check:
            print(
                _json(
                    {
                        "graph_path": str(index.graph_path),
                        "link_count": index.link_count,
                        "node_count": len(index.nodes_by_id),
                        "status": "ok",
                        "tools": _TOOL_NAMES,
                    }
                )
            )
            return 0
        server = create_server()
    except (GraphServerInitializationError, RuntimeError) as exc:
        print(_json({"status": "error", "error": str(exc)}), file=os.sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
