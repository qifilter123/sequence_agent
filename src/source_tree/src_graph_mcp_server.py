"""Lightweight MCP facade for source_tree DAG graph generation."""

from __future__ import annotations

import argparse
import json
from typing import Any

try:
    from .src_graph_generator import (
        GraphGenerationError,
        GraphValidationError,
        generate_graph,
    )
except ImportError:  # Allows direct execution of this file during development.
    from src_graph_generator import (  # type: ignore[no-redef]
        GraphGenerationError,
        GraphValidationError,
        generate_graph,
    )


def init_graph(
    source_root: str | None = None,
    input_glob: str | None = None,
    output_path: str | None = None,
    schema_path: str | None = None,
) -> dict[str, Any]:
    """Generate the base graph and return nodes needing CODE resolution.

    The graph file contains no placeholders.  Each item in
    ``unresolved_nodes`` identifies an existing OPERATION or CFG_PARAM node,
    its original YAML reference, and the link type an agent should eventually
    create after verifying a CODE target.
    """

    try:
        return generate_graph(
            source_root,
            input_glob=input_glob,
            output_path=output_path,
            schema_path=schema_path,
        )
    except (GraphGenerationError, GraphValidationError) as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def query() -> dict[str, str]:
    """Placeholder for the graph query API used by subsequent agents."""

    return {"status": "not_implemented"}


def create_server() -> Any:
    """Create the MCP server without making MCP a generator dependency.

    MCP SDK 1.x exposes ``FastMCP`` while SDK 2.x renamed the same high-level
    server API to ``MCPServer``.  The registration calls used here are shared
    by both versions.
    """

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP
        except ImportError as exc:
            raise RuntimeError(
                "The MCP Python package is required to run src_graph_mcp_server. "
                "Install it in the source_tree runtime environment."
            ) from exc

    server = FastMCP("source-tree-graph")
    server.tool(name="init_graph")(init_graph)
    server.tool(name="query")(query)
    return server


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the source_tree graph MCP server.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate imports and tool functions without starting the MCP transport.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    if arguments.check:
        print(json.dumps({"tools": ["init_graph", "query"], "status": "ok"}))
        return 0
    try:
        server = create_server()
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
