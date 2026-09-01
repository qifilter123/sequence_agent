"""Query deterministic source facts through a constrained local interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .fact_store import (
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    FactStoreError,
    QueryRequest,
    query_facts,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = PACKAGE_ROOT / "generated"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=GENERATED_ROOT / "facts.sqlite")
    parser.add_argument("--manifest", type=Path, default=GENERATED_ROOT / "manifest.json")
    parser.add_argument("--file")
    parser.add_argument("--file-prefix")
    parser.add_argument("--kind", action="append", default=[], dest="kinds")
    parser.add_argument("--node-type")
    parser.add_argument("--scope")
    parser.add_argument("--parent")
    parser.add_argument("--name")
    parser.add_argument("--qualified-name")
    parser.add_argument("--symbol-kind")
    parser.add_argument("--target")
    parser.add_argument("--callee")
    parser.add_argument("--caller")
    parser.add_argument("--module")
    parser.add_argument("--argument")
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = QueryRequest(
        file=args.file,
        file_prefix=args.file_prefix,
        kinds=tuple(dict.fromkeys(args.kinds)),
        node_type=args.node_type,
        scope=args.scope,
        parent=args.parent,
        name=args.name,
        qualified_name=args.qualified_name,
        symbol_kind=args.symbol_kind,
        target=args.target,
        callee=args.callee,
        caller=args.caller,
        module=args.module,
        argument=args.argument,
        limit=args.limit,
        offset=args.offset,
    )
    try:
        response = query_facts(args.database, args.manifest, request)
    except (OSError, FactStoreError) as exc:
        print(f"source-tree query: error: {exc}", file=os.sys.stderr)
        return 2
    if args.pretty:
        print(json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(
            json.dumps(
                response,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
