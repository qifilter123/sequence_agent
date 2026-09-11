"""Build a disposable SQLite query index from source-tree JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from source_tree.ts_fact_store import FactStoreError, build_fact_database

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
GENERATED_ROOT = SOURCE_ROOT / "generated" / "ts_source_tree"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=GENERATED_ROOT / "manifest.json")
    parser.add_argument("--facts", type=Path, default=GENERATED_ROOT / "facts.jsonl")
    parser.add_argument("--output", type=Path, default=GENERATED_ROOT / "facts.sqlite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = build_fact_database(args.manifest, args.facts, args.output)
    except (OSError, FactStoreError) as exc:
        print(f"source-tree index: error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
