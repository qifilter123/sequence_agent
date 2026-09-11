"""Serialize Tree-sitter named CST nodes for the configured source files."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "ts_source_tree.json"
DEFAULT_OUTPUT = SOURCE_ROOT / "generated" / "ts_source_tree"


class SourceTreeError(RuntimeError):
    """Raised when the configured source tree cannot be built."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceTreeError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise SourceTreeError("config must be an object with schema_version 1")
    return config


def _language_map(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for language in config.get("languages", []):
        if not isinstance(language, dict):
            raise SourceTreeError("each language must be an object")
        try:
            name = language["name"]
            extensions = language["extensions"]
            module = language["module"]
            factory = language.get("factory", "language")
        except KeyError as exc:
            raise SourceTreeError(f"language is missing {exc.args[0]}") from exc
        if not isinstance(name, str) or not isinstance(module, str) or not isinstance(factory, str):
            raise SourceTreeError("language name, module, and factory must be strings")
        if not isinstance(extensions, list) or not extensions:
            raise SourceTreeError(f"language {name} must define extensions")
        for extension in extensions:
            if not isinstance(extension, str) or not extension.startswith("."):
                raise SourceTreeError(f"invalid extension for {name}: {extension!r}")
            if extension in result:
                raise SourceTreeError(f"extension configured more than once: {extension}")
            result[extension] = {
                "name": name,
                "module": module,
                "factory": factory,
            }
    if not result:
        raise SourceTreeError("config.languages must not be empty")
    return result


def _source_files(root: Path, config: dict[str, Any]) -> list[Path]:
    includes = config.get("include")
    excludes = config.get("exclude", [])
    if not isinstance(includes, list) or not includes:
        raise SourceTreeError("config.include must be a non-empty list")
    if not isinstance(excludes, list):
        raise SourceTreeError("config.exclude must be a list")
    try:
        included = {path.resolve() for pattern in includes for path in root.glob(pattern) if path.is_file()}
        excluded = {path.resolve() for pattern in excludes for path in root.glob(pattern) if path.is_file()}
    except (TypeError, ValueError) as exc:
        raise SourceTreeError(f"invalid include/exclude pattern: {exc}") from exc
    files = sorted(included - excluded, key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise SourceTreeError("no source files matched config.include")
    return files


def _load_parser(language_config: dict[str, str]) -> Any:
    try:
        tree_sitter = importlib.import_module("tree_sitter")
        grammar = importlib.import_module(language_config["module"])
        language = tree_sitter.Language(getattr(grammar, language_config["factory"])())
        return tree_sitter.Parser(language)
    except (ImportError, AttributeError, TypeError) as exc:
        raise SourceTreeError(
            f"cannot load Tree-sitter language {language_config['name']}: {exc}"
        ) from exc


def _node_id(file: str, tree_path: str) -> str:
    return f"{file}#{tree_path}"


def _named_nodes(
    node: Any,
    source: bytes,
    file: str,
    *,
    tree_path: str = "0",
    parent_id: str | None = None,
    field_name: str | None = None,
) -> Iterator[dict[str, Any]]:
    node_id = _node_id(file, tree_path)
    record: dict[str, Any] = {
        "id": node_id,
        "file": file,
        "node_type": node.type,
        "parent_id": parent_id,
        "field_name": field_name,
        "start_line": node.start_point.row + 1,
        "start_column": node.start_point.column,
        "end_line": node.end_point.row + 1,
        "end_column": node.end_point.column,
    }
    if not node.named_children:
        record["text"] = source[node.start_byte : node.end_byte].decode("utf-8")
    yield record

    named_index = 0
    for child_index, child in enumerate(node.children):
        if not child.is_named:
            continue
        yield from _named_nodes(
            child,
            source,
            file,
            tree_path=f"{tree_path}.{named_index}",
            parent_id=node_id,
            field_name=node.field_name_for_child(child_index),
        )
        named_index += 1


def _write_outputs(
    output: Path,
    manifest: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "facts.jsonl").write_text(
        "".join(
            json.dumps(node, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for node in nodes
        ),
        encoding="utf-8",
        newline="\n",
    )


def build(
    project_root: Path = PROJECT_ROOT,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    config = _load_config(config_path.resolve(strict=True))
    languages = _language_map(config)
    parser_cache: dict[str, Any] = {}
    nodes: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    parse_error_files = 0

    for path in _source_files(root, config):
        relative = path.relative_to(root).as_posix()
        language_config = languages.get(path.suffix)
        if language_config is None:
            raise SourceTreeError(f"no language configured for {relative}")
        language_name = language_config["name"]
        parser = parser_cache.get(language_name)
        if parser is None:
            parser = _load_parser(language_config)
            parser_cache[language_name] = parser
        source = path.read_bytes()
        try:
            source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceTreeError(f"source is not UTF-8: {relative}") from exc
        tree = parser.parse(source)
        has_error = tree.root_node.has_error
        parse_error_files += int(has_error)
        file_nodes = list(_named_nodes(tree.root_node, source, relative))
        nodes.extend(file_nodes)
        files.append(
            {
                "path": relative,
                "language": language_name,
                "parse_status": "error" if has_error else "ok",
                "node_count": len(file_nodes),
            }
        )

    complete = parse_error_files == 0
    manifest = {
        "schema_version": 1,
        "complete": complete,
        "files": files,
        "totals": {
            "files": len(files),
            "nodes": len(nodes),
            "parse_error_files": parse_error_files,
        },
    }
    if not complete and config.get("fail_on_parse_error", True):
        raise SourceTreeError("Tree-sitter reported parse errors")
    _write_outputs(output.resolve(), manifest, nodes)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build(args.project_root, args.config, args.output)
    except (OSError, SourceTreeError) as exc:
        print(f"source-tree: error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(manifest["totals"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
