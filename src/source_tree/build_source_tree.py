#!/usr/bin/env python3
"""Build a compact, deterministic Tree-sitter fact index for LLM agents.

The output is intentionally not a serialized concrete syntax tree.  It keeps
the source locations and syntax-derived facts that an agent normally needs,
while omitting punctuation and most expression detail that would waste tokens.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
FACT_KINDS = ("module", "import", "symbol", "call", "assignment")


class SourceTreeError(RuntimeError):
    """Raised when deterministic indexing cannot be completed safely."""


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    module: str
    factory: str
    extractor: str


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _span(node: Any) -> dict[str, list[int]]:
    return {
        "start": [node.start_point.row + 1, node.start_point.column],
        "end": [node.end_point.row + 1, node.end_point.column],
    }


def _fact_id(path: str, kind: str, node: Any) -> str:
    return (
        f"{path}:{node.start_point.row + 1}:{node.start_point.column}"
        f"-{node.end_point.row + 1}:{node.end_point.column}:{kind}"
    )


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="strict")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _bounded_text(text: str, limit: int = 160) -> str | dict[str, Any]:
    normalized = _one_line(text)
    if len(normalized) <= limit:
        return normalized
    encoded = normalized.encode("utf-8")
    return {
        "preview": normalized[:limit],
        "chars": len(normalized),
        "sha256": _sha256(encoded),
        "truncated": True,
    }


def _iter_nodes(node: Any) -> Iterator[Any]:
    yield node
    # Missing recovery tokens can be anonymous. Error detection must traverse
    # the full CST even though fact extraction intentionally uses named nodes.
    for child in node.children:
        yield from _iter_nodes(child)


def _error_summary(root: Any) -> tuple[int, int, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    error_nodes = 0
    missing_nodes = 0
    for node in _iter_nodes(root):
        if node.is_error:
            error_nodes += 1
            errors.append({"kind": "ERROR", "node_type": node.type, "span": _span(node)})
        if node.is_missing:
            missing_nodes += 1
            errors.append({"kind": "MISSING", "node_type": node.type, "span": _span(node)})
    errors.sort(key=lambda item: (item["span"]["start"], item["kind"], item["node_type"]))
    return error_nodes, missing_nodes, errors


def _validate_pattern(pattern: str) -> None:
    candidate = PurePosixPath(pattern.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SourceTreeError(f"unsafe glob pattern: {pattern!r}")
    if not pattern.strip():
        raise SourceTreeError("glob patterns must not be empty")


def _expand_patterns(
    root: Path,
    patterns: Sequence[str],
    label: str,
    *,
    require_match: bool,
) -> set[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        _validate_pattern(pattern)
        current = sorted(path for path in root.glob(pattern) if path.is_file())
        if not current and require_match:
            raise SourceTreeError(f"{label} pattern matched no files: {pattern}")
        for lexical_path in current:
            resolved = lexical_path.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise SourceTreeError(
                    f"{label} pattern escapes project root through symlink: {lexical_path}"
                ) from exc
            matches.add(resolved)
    return matches


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceTreeError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SourceTreeError("config root must be a JSON object")
    if config.get("schema_version") != 1:
        raise SourceTreeError("config schema_version must be 1")
    return config


def _language_specs(config: Mapping[str, Any]) -> tuple[LanguageSpec, ...]:
    raw_languages = config.get("languages")
    if not isinstance(raw_languages, list) or not raw_languages:
        raise SourceTreeError("config.languages must be a non-empty list")
    specs: list[LanguageSpec] = []
    claimed_extensions: set[str] = set()
    for raw in raw_languages:
        if not isinstance(raw, dict):
            raise SourceTreeError("each language entry must be an object")
        name = raw.get("name")
        extensions = raw.get("extensions")
        module = raw.get("module")
        factory = raw.get("factory", "language")
        extractor = raw.get("extractor")
        if not isinstance(name, str) or not name:
            raise SourceTreeError("language.name must be a non-empty string")
        if not isinstance(extensions, list) or not extensions:
            raise SourceTreeError(f"language {name}: extensions must be non-empty")
        normalized_extensions: list[str] = []
        for extension in extensions:
            if not isinstance(extension, str) or not extension.startswith("."):
                raise SourceTreeError(f"language {name}: invalid extension {extension!r}")
            if extension in claimed_extensions:
                raise SourceTreeError(f"extension claimed by multiple languages: {extension}")
            claimed_extensions.add(extension)
            normalized_extensions.append(extension)
        if not isinstance(module, str) or not module:
            raise SourceTreeError(f"language {name}: module must be non-empty")
        if not isinstance(factory, str) or not factory:
            raise SourceTreeError(f"language {name}: factory must be non-empty")
        if extractor != "python":
            raise SourceTreeError(
                f"language {name}: unsupported extractor {extractor!r}; "
                "add a language-specific fact extractor before enabling the grammar"
            )
        specs.append(
            LanguageSpec(
                name=name,
                extensions=tuple(normalized_extensions),
                module=module,
                factory=factory,
                extractor=extractor,
            )
        )
    return tuple(specs)


def _spec_for(path: Path, specs: Sequence[LanguageSpec]) -> LanguageSpec:
    candidates = [spec for spec in specs if path.suffix in spec.extensions]
    if len(candidates) != 1:
        raise SourceTreeError(f"no unique language configured for {path.name}")
    return candidates[0]


def _load_parser(spec: LanguageSpec) -> tuple[Any, dict[str, str]]:
    try:
        tree_sitter = importlib.import_module("tree_sitter")
        grammar_module = importlib.import_module(spec.module)
        factory = getattr(grammar_module, spec.factory)
        language = tree_sitter.Language(factory())
        parser = tree_sitter.Parser(language)
    except (ImportError, AttributeError, TypeError) as exc:
        raise SourceTreeError(f"cannot load Tree-sitter language {spec.name}: {exc}") from exc
    versions = {
        "tree_sitter": importlib.metadata.version("tree-sitter"),
        spec.module: importlib.metadata.version(spec.module.replace("_", "-")),
    }
    return parser, versions


def _decorators(node: Any, source: bytes) -> list[str | dict[str, Any]]:
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return []
    return [
        _bounded_text(_node_text(child, source))
        for child in parent.named_children
        if child.type == "decorator"
    ]


def _parameter_list(node: Any, source: bytes) -> list[str | dict[str, Any]]:
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return []
    return [_bounded_text(_node_text(child, source), 120) for child in parameters.named_children]


def _base_list(node: Any, source: bytes) -> list[str | dict[str, Any]]:
    superclasses = node.child_by_field_name("superclasses")
    if superclasses is None:
        return []
    return [_bounded_text(_node_text(child, source), 120) for child in superclasses.named_children]


def _string_value(text: str) -> Any:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return _bounded_text(text)
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, bytes):
        return {
            "literal_type": "bytes",
            "bytes": len(value),
            "sha256": _sha256(value),
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {
        "literal_type": type(value).__name__,
        "source": _bounded_text(text),
    }


def _expression_summary(node: Any, source: bytes) -> dict[str, Any]:
    node_type = node.type
    text = _node_text(node, source)
    if node_type in {"string", "concatenated_string"}:
        return {"syntax": node_type, "value": _string_value(text)}
    if node_type in {"integer", "float", "true", "false", "none"}:
        return {"syntax": node_type, "value": _bounded_text(text, 80)}
    if node_type in {"identifier", "attribute"}:
        return {"syntax": node_type, "name": _bounded_text(text, 160)}
    if node_type == "call":
        function = node.child_by_field_name("function")
        return {
            "syntax": "call",
            "callee": _bounded_text(_node_text(function, source), 200) if function else None,
        }
    summary: dict[str, Any] = {"syntax": node_type}
    bounded = _bounded_text(text, 120)
    if isinstance(bounded, str):
        summary["text"] = bounded
    else:
        summary["text"] = bounded
    return summary


def _call_arguments(call: Any, source: bytes) -> list[dict[str, Any]]:
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return []
    result: list[dict[str, Any]] = []
    position = 0
    for argument in arguments.named_children:
        if argument.type == "keyword_argument":
            name = argument.child_by_field_name("name")
            value = argument.child_by_field_name("value")
            result.append(
                {
                    "keyword": _node_text(name, source) if name else None,
                    "value": _expression_summary(value, source) if value else None,
                }
            )
        else:
            result.append({"position": position, "value": _expression_summary(argument, source)})
            position += 1
    return result


def _python_facts(path: str, root: Any, source: bytes) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = [
        {
            "file": path,
            "id": f"{path}:module",
            "kind": "module",
            "node_type": root.type,
            "span": _span(root),
        }
    ]

    def scope_name(scope: tuple[tuple[str, str], ...]) -> str | None:
        return ".".join(item[0] for item in scope) if scope else None

    def visit(node: Any, scope: tuple[tuple[str, str], ...]) -> None:
        active_scope = scope
        if node.type in {"class_definition", "function_definition"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = _node_text(name_node, source)
                qualified_name = ".".join((*[item[0] for item in scope], name))
                if node.type == "class_definition":
                    symbol_kind = "class"
                else:
                    symbol_kind = "method" if scope and scope[-1][1] == "class" else "function"
                fact: dict[str, Any] = {
                    "file": path,
                    "id": _fact_id(path, "symbol", node),
                    "kind": "symbol",
                    "node_type": node.type,
                    "symbol_kind": symbol_kind,
                    "name": name,
                    "qualified_name": qualified_name,
                    "parent": scope_name(scope),
                    "span": _span(node),
                }
                decorators = _decorators(node, source)
                if decorators:
                    fact["decorators"] = decorators
                if node.type == "class_definition":
                    bases = _base_list(node, source)
                    if bases:
                        fact["bases"] = bases
                else:
                    fact["parameters"] = _parameter_list(node, source)
                    return_type = node.child_by_field_name("return_type")
                    if return_type is not None:
                        fact["return_type"] = _bounded_text(_node_text(return_type, source), 120)
                facts.append(fact)
                active_scope = (*scope, (name, symbol_kind))

        if node.type in {"import_statement", "import_from_statement"}:
            import_fact: dict[str, Any] = {
                "file": path,
                "id": _fact_id(path, "import", node),
                "kind": "import",
                "node_type": node.type,
                "scope": scope_name(scope),
                "statement": _bounded_text(_node_text(node, source), 240),
                "span": _span(node),
            }
            module = node.child_by_field_name("module_name")
            if module is not None:
                import_fact["module"] = _bounded_text(_node_text(module, source), 200)
            names: list[str | dict[str, Any]] = []
            for index, child in enumerate(node.children):
                if node.field_name_for_child(index) == "name":
                    names.append(_bounded_text(_node_text(child, source), 200))
            import_fact["names"] = names
            facts.append(import_fact)

        if node.type == "call":
            function = node.child_by_field_name("function")
            facts.append(
                {
                    "file": path,
                    "id": _fact_id(path, "call", node),
                    "kind": "call",
                    "node_type": node.type,
                    "caller": scope_name(scope) or "<module>",
                    "callee": _bounded_text(_node_text(function, source), 240) if function else None,
                    "arguments": _call_arguments(node, source),
                    "span": _span(node),
                }
            )

        if node.type in {"assignment", "augmented_assignment", "named_expression"}:
            left = node.child_by_field_name("left") or node.child_by_field_name("name")
            right = node.child_by_field_name("right") or node.child_by_field_name("value")
            facts.append(
                {
                    "file": path,
                    "id": _fact_id(path, "assignment", node),
                    "kind": "assignment",
                    "node_type": node.type,
                    "scope": scope_name(scope) or "<module>",
                    "target": _bounded_text(_node_text(left, source), 200) if left else None,
                    "value": _expression_summary(right, source) if right else None,
                    "span": _span(node),
                }
            )

        for child in node.named_children:
            visit(child, active_scope)

    visit(root, ())
    facts.sort(
        key=lambda fact: (
            fact["span"]["start"][0],
            fact["span"]["start"][1],
            FACT_KINDS.index(fact["kind"]),
            fact["id"],
        )
    )
    return facts


def _write_outputs(
    output_dir: Path,
    manifest: Mapping[str, Any],
    facts: Iterable[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    facts_path = output_dir / "facts.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    facts_path.write_text(
        "".join(_compact_json(fact) + "\n" for fact in facts),
        encoding="utf-8",
        newline="\n",
    )


def build(project_root: Path, config_path: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = project_root.resolve(strict=True)
    config_resolved = config_path.resolve(strict=True)
    try:
        config_resolved.relative_to(root)
    except ValueError as exc:
        raise SourceTreeError("config file must be inside the project root") from exc
    config = _load_config(config_resolved)
    specs = _language_specs(config)

    includes = config.get("include")
    excludes = config.get("exclude", [])
    if not isinstance(includes, list) or not includes or not all(isinstance(x, str) for x in includes):
        raise SourceTreeError("config.include must be a non-empty string list")
    if not isinstance(excludes, list) or not all(isinstance(x, str) for x in excludes):
        raise SourceTreeError("config.exclude must be a string list")
    included = _expand_patterns(root, includes, "include", require_match=True)
    excluded = (
        _expand_patterns(root, excludes, "exclude", require_match=False) if excludes else set()
    )
    selected = sorted(included - excluded, key=lambda path: path.relative_to(root).as_posix())
    if not selected:
        raise SourceTreeError("no source files remain after exclusions")

    output_resolved = output_dir.resolve()
    if output_resolved == root or root not in output_resolved.parents:
        raise SourceTreeError("output directory must be strictly inside the project root")

    parser_cache: dict[str, Any] = {}
    dependency_versions: dict[str, str] = {}
    all_facts: list[dict[str, Any]] = []
    file_entries: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    has_parse_errors = False

    for source_path in selected:
        relative_path = source_path.relative_to(root).as_posix()
        spec = _spec_for(source_path, specs)
        if spec.name not in parser_cache:
            parser, versions = _load_parser(spec)
            parser_cache[spec.name] = parser
            dependency_versions.update(versions)
        parser = parser_cache[spec.name]
        source = source_path.read_bytes()
        try:
            source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceTreeError(f"source is not valid UTF-8: {relative_path}") from exc
        tree = parser.parse(source)
        error_nodes, missing_nodes, errors = _error_summary(tree.root_node)
        status = "ok" if error_nodes == 0 and missing_nodes == 0 else "partial"
        has_parse_errors |= status != "ok"
        facts = _python_facts(relative_path, tree.root_node, source)
        counts = Counter(fact["kind"] for fact in facts)
        total_counts.update(counts)
        all_facts.extend(facts)
        file_entry: dict[str, Any] = {
            "path": relative_path,
            "language": spec.name,
            "sha256": _sha256(source),
            "bytes": len(source),
            "lines": source.count(b"\n") + (0 if not source or source.endswith(b"\n") else 1),
            "parse": {
                "status": status,
                "error_nodes": error_nodes,
                "missing_nodes": missing_nodes,
            },
            "facts": {kind: counts.get(kind, 0) for kind in FACT_KINDS},
        }
        if errors:
            file_entry["parse"]["errors"] = errors
        file_entries.append(file_entry)

    fact_ids = [fact["id"] for fact in all_facts]
    if len(fact_ids) != len(set(fact_ids)):
        duplicates = sorted(fact_id for fact_id, count in Counter(fact_ids).items() if count > 1)
        raise SourceTreeError(f"non-unique fact ids generated: {duplicates[:3]}")

    fail_on_parse_error = config.get("fail_on_parse_error", True)
    if not isinstance(fail_on_parse_error, bool):
        raise SourceTreeError("config.fail_on_parse_error must be boolean")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "name": "build_source_tree",
            "version": GENERATOR_VERSION,
            "dependencies": dict(sorted(dependency_versions.items())),
        },
        "complete": not has_parse_errors,
        "strict_parse": fail_on_parse_error,
        "inputs": {
            "include": includes,
            "exclude": excludes,
        },
        "files": file_entries,
        "totals": {
            "files": len(file_entries),
            "facts": {kind: total_counts.get(kind, 0) for kind in FACT_KINDS},
            "parse_error_files": sum(1 for item in file_entries if item["parse"]["status"] != "ok"),
        },
    }
    _write_outputs(output_resolved, manifest, all_facts)
    if has_parse_errors and fail_on_parse_error:
        raise SourceTreeError(
            "Tree-sitter reported parse errors; outputs were marked complete=false and must not be consumed"
        )
    return manifest, all_facts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, _ = build(args.project_root, args.config, args.output)
    except (OSError, SourceTreeError) as exc:
        print(f"source-tree: error: {exc}", file=os.sys.stderr)
        return 2
    print(_compact_json({"complete": manifest["complete"], "totals": manifest["totals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
