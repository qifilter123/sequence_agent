"""Build a flat DAG graph from YAML declarations.

The generator deliberately stops at source-code resolution.  OPERATION and
CFG_PARAM nodes are emitted into the graph and returned as unresolved records;
an agent can later create verified CODE nodes and the corresponding links.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_GLOB = str(
    DEFAULT_SOURCE_ROOT / "model_diagnostic" / "config" / "*.yaml"
)
DEFAULT_OUTPUT_PATH = str(
    DEFAULT_SOURCE_ROOT / "generated" / "source_tree" / "dag_graph.json"
)
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("dag-schema.json")

_ALLOWED_LINK_TYPES = {
    ("DAG", "contains", "DAG_NODE"),
    ("DAG", "contains", "OUTPUT_MAP"),
    ("DAG_NODE", "contains", "OPERATION"),
    ("DAG_NODE", "contains", "PARAM"),
    ("DAG_NODE", "contains", "CFG_PARAM"),
    ("DAG_NODE", "contains", "DERIVED_PARAM"),
    ("DAG_NODE", "contains", "INPUT_LIST"),
    ("DAG_NODE", "contains", "INPUT_MAP"),
    ("DAG_NODE", "contains", "INPUT_FIELD"),
    ("INPUT_LIST", "contains", "INPUT_FIELD"),
    ("INPUT_MAP", "contains", "INPUT_FIELD"),
    ("OUTPUT_MAP", "contains", "OUTPUT_FIELD"),
    ("INPUT_FIELD", "uses", "DAG_NODE"),
    ("DERIVED_PARAM", "uses", "DAG_NODE"),
    ("OUTPUT_FIELD", "uses", "DAG_NODE"),
    ("OPERATION", "references", "CODE"),
    ("CFG_PARAM", "uses", "CODE"),
}

_LABELED_NODE_TYPES = {
    "DAG",
    "DAG_NODE",
    "OPERATION",
    "PARAM",
    "CFG_PARAM",
    "DERIVED_PARAM",
    "INPUT_LIST",
    "INPUT_MAP",
    "OUTPUT_MAP",
    "OUTPUT_FIELD",
    "CODE",
}

_MISSING = object()


class GraphGenerationError(ValueError):
    """Raised when the YAML declarations cannot produce a valid base graph."""


class GraphValidationError(ValueError):
    """Raised when a generated or supplied graph violates the graph contract."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        message = "Graph validation failed:\n- " + "\n- ".join(self.errors)
        super().__init__(message)


@dataclass
class _NodeDeclaration:
    data: dict[str, Any]
    yaml_node: MappingNode
    yaml_id: str
    graph_id: str


@dataclass
class _DagDocument:
    path: Path
    source_file: str
    data: dict[str, Any]
    root: MappingNode
    dag_id: str
    dag_graph_id: str = ""
    declarations: list[_NodeDeclaration] = field(default_factory=list)
    node_lookup: dict[str, str] = field(default_factory=dict)


class _GraphBuilder:
    def __init__(self) -> None:
        self.graph: dict[str, list[dict[str, Any]]] = {
            "nodes": [],
            "links": [],
        }
        self.unresolved_nodes: list[dict[str, Any]] = []

    def add_node(
        self,
        *,
        dag_id: str,
        file_type: str,
        source_file: str,
        source_location: str,
        label: str | None = None,
        value: Any = _MISSING,
    ) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": f"{dag_id}.{uuid.uuid4()}",
            "file_type": file_type,
            "source_file": source_file,
            "source_location": source_location,
        }
        if label is not None:
            node["label"] = label
        if value is not _MISSING:
            node["value"] = value
        self.graph["nodes"].append(node)
        return node

    def add_link(self, source: str, target: str, relation: str) -> None:
        self.graph["links"].append(
            {"source": source, "target": target, "relation": relation}
        )

    def add_unresolved(
        self,
        source_node: Mapping[str, Any],
        *,
        reference: str,
        relation: str,
    ) -> None:
        self.unresolved_nodes.append(
            {
                "source_node": dict(source_node),
                "reference": reference,
                "relation": relation,
                "target_file_type": "CODE",
            }
        )


def _line(node: Node) -> str:
    return f"L{node.start_mark.line + 1}"


def _mapping_entries(
    node: Node,
    *,
    context: str,
) -> dict[str, tuple[ScalarNode, Node]]:
    if not isinstance(node, MappingNode):
        raise GraphGenerationError(f"{context} must be a YAML mapping")

    entries: dict[str, tuple[ScalarNode, Node]] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
            raise GraphGenerationError(f"{context} contains a non-string key at {_line(key_node)}")
        key = key_node.value
        if key in entries:
            raise GraphGenerationError(
                f"{context} contains duplicate key {key!r} at {_line(key_node)}"
            )
        entries[key] = (key_node, value_node)
    return entries


def _require_mapping_data(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphGenerationError(f"{context} must be a YAML mapping")
    for key in value:
        if not isinstance(key, str):
            raise GraphGenerationError(f"{context} contains non-string key {key!r}")
    return value


def _relative_source_file(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError as exc:
        raise GraphGenerationError(
            f"Input YAML must be inside source root {source_root}: {path}"
        ) from exc


def _load_document(path: Path, source_root: Path) -> _DagDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphGenerationError(f"Cannot read YAML {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise GraphGenerationError(f"Invalid YAML {path}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(root, MappingNode):
        raise GraphGenerationError(f"YAML root must be a mapping: {path}")
    _require_mapping_data(data, context=str(path))
    root_entries = _mapping_entries(root, context=str(path))

    if "dag_id" not in root_entries:
        raise GraphGenerationError(f"Missing dag_id in {path}")
    dag_id = data.get("dag_id")
    if not isinstance(dag_id, str) or not dag_id:
        raise GraphGenerationError(f"dag_id must be a non-empty string in {path}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", dag_id):
        raise GraphGenerationError(
            f"dag_id {dag_id!r} contains characters unsupported by graph IDs in {path}"
        )

    if "nodes" not in root_entries:
        raise GraphGenerationError(f"Missing nodes in {path}")
    nodes_data = data.get("nodes")
    nodes_yaml = root_entries["nodes"][1]
    if not isinstance(nodes_data, list) or not isinstance(nodes_yaml, SequenceNode):
        raise GraphGenerationError(f"nodes must be a YAML sequence in {path}")
    if len(nodes_data) != len(nodes_yaml.value):
        raise GraphGenerationError(f"Could not align parsed nodes with YAML locations in {path}")

    if "outputs" not in root_entries:
        raise GraphGenerationError(f"Missing explicit outputs mapping in {path}")
    _require_mapping_data(data.get("outputs"), context=f"{path}: outputs")
    _mapping_entries(root_entries["outputs"][1], context=f"{path}: outputs")

    return _DagDocument(
        path=path,
        source_file=_relative_source_file(path, source_root),
        data=data,
        root=root,
        dag_id=dag_id,
    )


def _collect_input_files(source_root: Path, input_glob: str) -> list[Path]:
    pattern_path = Path(input_glob)
    pattern = str(pattern_path if pattern_path.is_absolute() else source_root / pattern_path)
    paths = sorted(
        {Path(match).resolve() for match in glob.glob(pattern, recursive=True) if Path(match).is_file()},
        key=lambda item: item.as_posix(),
    )
    if not paths:
        raise GraphGenerationError(f"No YAML files matched input pattern: {input_glob}")
    for path in paths:
        _relative_source_file(path, source_root)
    return paths


def _add_provider_link_if_local(
    builder: _GraphBuilder,
    *,
    source_node_id: str,
    provider_reference: Any,
    document: _DagDocument,
    context: str,
) -> None:
    if not isinstance(provider_reference, str) or not provider_reference:
        raise GraphGenerationError(f"{context} must reference a provider by non-empty string")
    target = document.node_lookup.get(provider_reference)
    if target is not None:
        builder.add_link(source_node_id, target, "uses")


def _build_declaration_details(
    builder: _GraphBuilder,
    document: _DagDocument,
    declaration: _NodeDeclaration,
) -> None:
    source_file = document.source_file
    fields = _mapping_entries(
        declaration.yaml_node,
        context=f"{source_file}: node {declaration.yaml_id}",
    )
    owner_id = declaration.graph_id

    operation = declaration.data.get("op")
    if "op" not in fields or not isinstance(operation, str) or not operation:
        raise GraphGenerationError(
            f"Node {declaration.yaml_id!r} requires a non-empty op in {source_file}"
        )
    operation_node = builder.add_node(
        dag_id=document.dag_id,
        file_type="OPERATION",
        label=operation,
        source_file=source_file,
        source_location=_line(fields["op"][0]),
    )
    builder.add_link(owner_id, operation_node["id"], "contains")
    builder.add_unresolved(
        operation_node,
        reference=operation,
        relation="references",
    )

    if "params" in fields:
        params = _require_mapping_data(
            declaration.data.get("params"),
            context=f"{source_file}: node {declaration.yaml_id} params",
        )
        param_entries = _mapping_entries(
            fields["params"][1],
            context=f"{source_file}: node {declaration.yaml_id} params",
        )
        if set(params) != set(param_entries):
            raise GraphGenerationError(
                f"Could not align params with YAML locations for {declaration.yaml_id}"
            )
        for name, (key_node, _) in param_entries.items():
            value = params[name]
            if not _is_json_value(value):
                raise GraphGenerationError(
                    f"PARAM {name!r} on {declaration.yaml_id!r} is not a standard JSON value"
                )
            param_node = builder.add_node(
                dag_id=document.dag_id,
                file_type="PARAM",
                label=name,
                value=value,
                source_file=source_file,
                source_location=_line(key_node),
            )
            builder.add_link(owner_id, param_node["id"], "contains")

    if "cfg_params" in fields:
        cfg_params = _require_mapping_data(
            declaration.data.get("cfg_params"),
            context=f"{source_file}: node {declaration.yaml_id} cfg_params",
        )
        cfg_entries = _mapping_entries(
            fields["cfg_params"][1],
            context=f"{source_file}: node {declaration.yaml_id} cfg_params",
        )
        if set(cfg_params) != set(cfg_entries):
            raise GraphGenerationError(
                f"Could not align cfg_params with YAML locations for {declaration.yaml_id}"
            )
        for name, (key_node, _) in cfg_entries.items():
            reference = cfg_params[name]
            if not isinstance(reference, str) or not reference:
                raise GraphGenerationError(
                    f"CFG_PARAM {name!r} on {declaration.yaml_id!r} must bind a non-empty attribute name"
                )
            cfg_node = builder.add_node(
                dag_id=document.dag_id,
                file_type="CFG_PARAM",
                label=name,
                source_file=source_file,
                source_location=_line(key_node),
            )
            builder.add_link(owner_id, cfg_node["id"], "contains")
            builder.add_unresolved(cfg_node, reference=reference, relation="uses")

    if "derived_params" in fields:
        derived_params = _require_mapping_data(
            declaration.data.get("derived_params"),
            context=f"{source_file}: node {declaration.yaml_id} derived_params",
        )
        derived_entries = _mapping_entries(
            fields["derived_params"][1],
            context=f"{source_file}: node {declaration.yaml_id} derived_params",
        )
        if set(derived_params) != set(derived_entries):
            raise GraphGenerationError(
                f"Could not align derived_params with YAML locations for {declaration.yaml_id}"
            )
        for name, (key_node, _) in derived_entries.items():
            reference = derived_params[name]
            if not isinstance(reference, str) or not reference:
                raise GraphGenerationError(
                    f"DERIVED_PARAM {name!r} on {declaration.yaml_id!r} requires a provider node id"
                )
            target = document.node_lookup.get(reference)
            if target is None:
                raise GraphGenerationError(
                    f"DERIVED_PARAM {name!r} on {declaration.yaml_id!r} references unknown node {reference!r}"
                )
            derived_node = builder.add_node(
                dag_id=document.dag_id,
                file_type="DERIVED_PARAM",
                label=name,
                source_file=source_file,
                source_location=_line(key_node),
            )
            builder.add_link(owner_id, derived_node["id"], "contains")
            builder.add_link(derived_node["id"], target, "uses")

    if "inputs" in fields:
        inputs = _require_mapping_data(
            declaration.data.get("inputs"),
            context=f"{source_file}: node {declaration.yaml_id} inputs",
        )
        input_entries = _mapping_entries(
            fields["inputs"][1],
            context=f"{source_file}: node {declaration.yaml_id} inputs",
        )
        if set(inputs) != set(input_entries):
            raise GraphGenerationError(
                f"Could not align inputs with YAML locations for {declaration.yaml_id}"
            )
        for name, (key_node, value_node) in input_entries.items():
            value = inputs[name]
            context = f"{source_file}: node {declaration.yaml_id} input {name}"

            if isinstance(value_node, ScalarNode):
                input_node = builder.add_node(
                    dag_id=document.dag_id,
                    file_type="INPUT_FIELD",
                    label=name,
                    source_file=source_file,
                    source_location=_line(key_node),
                )
                builder.add_link(owner_id, input_node["id"], "contains")
                _add_provider_link_if_local(
                    builder,
                    source_node_id=input_node["id"],
                    provider_reference=value,
                    document=document,
                    context=context,
                )
                continue

            if isinstance(value_node, SequenceNode):
                if not isinstance(value, list) or len(value) != len(value_node.value):
                    raise GraphGenerationError(f"Could not align list values for {context}")
                container = builder.add_node(
                    dag_id=document.dag_id,
                    file_type="INPUT_LIST",
                    label=name,
                    source_file=source_file,
                    source_location=_line(key_node),
                )
                builder.add_link(owner_id, container["id"], "contains")
                for index, (item, item_node) in enumerate(zip(value, value_node.value)):
                    if not isinstance(item_node, ScalarNode):
                        raise GraphGenerationError(
                            f"Nested input container is unsupported for {context}[{index}]"
                        )
                    input_node = builder.add_node(
                        dag_id=document.dag_id,
                        file_type="INPUT_FIELD",
                        source_file=source_file,
                        source_location=_line(item_node),
                    )
                    builder.add_link(container["id"], input_node["id"], "contains")
                    _add_provider_link_if_local(
                        builder,
                        source_node_id=input_node["id"],
                        provider_reference=item,
                        document=document,
                        context=f"{context}[{index}]",
                    )
                continue

            if isinstance(value_node, MappingNode):
                input_map = _require_mapping_data(value, context=context)
                map_entries = _mapping_entries(value_node, context=context)
                if set(input_map) != set(map_entries):
                    raise GraphGenerationError(f"Could not align map values for {context}")
                container = builder.add_node(
                    dag_id=document.dag_id,
                    file_type="INPUT_MAP",
                    label=name,
                    source_file=source_file,
                    source_location=_line(key_node),
                )
                builder.add_link(owner_id, container["id"], "contains")
                for field_name, (field_key_node, field_value_node) in map_entries.items():
                    if not isinstance(field_value_node, ScalarNode):
                        raise GraphGenerationError(
                            f"Nested input container is unsupported for {context}.{field_name}"
                        )
                    input_node = builder.add_node(
                        dag_id=document.dag_id,
                        file_type="INPUT_FIELD",
                        label=field_name,
                        source_file=source_file,
                        source_location=_line(field_key_node),
                    )
                    builder.add_link(container["id"], input_node["id"], "contains")
                    _add_provider_link_if_local(
                        builder,
                        source_node_id=input_node["id"],
                        provider_reference=input_map[field_name],
                        document=document,
                        context=f"{context}.{field_name}",
                    )
                continue

            raise GraphGenerationError(f"Unsupported YAML input shape for {context}")


def _build_graph(documents: list[_DagDocument]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    builder = _GraphBuilder()
    dag_ids: set[str] = set()

    # Pass one creates every DAG and DAG_NODE, making forward references stable.
    for document in documents:
        if document.dag_id in dag_ids:
            raise GraphGenerationError(f"Duplicate dag_id {document.dag_id!r}")
        dag_ids.add(document.dag_id)
        root_entries = _mapping_entries(document.root, context=document.source_file)

        dag_node = builder.add_node(
            dag_id=document.dag_id,
            file_type="DAG",
            label=document.dag_id,
            source_file=document.source_file,
            source_location=_line(root_entries["dag_id"][0]),
        )
        document.dag_graph_id = dag_node["id"]

        node_sequence = root_entries["nodes"][1]
        assert isinstance(node_sequence, SequenceNode)
        nodes_data = document.data["nodes"]
        for index, (node_data, yaml_node) in enumerate(zip(nodes_data, node_sequence.value)):
            context = f"{document.source_file}: nodes[{index}]"
            node_data = _require_mapping_data(node_data, context=context)
            if not isinstance(yaml_node, MappingNode):
                raise GraphGenerationError(f"{context} must be a YAML mapping")
            node_entries = _mapping_entries(yaml_node, context=context)
            yaml_id = node_data.get("id")
            if "id" not in node_entries or not isinstance(yaml_id, str) or not yaml_id:
                raise GraphGenerationError(f"{context} requires a non-empty id")
            if yaml_id in document.node_lookup:
                raise GraphGenerationError(
                    f"Duplicate node id {yaml_id!r} in {document.source_file}"
                )
            if "op" not in node_entries:
                raise GraphGenerationError(
                    f"Node {yaml_id!r} is missing op in {document.source_file}"
                )

            graph_node = builder.add_node(
                dag_id=document.dag_id,
                file_type="DAG_NODE",
                label=yaml_id,
                source_file=document.source_file,
                source_location=_line(node_entries["id"][0]),
            )
            builder.add_link(document.dag_graph_id, graph_node["id"], "contains")
            document.node_lookup[yaml_id] = graph_node["id"]
            document.declarations.append(
                _NodeDeclaration(
                    data=node_data,
                    yaml_node=yaml_node,
                    yaml_id=yaml_id,
                    graph_id=graph_node["id"],
                )
            )

    # Pass two creates declaration children and all deterministic provider links.
    for document in documents:
        for declaration in document.declarations:
            _build_declaration_details(builder, document, declaration)

        root_entries = _mapping_entries(document.root, context=document.source_file)
        outputs_key, outputs_yaml = root_entries["outputs"]
        outputs = _require_mapping_data(
            document.data["outputs"], context=f"{document.source_file}: outputs"
        )
        output_entries = _mapping_entries(
            outputs_yaml, context=f"{document.source_file}: outputs"
        )
        if set(outputs) != set(output_entries):
            raise GraphGenerationError(
                f"Could not align outputs with YAML locations in {document.source_file}"
            )

        output_map = builder.add_node(
            dag_id=document.dag_id,
            file_type="OUTPUT_MAP",
            label="outputs",
            source_file=document.source_file,
            source_location=_line(outputs_key),
        )
        builder.add_link(document.dag_graph_id, output_map["id"], "contains")
        for name, (key_node, _) in output_entries.items():
            reference = outputs[name]
            if not isinstance(reference, str) or not reference:
                raise GraphGenerationError(
                    f"Output {name!r} in {document.source_file} requires a provider node id"
                )
            target = document.node_lookup.get(reference)
            if target is None:
                raise GraphGenerationError(
                    f"Output {name!r} in {document.source_file} references unknown node {reference!r}"
                )
            output_field = builder.add_node(
                dag_id=document.dag_id,
                file_type="OUTPUT_FIELD",
                label=name,
                source_file=document.source_file,
                source_location=_line(key_node),
            )
            builder.add_link(output_map["id"], output_field["id"], "contains")
            builder.add_link(output_field["id"], target, "uses")

    return builder.graph, builder.unresolved_nodes


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _format_jsonschema_error(error: Any) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return f"{path}: {error.message}"


def _validate_graph(
    graph: Any,
    schema_path: str | os.PathLike[str] | None = None,
) -> None:
    """Load the JSON Schema and validate critical graph invariants.

    When the optional ``jsonschema`` package is installed, the complete formal
    schema is applied.  The dependency-free checks below enforce the same core
    contract plus graph-level constraints such as endpoint existence and legal
    source/relation/target combinations.
    """

    schema_file = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphValidationError([f"Cannot load JSON Schema {schema_file}: {exc}"]) from exc

    errors: list[str] = []
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        try:
            validator_cls = jsonschema.validators.validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
            errors.extend(
                _format_jsonschema_error(error)
                for error in sorted(
                    validator.iter_errors(graph),
                    key=lambda item: tuple(str(part) for part in item.absolute_path),
                )
            )
        except jsonschema.exceptions.SchemaError as exc:
            raise GraphValidationError([f"Invalid JSON Schema {schema_file}: {exc.message}"]) from exc

    if not isinstance(graph, dict):
        errors.append("top level must be an object")
        raise GraphValidationError(_dedupe(errors))
    if set(graph) != {"nodes", "links"}:
        errors.append("top-level keys must be exactly 'nodes' and 'links'")
    nodes = graph.get("nodes")
    links = graph.get("links")
    if not isinstance(nodes, list):
        errors.append("nodes must be an array")
    if not isinstance(links, list):
        errors.append("links must be an array")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise GraphValidationError(_dedupe(errors))

    try:
        node_schema = schema["$defs"]["node"]
        link_schema = schema["$defs"]["link"]
        allowed_node_fields = set(node_schema["properties"])
        required_node_fields = set(node_schema["required"])
        allowed_file_types = set(node_schema["properties"]["file_type"]["enum"])
        id_pattern = re.compile(node_schema["properties"]["id"]["pattern"])
        location_pattern = re.compile(
            node_schema["properties"]["source_location"]["pattern"]
        )
        allowed_link_fields = set(link_schema["properties"])
        required_link_fields = set(link_schema["required"])
        allowed_relations = set(link_schema["properties"]["relation"]["enum"])
    except (KeyError, TypeError) as exc:
        raise GraphValidationError(
            [f"JSON Schema {schema_file} is missing required graph definitions: {exc}"]
        ) from exc

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        prefix = f"nodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra = set(node) - allowed_node_fields
        missing = required_node_fields - set(node)
        if extra:
            errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
        if missing:
            errors.append(f"{prefix} is missing fields: {sorted(missing)}")

        node_id = node.get("id")
        if not isinstance(node_id, str) or id_pattern.fullmatch(node_id) is None:
            errors.append(f"{prefix}.id is not a valid DAG/CODE UUIDv4 graph ID")
        elif node_id in nodes_by_id:
            errors.append(f"duplicate node id: {node_id}")
        else:
            nodes_by_id[node_id] = node

        file_type = node.get("file_type")
        if file_type not in allowed_file_types:
            errors.append(f"{prefix}.file_type is invalid: {file_type!r}")
        label = node.get("label", _MISSING)
        if file_type in _LABELED_NODE_TYPES and (
            not isinstance(label, str) or not label
        ):
            errors.append(f"{prefix}.label is required for {file_type}")
        elif label is not _MISSING and (not isinstance(label, str) or not label):
            errors.append(f"{prefix}.label must be a non-empty string when present")

        if file_type == "PARAM":
            if "value" not in node:
                errors.append(f"{prefix}.value is required for PARAM")
            elif not _is_json_value(node["value"]):
                errors.append(f"{prefix}.value is not a standard JSON-compatible value")
        elif "value" in node:
            errors.append(f"{prefix}.value is allowed only for PARAM")

        source_file = node.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            errors.append(f"{prefix}.source_file must be a non-empty string")
        source_location = node.get("source_location")
        if not isinstance(source_location, str) or location_pattern.fullmatch(source_location) is None:
            errors.append(f"{prefix}.source_location must have form L<number>")

    link_triples: set[tuple[str, str, str]] = set()
    structural_parents: dict[str, list[str]] = {}
    typed_outgoing: dict[tuple[str, str], list[str]] = {}
    for index, link in enumerate(links):
        prefix = f"links[{index}]"
        if not isinstance(link, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra = set(link) - allowed_link_fields
        missing = required_link_fields - set(link)
        if extra:
            errors.append(f"{prefix} has unsupported fields: {sorted(extra)}")
        if missing:
            errors.append(f"{prefix} is missing fields: {sorted(missing)}")
        if extra or missing:
            continue

        source = link["source"]
        target = link["target"]
        relation = link["relation"]
        if not all(isinstance(item, str) and item for item in (source, target, relation)):
            errors.append(f"{prefix} values must be non-empty strings")
            continue
        if relation not in allowed_relations:
            errors.append(f"{prefix}.relation is invalid: {relation!r}")
        triple = (source, relation, target)
        if triple in link_triples:
            errors.append(f"duplicate link: {triple}")
        link_triples.add(triple)

        source_node = nodes_by_id.get(source)
        target_node = nodes_by_id.get(target)
        if source_node is None:
            errors.append(f"{prefix}.source does not exist: {source}")
        if target_node is None:
            errors.append(f"{prefix}.target does not exist: {target}")
        if source_node is None or target_node is None or relation not in allowed_relations:
            continue
        signature = (
            source_node["file_type"],
            relation,
            target_node["file_type"],
        )
        if signature not in _ALLOWED_LINK_TYPES:
            errors.append(f"{prefix} has unsupported endpoint types: {signature}")
        if relation == "contains":
            structural_parents.setdefault(target, []).append(source)
        typed_outgoing.setdefault((source, relation), []).append(target)

    for node_id, node in nodes_by_id.items():
        file_type = node.get("file_type")
        parent_count = len(structural_parents.get(node_id, []))
        if file_type not in {"DAG", "CODE"} and parent_count != 1:
            errors.append(
                f"node {node_id} ({file_type}) must have exactly one contains parent; found {parent_count}"
            )
        if file_type == "DAG_NODE":
            operation_count = sum(
                nodes_by_id[target].get("file_type") == "OPERATION"
                for target in typed_outgoing.get((node_id, "contains"), [])
                if target in nodes_by_id
            )
            if operation_count != 1:
                errors.append(
                    f"DAG_NODE {node_id} must contain exactly one OPERATION; found {operation_count}"
                )
        elif file_type == "DAG":
            output_map_count = sum(
                nodes_by_id[target].get("file_type") == "OUTPUT_MAP"
                for target in typed_outgoing.get((node_id, "contains"), [])
                if target in nodes_by_id
            )
            if output_map_count != 1:
                errors.append(
                    f"DAG {node_id} must contain exactly one OUTPUT_MAP; found {output_map_count}"
                )
        elif file_type in {"DERIVED_PARAM", "OUTPUT_FIELD"}:
            use_count = len(typed_outgoing.get((node_id, "uses"), []))
            if use_count != 1:
                errors.append(
                    f"{file_type} {node_id} must have exactly one uses link; found {use_count}"
                )
        elif file_type == "INPUT_FIELD":
            use_count = len(typed_outgoing.get((node_id, "uses"), []))
            if use_count > 1:
                errors.append(f"INPUT_FIELD {node_id} has more than one uses link")
        elif file_type == "OPERATION":
            if len(typed_outgoing.get((node_id, "references"), [])) > 1:
                errors.append(f"OPERATION {node_id} has more than one references link")
        elif file_type == "CFG_PARAM":
            if len(typed_outgoing.get((node_id, "uses"), [])) > 1:
                errors.append(f"CFG_PARAM {node_id} has more than one uses link")

    try:
        json.dumps(graph, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"graph is not strict JSON: {exc}")

    if errors:
        raise GraphValidationError(_dedupe(errors))


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _resolve_from_root(source_root: Path, path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else source_root / candidate


def _atomic_write_graph(
    graph: dict[str, Any],
    output_path: Path,
    schema_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                graph,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        with temporary_path.open("r", encoding="utf-8") as handle:
            written_graph = json.load(handle)
        _validate_graph(written_graph, schema_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def generate_graph(
    source_root: str | os.PathLike[str] | None = None,
    *,
    input_glob: str | None = None,
    output_path: str | os.PathLike[str] | None = None,
    schema_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish a base DAG graph.

    The returned ``unresolved_nodes`` are diagnostics for later agent work and
    are never inserted into ``dag_graph.json``.
    """

    root = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else DEFAULT_SOURCE_ROOT
    )
    if not root.is_dir():
        raise GraphGenerationError(f"Source root is not a directory: {root}")
    resolved_schema = (
        _resolve_from_root(root, schema_path).resolve()
        if schema_path is not None
        else DEFAULT_SCHEMA_PATH
    )
    resolved_input_glob = (
        input_glob
        if input_glob is not None
        else str(root / "model_diagnostic" / "config" / "*.yaml")
    )
    resolved_output = (
        _resolve_from_root(root, output_path).resolve()
        if output_path is not None
        else (root / "generated" / "source_tree" / "dag_graph.json").resolve()
    )
    input_paths = _collect_input_files(root, resolved_input_glob)
    documents = [_load_document(path, root) for path in input_paths]
    graph, unresolved_nodes = _build_graph(documents)

    # Validate in memory first, then validate the serialized temporary file
    # before replacing the configured output.
    _validate_graph(graph, resolved_schema)
    _atomic_write_graph(graph, resolved_output, resolved_schema)

    graph_report = {
        "source_root": str(root),
        "output_path": str(resolved_output),
        "node_count": len(graph["nodes"]),
        "link_count": len(graph["links"]),
        "unresolved_count": len(unresolved_nodes),
        "unresolved_nodes": unresolved_nodes,
    }
    #TODO save graph_report
    print(json.dumps(graph_report))
    return graph_report

def get_graph_report() -> str :
    # load file and return as JSON
    return ""



def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a validated flat DAG graph from YAML files."
    )
    parser.add_argument(
        "--source-root",
        "--project-root",
        dest="source_root",
        default=str(DEFAULT_SOURCE_ROOT),
        help=(
            "Python source root. --project-root is retained as a deprecated "
            "alias, but must also point directly to the source folder."
        ),
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help=f"YAML glob; default for the inferred source root is {DEFAULT_INPUT_GLOB}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Graph output; default for the inferred source root is {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--schema", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    try:
        result = generate_graph(
            arguments.source_root,
            input_glob=arguments.input_glob,
            output_path=arguments.output,
            schema_path=arguments.schema,
        )
    except (GraphGenerationError, GraphValidationError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    #print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
