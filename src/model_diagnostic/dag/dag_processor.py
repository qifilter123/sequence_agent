from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from model_diagnostic.dag.operation_registry import OperationRegistry


class DagConfigError(ValueError):
    """Raised when a DAG configuration is structurally invalid."""


class DagExecutionError(RuntimeError):
    """Raised when a DAG node cannot be executed."""


class DagProcessor:
    """Execute declarative DAG configs using a supplied operation registry.

    The processor owns graph orchestration only: config validation, dependency
    ordering, input resolution, operation dispatch, and named output assembly.
    Tensor/feature-specific behavior belongs in registered operations.
    """

    def __init__(self, registry: OperationRegistry) -> None:
        self.registry = registry

    def load_config(self, path: str | Path) -> dict[str, Any]:
        """Load and validate one YAML DAG config.

        Call this during initialization, not inside a per-batch training loop.
        """
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"DAG config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        if not isinstance(config, dict):
            raise DagConfigError(
                f"DAG config must be a YAML mapping: {config_path}"
            )

        self.validate_config(config)
        # Isolate runtime callers from accidental mutation of the parsed config.
        return deepcopy(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        dag_id = config.get("dag_id")
        if not isinstance(dag_id, str) or not dag_id:
            raise DagConfigError("DAG config requires non-empty 'dag_id'")

        version = config.get("version")
        if version != 1:
            raise DagConfigError(
                f"Unsupported DAG version for '{dag_id}': {version!r}; expected 1"
            )

        nodes = config.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise DagConfigError(f"DAG '{dag_id}' requires a non-empty 'nodes' list")

        seen: set[str] = set()
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise DagConfigError(
                    f"DAG '{dag_id}' node #{index} must be a mapping"
                )

            node_id = node.get("id")
            op_name = node.get("op")
            if not isinstance(node_id, str) or not node_id:
                raise DagConfigError(
                    f"DAG '{dag_id}' node #{index} requires non-empty 'id'"
                )
            if node_id in seen:
                raise DagConfigError(
                    f"DAG '{dag_id}' has duplicate node id '{node_id}'"
                )
            seen.add(node_id)

            if not isinstance(op_name, str) or not op_name:
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' requires non-empty 'op'"
                )
            try:
                self.registry.get(op_name)
            except KeyError as exc:
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' references unknown op '{op_name}'"
                ) from exc

            inputs = node.get("inputs", {})
            params = node.get("params", {})
            if not isinstance(inputs, dict):
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' inputs must be a mapping"
                )
            if not isinstance(params, dict):
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' params must be a mapping"
                )

        outputs = config.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise DagConfigError(
                f"DAG '{dag_id}' requires a non-empty 'outputs' mapping"
            )
        for output_name, source in outputs.items():
            if not isinstance(output_name, str) or not output_name:
                raise DagConfigError(
                    f"DAG '{dag_id}' output names must be non-empty strings"
                )
            if not isinstance(source, str) or not source:
                raise DagConfigError(
                    f"DAG '{dag_id}' output '{output_name}' must reference one source name"
                )

        # Detect cycles between nodes before execution. External inputs are
        # deliberately resolved at runtime because they vary by caller.
        self._topological_nodes(config)

    def run(
        self,
        config: dict[str, Any],
        inputs: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a validated DAG and return its configured named outputs."""
        if not isinstance(inputs, dict):
            raise TypeError("DAG inputs must be a dictionary")

        runtime_context = {} if runtime is None else dict(runtime)
        context: dict[str, Any] = dict(inputs)
        dag_id = config["dag_id"]

        for node in self._topological_nodes(config):
            node_id = node["id"]
            op_name = node["op"]

            try:
                resolved_inputs = self._resolve_input_spec(
                    node.get("inputs", {}),
                    context,
                    dag_id=dag_id,
                    node_id=node_id,
                )
                operation = self.registry.get(op_name)
                result = operation(
                    resolved_inputs,
                    dict(node.get("params", {})),
                    runtime_context,
                )
            except Exception as exc:
                if isinstance(exc, (DagConfigError, DagExecutionError)):
                    raise
                raise DagExecutionError(
                    f"DAG '{dag_id}' node '{node_id}' (op='{op_name}') failed: {exc}"
                ) from exc

            context[node_id] = result

        result: dict[str, Any] = {}
        for output_name, source_name in config["outputs"].items():
            if source_name not in context:
                raise DagExecutionError(
                    f"DAG '{dag_id}' output '{output_name}' references unavailable "
                    f"source '{source_name}'"
                )
            result[output_name] = context[source_name]

        return result

    def _topological_nodes(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = config["nodes"]
        by_id = {node["id"]: node for node in nodes}
        node_ids = set(by_id)

        dependencies: dict[str, set[str]] = {}
        for node in nodes:
            refs = set(self._iter_source_refs(node.get("inputs", {})))
            dependencies[node["id"]] = refs & node_ids

        ordered: list[dict[str, Any]] = []
        ready = [node["id"] for node in nodes if not dependencies[node["id"]]]
        completed: set[str] = set()

        while ready:
            current = ready.pop(0)
            if current in completed:
                continue
            ordered.append(by_id[current])
            completed.add(current)

            for node in nodes:
                node_id = node["id"]
                if node_id in completed or node_id in ready:
                    continue
                if dependencies[node_id] <= completed:
                    ready.append(node_id)

        if len(ordered) != len(nodes):
            blocked = sorted(node_ids - completed)
            raise DagConfigError(
                f"DAG '{config.get('dag_id')}' contains a dependency cycle among: "
                + ", ".join(blocked)
            )

        return ordered

    def _iter_source_refs(self, spec: Any):
        if isinstance(spec, str):
            yield spec
        elif isinstance(spec, list):
            for item in spec:
                yield from self._iter_source_refs(item)
        elif isinstance(spec, dict):
            for value in spec.values():
                yield from self._iter_source_refs(value)

    def _resolve_input_spec(
        self,
        spec: Any,
        context: dict[str, Any],
        *,
        dag_id: str,
        node_id: str,
    ) -> Any:
        """Resolve source-name references recursively.

        YAML node ``inputs`` are declarative references, not Python expressions.
        Strings always mean context source names; lists/dicts preserve structure.
        Literal operation parameters belong under ``params``.
        """
        if isinstance(spec, str):
            if spec not in context:
                raise DagExecutionError(
                    f"DAG '{dag_id}' node '{node_id}' requires unavailable input "
                    f"'{spec}'. Available sources: {', '.join(sorted(context))}"
                )
            return context[spec]

        if isinstance(spec, list):
            return [
                self._resolve_input_spec(
                    item,
                    context,
                    dag_id=dag_id,
                    node_id=node_id,
                )
                for item in spec
            ]

        if isinstance(spec, dict):
            return {
                key: self._resolve_input_spec(
                    value,
                    context,
                    dag_id=dag_id,
                    node_id=node_id,
                )
                for key, value in spec.items()
            }

        raise DagConfigError(
            f"DAG '{dag_id}' node '{node_id}' input references must be strings, "
            "lists, or mappings. Put literal values under 'params'."
        )
