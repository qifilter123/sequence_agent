from __future__ import annotations

from collections.abc import Mapping
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
    ordering, input resolution, parameter binding, operation dispatch, and
    output assembly. Domain-specific behavior belongs in registered operations.

    Framework contract for node configuration:
      inputs:
        DAG data dependencies. Mapping/list structure is preserved while source
        names are resolved from the DAG context. If the ``inputs`` key is
        omitted entirely, the operation receives the original external inputs
        passed to ``run``. An explicit ``inputs: {}`` remains an empty mapping.

      params:
        Literal operation parameters written directly in YAML.

      cfg_params:
        ``operation_param: cfg_attribute`` bindings. Every right-hand value is
        the name of one attribute on ``runtime['cfg']``. These values are
        resolved by DagProcessor before the operation is called.

    A parameter name may not appear in both ``params`` and ``cfg_params``.

    DAG outputs:
      If ``outputs`` is configured, it is used exactly as declared. If omitted,
      the final node result is returned directly when it is a mapping; otherwise
      it is returned as ``{final_node_id: result}``.
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

            if "inputs" in node:
                inputs = node["inputs"]
                if not isinstance(inputs, (dict, list)):
                    raise DagConfigError(
                        f"DAG '{dag_id}' node '{node_id}' inputs must be a mapping or list"
                    )
                if isinstance(inputs, list):
                    if not inputs:
                        raise DagConfigError(
                            f"DAG '{dag_id}' node '{node_id}' input list must be non-empty"
                        )
                    if not all(isinstance(name, str) and name for name in inputs):
                        raise DagConfigError(
                            f"DAG '{dag_id}' node '{node_id}' top-level input list must "
                            "contain non-empty source names"
                        )
                    if len(set(inputs)) != len(inputs):
                        raise DagConfigError(
                            f"DAG '{dag_id}' node '{node_id}' top-level input list "
                            "contains duplicate source names"
                        )

            params = node.get("params", {})
            if not isinstance(params, dict):
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' params must be a mapping"
                )

            cfg_params = node.get("cfg_params", {})
            if not isinstance(cfg_params, dict):
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' cfg_params must be a mapping"
                )
            for param_name, cfg_name in cfg_params.items():
                if not isinstance(param_name, str) or not param_name:
                    raise DagConfigError(
                        f"DAG '{dag_id}' node '{node_id}' cfg_params keys must be "
                        "non-empty strings"
                    )
                if not isinstance(cfg_name, str) or not cfg_name:
                    raise DagConfigError(
                        f"DAG '{dag_id}' node '{node_id}' cfg_params['{param_name}'] "
                        "must name one runtime cfg attribute"
                    )

            overlap = set(params) & set(cfg_params)
            if overlap:
                raise DagConfigError(
                    f"DAG '{dag_id}' node '{node_id}' defines the same parameter in "
                    f"params and cfg_params: {', '.join(sorted(overlap))}"
                )

        outputs = config.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, dict) or not outputs:
                raise DagConfigError(
                    f"DAG '{dag_id}' outputs must be a non-empty mapping when provided"
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

        self._topological_nodes(config)

    def run(
        self,
        config: dict[str, Any],
        inputs: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a validated DAG and return named outputs."""
        if not isinstance(inputs, dict):
            raise TypeError("DAG inputs must be a dictionary")

        runtime_context = {} if runtime is None else dict(runtime)
        external_inputs: dict[str, Any] = dict(inputs)
        context: dict[str, Any] = dict(inputs)
        dag_id = config["dag_id"]
        ordered_nodes = self._topological_nodes(config)

        for node in ordered_nodes:
            node_id = node["id"]
            op_name = node["op"]

            try:
                if "inputs" not in node:
                    # Generic shorthand for operations that consume the DAG's
                    # original external input contract. Do not use the growing
                    # execution context here, otherwise hidden node dependencies
                    # would be introduced.
                    resolved_inputs = dict(external_inputs)
                else:
                    resolved_inputs = self._resolve_node_inputs(
                        node["inputs"],
                        context,
                        dag_id=dag_id,
                        node_id=node_id,
                    )

                operation_params = dict(node.get("params", {}))
                operation_params.update(
                    self._resolve_cfg_params(
                        node.get("cfg_params", {}),
                        runtime_context,
                        dag_id=dag_id,
                        node_id=node_id,
                    )
                )

                operation = self.registry.get(op_name)
                node_runtime = dict(runtime_context)
                node_runtime["dag_id"] = dag_id
                node_runtime["node_id"] = node_id
                result = operation(
                    resolved_inputs,
                    operation_params,
                    node_runtime,
                )
            except Exception as exc:
                if isinstance(exc, (DagConfigError, DagExecutionError)):
                    raise
                raise DagExecutionError(
                    f"DAG '{dag_id}' node '{node_id}' (op='{op_name}') failed: {exc}"
                ) from exc

            context[node_id] = result

        configured_outputs = config.get("outputs")
        if configured_outputs is not None:
            result: dict[str, Any] = {}
            for output_name, source_name in configured_outputs.items():
                if source_name not in context:
                    raise DagExecutionError(
                        f"DAG '{dag_id}' output '{output_name}' references unavailable "
                        f"source '{source_name}'"
                    )
                result[output_name] = context[source_name]
            return result

        final_node_id = ordered_nodes[-1]["id"]
        final_result = context[final_node_id]
        if isinstance(final_result, Mapping):
            return dict(final_result)
        return {final_node_id: final_result}

    def _topological_nodes(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = config["nodes"]
        by_id = {node["id"]: node for node in nodes}
        node_ids = set(by_id)

        dependencies: dict[str, set[str]] = {}
        for node in nodes:
            # Omitted inputs mean external DAG inputs only, so they create no
            # dependencies on other DAG nodes.
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

    def _resolve_node_inputs(
        self,
        spec: dict[str, Any] | list[str],
        context: dict[str, Any],
        *,
        dag_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        """Resolve one node's explicit input declaration.

        A top-level list is shorthand for selecting same-named context values:

            inputs: [mask, dt]

        becomes:

            {"mask": context["mask"], "dt": context["dt"]}

        Mapping inputs retain alias/nested-structure semantics.
        """
        if isinstance(spec, list):
            resolved: dict[str, Any] = {}
            for source_name in spec:
                if source_name not in context:
                    raise DagExecutionError(
                        f"DAG '{dag_id}' node '{node_id}' requires unavailable input "
                        f"'{source_name}'. Available sources: {', '.join(sorted(context))}"
                    )
                resolved[source_name] = context[source_name]
            return resolved

        resolved = self._resolve_input_spec(
            spec,
            context,
            dag_id=dag_id,
            node_id=node_id,
        )
        if not isinstance(resolved, dict):
            raise DagConfigError(
                f"DAG '{dag_id}' node '{node_id}' resolved inputs must be a mapping"
            )
        return resolved

    def _resolve_input_spec(
        self,
        spec: Any,
        context: dict[str, Any],
        *,
        dag_id: str,
        node_id: str,
    ) -> Any:
        """Resolve source-name references recursively for mapping-style inputs."""
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

    def _resolve_cfg_params(
        self,
        cfg_params: dict[str, str],
        runtime: dict[str, Any],
        *,
        dag_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        """Resolve cfg_params using the strict ``param: cfg_attribute`` contract."""
        if not cfg_params:
            return {}

        cfg = runtime.get("cfg")
        if cfg is None:
            raise DagExecutionError(
                f"DAG '{dag_id}' node '{node_id}' declares cfg_params but "
                "runtime['cfg'] is unavailable"
            )

        resolved: dict[str, Any] = {}
        for param_name, cfg_name in cfg_params.items():
            if not hasattr(cfg, cfg_name):
                raise DagExecutionError(
                    f"DAG '{dag_id}' node '{node_id}' cfg_params['{param_name}'] "
                    f"references missing cfg attribute '{cfg_name}'"
                )
            resolved[param_name] = getattr(cfg, cfg_name)
        return resolved
