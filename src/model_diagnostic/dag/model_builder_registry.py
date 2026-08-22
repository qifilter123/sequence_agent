from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch.nn as nn

from model_diagnostic.dag.dag_processor import DagConfigError
from model_diagnostic.dag.operation_registry import OperationRegistry


MODEL_BUILDER_REGISTRY = OperationRegistry()


BUILDER_FIELDS_KEY = "fields"


@dataclass(frozen=True)
class ModelInput:
    """Symbolic external input used while the model-builder DAG is running."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelInput name must be non-empty")


@dataclass
class BuiltModelNode:
    """One symbolic model node produced by a registered model builder operation.

    Builder operations construct PyTorch modules but do not execute tensor flow.
    Inputs remain symbolic until ``root_model`` compiles the dependency graph into
    the final executable torch.nn.Module.
    """

    node_id: str
    op_name: str
    module: nn.Module
    inputs: dict[str, Any]
    params: dict[str, Any]


@dataclass(frozen=True)
class ModelInputRef:
    """Compiled reference to one external input of a DAG-built model."""

    name: str


@dataclass(frozen=True)
class ModelNodeRef:
    """Compiled reference to the output of another DAG-built model node."""

    node_id: str


@dataclass(frozen=True)
class ModelExecutionNode:
    """Runtime execution instruction produced by the model-builder DAG."""

    node_id: str
    op_name: str
    inputs: dict[str, Any]


@dataclass(frozen=True)
class CompiledModelGraph:
    """Build-time result consumed by the root_model operation."""

    modules: dict[str, nn.Module]
    execution_plan: list[ModelExecutionNode]
    output_ref: Any
    input_names: list[str]
    node_metadata: dict[str, dict[str, Any]]


def symbolic_input(name: str) -> ModelInput:
    return ModelInput(name)


def require_node_id(runtime: dict[str, Any]) -> str:
    node_id = runtime.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("Model builder operation requires runtime['node_id']")
    if "." in node_id:
        raise DagConfigError(
            f"Model DAG node id '{node_id}' cannot contain '.' because it becomes "
            "an nn.ModuleDict key"
        )
    return node_id


def _validate_builder_inputs(
    inputs: dict[str, Any],
    *,
    node_id: str,
) -> None:
    """Validate the Builder-DAG input calling contract.

    ``inputs.fields`` is reserved by the Builder DAG layer. When present it
    must be the only input key and must contain a non-empty ordered list. The
    compiled PyTorch runtime expands that list as positional arguments.

    A mapping without ``fields`` is a semantic keyword-binding contract and is
    passed to the downstream module as ``module(**inputs)``.
    """
    if BUILDER_FIELDS_KEY not in inputs:
        return

    if set(inputs) != {BUILDER_FIELDS_KEY}:
        other = sorted(set(inputs) - {BUILDER_FIELDS_KEY})
        raise DagConfigError(
            f"Builder DAG node '{node_id}' cannot mix reserved input 'fields' "
            f"with named inputs: {', '.join(other)}"
        )

    fields = inputs[BUILDER_FIELDS_KEY]
    if not isinstance(fields, list) or not fields:
        raise DagConfigError(
            f"Builder DAG node '{node_id}' reserved input 'fields' must be "
            "a non-empty ordered list"
        )


def build_node(
    *,
    op_name: str,
    module: nn.Module,
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> BuiltModelNode:
    if not isinstance(module, nn.Module):
        raise TypeError(f"Model builder op '{op_name}' must construct nn.Module")

    node_id = require_node_id(runtime)
    _validate_builder_inputs(inputs, node_id=node_id)

    return BuiltModelNode(
        node_id=node_id,
        op_name=op_name,
        module=module,
        inputs=dict(inputs),
        params=dict(params),
    )


def _iter_dependencies(spec: Any) -> Iterable[BuiltModelNode]:
    if isinstance(spec, BuiltModelNode):
        yield spec
    elif isinstance(spec, list):
        for item in spec:
            yield from _iter_dependencies(item)
    elif isinstance(spec, dict):
        for item in spec.values():
            yield from _iter_dependencies(item)


def _compile_ref(spec: Any) -> Any:
    if isinstance(spec, ModelInput):
        return ModelInputRef(spec.name)
    if isinstance(spec, BuiltModelNode):
        return ModelNodeRef(spec.node_id)
    if isinstance(spec, list):
        return [_compile_ref(item) for item in spec]
    if isinstance(spec, dict):
        return {key: _compile_ref(value) for key, value in spec.items()}
    raise TypeError(
        "Built model inputs must contain ModelInput/BuiltModelNode references, "
        f"lists, or mappings; got {type(spec).__name__}"
    )


def _collect_external_inputs(spec: Any, names: list[str]) -> None:
    if isinstance(spec, ModelInput):
        if spec.name not in names:
            names.append(spec.name)
        return
    if isinstance(spec, BuiltModelNode):
        for value in spec.inputs.values():
            _collect_external_inputs(value, names)
        return
    if isinstance(spec, list):
        for item in spec:
            _collect_external_inputs(item, names)
        return
    if isinstance(spec, dict):
        for item in spec.values():
            _collect_external_inputs(item, names)


def compile_model_graph(root_inputs: dict[str, Any]) -> CompiledModelGraph:
    """Compile symbolic root inputs into an executable model plan.

    The root node adds no model computation. Its input mapping names the exposed
    model outputs. External model inputs are discovered from ModelInput objects.
    With one root input, forward() returns that value directly. With multiple
    root inputs, forward() returns a mapping preserving the root input names.
    """
    if not isinstance(root_inputs, dict) or not root_inputs:
        raise DagConfigError("root_model requires at least one input")

    ordered: list[BuiltModelNode] = []
    visiting: set[str] = set()
    completed: set[str] = set()

    def visit(node: BuiltModelNode) -> None:
        if node.node_id in completed:
            return
        if node.node_id in visiting:
            raise DagConfigError(
                f"Cycle detected while assembling model at node '{node.node_id}'"
            )

        # Validate again at graph-compile boundary so custom builder operations
        # that construct BuiltModelNode directly cannot bypass the contract.
        _validate_builder_inputs(node.inputs, node_id=node.node_id)

        visiting.add(node.node_id)
        for dependency in _iter_dependencies(node.inputs):
            visit(dependency)
        visiting.remove(node.node_id)
        completed.add(node.node_id)
        ordered.append(node)

    for value in root_inputs.values():
        if not isinstance(value, (BuiltModelNode, ModelInput)):
            raise TypeError(
                "root_model inputs must reference BuiltModelNode or ModelInput; "
                f"got {type(value).__name__}"
            )
        if isinstance(value, BuiltModelNode):
            visit(value)

    discovered_inputs: list[str] = []
    _collect_external_inputs(root_inputs, discovered_inputs)

    modules = {node.node_id: node.module for node in ordered}
    execution_plan = [
        ModelExecutionNode(
            node_id=node.node_id,
            op_name=node.op_name,
            inputs=_compile_ref(node.inputs),
        )
        for node in ordered
    ]
    node_metadata = {
        node.node_id: {
            "op": node.op_name,
            "params": dict(node.params),
        }
        for node in ordered
    }

    if len(root_inputs) == 1:
        output_ref = _compile_ref(next(iter(root_inputs.values())))
    else:
        output_ref = _compile_ref(root_inputs)

    return CompiledModelGraph(
        modules=modules,
        execution_plan=execution_plan,
        output_ref=output_ref,
        input_names=discovered_inputs,
        node_metadata=node_metadata,
    )


def iter_trainable_model_nodes(model: nn.Module):
    """Yield DAG node modules that own trainable parameters."""
    modules = getattr(model, "model_nodes", None)
    if not isinstance(modules, nn.ModuleDict):
        return
    for node_id, module in modules.items():
        if any(param.requires_grad for param in module.parameters()):
            yield node_id, module
