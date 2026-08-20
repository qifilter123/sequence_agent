from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch.nn as nn

from model_diagnostic.dag.dag_processor import DagConfigError
from model_diagnostic.dag.operation_registry import OperationRegistry
from model_diagnostic.generic_model_ops import (
    ContinuousTimeEmbedding,
    DagTorchModel,
    GRUTransform,
    ModelExecutionNode,
    ModelInputRef,
    ModelNodeRef,
    ResidualAdd,
    TensorConcat,
)


MODEL_BUILDER_REGISTRY = OperationRegistry()


@dataclass(frozen=True)
class ModelInput:
    """Symbolic external input used while the model-builder DAG is running."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelInput name must be non-empty")


@dataclass
class BuiltModelNode:
    """One symbolic model node produced by MODEL_BUILDER_REGISTRY.

    Builder nodes construct PyTorch modules but do not execute tensor flow.
    Inputs remain symbolic until the terminal ``root_model`` node compiles the
    dependency graph into a ``DagTorchModel``.
    """

    node_id: str
    op_name: str
    module: nn.Module
    inputs: dict[str, Any]
    params: dict[str, Any]


def symbolic_input(name: str) -> ModelInput:
    return ModelInput(name)


def require_cfg(runtime: dict[str, Any]) -> Any:
    cfg = runtime.get("cfg")
    if cfg is None:
        raise ValueError("Model builder operation requires runtime['cfg']")
    return cfg


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


def resolve_param(value: Any, cfg: Any) -> Any:
    """Resolve safe declarative model parameter expressions.

    Supported expressions:
      {cfg: hidden_dim}
      {add: [{cfg: input_base_dim}, {cfg: time_emb_dim}]}
    """
    if isinstance(value, list):
        return [resolve_param(item, cfg) for item in value]

    if not isinstance(value, dict):
        return value

    if set(value) == {"cfg"}:
        key = value["cfg"]
        if not isinstance(key, str) or not key:
            raise DagConfigError("cfg parameter reference must be a non-empty string")
        if not hasattr(cfg, key):
            raise DagConfigError(f"Model config does not define attribute '{key}'")
        return getattr(cfg, key)

    if set(value) == {"add"}:
        items = value["add"]
        if not isinstance(items, list) or not items:
            raise DagConfigError("add parameter expression requires a non-empty list")
        return sum(resolve_param(item, cfg) for item in items)

    return {key: resolve_param(item, cfg) for key, item in value.items()}


def resolve_params(params: dict[str, Any], cfg: Any) -> dict[str, Any]:
    return {name: resolve_param(value, cfg) for name, value in params.items()}


def _params(params: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return resolve_params(params, require_cfg(runtime))


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
    return BuiltModelNode(
        node_id=require_node_id(runtime),
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


def iter_trainable_model_nodes(model: nn.Module):
    """Yield DAG node modules that own trainable parameters."""
    modules = getattr(model, "model_nodes", None)
    if not isinstance(modules, nn.ModuleDict):
        return
    for node_id, module in modules.items():
        if any(param.requires_grad for param in module.parameters()):
            yield node_id, module


@MODEL_BUILDER_REGISTRY.register("temporal_embedding")
def build_temporal_embedding(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = ContinuousTimeEmbedding(
        out_dim=int(resolved["out_dim"]),
        source_index=int(resolved.get("source_index", 0)),
    )
    return build_node(
        op_name="temporal_embedding",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("linear_transform")
def build_linear_transform(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = nn.Linear(
        int(resolved["input_dim"]),
        int(resolved["output_dim"]),
        bias=bool(resolved.get("bias", True)),
    )
    return build_node(
        op_name="linear_transform",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("activation")
def build_activation(inputs, params, runtime):
    resolved = _params(params, runtime)
    name = str(resolved.get("activation_name", "relu")).lower()
    builders = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    if name not in builders:
        raise ValueError(f"Unsupported activation_name '{name}'")
    module = builders[name]()
    return build_node(
        op_name="activation",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("dropout")
def build_dropout(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = nn.Dropout(float(resolved["dropout_rate"]))
    return build_node(
        op_name="dropout",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("GRU")
def build_gru(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = GRUTransform(
        input_dim=int(resolved["input_dim"]),
        hidden_dim=int(resolved["hidden_dim"]),
        batch_first=bool(resolved.get("batch_first", True)),
        num_layers=int(resolved.get("num_layers", 1)),
        bidirectional=bool(resolved.get("bidirectional", False)),
    )
    return build_node(
        op_name="GRU",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("LayerNorm")
def build_layer_norm(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = nn.LayerNorm(int(resolved["normalized_shape"]))
    return build_node(
        op_name="LayerNorm",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("concat")
def build_concat(inputs, params, runtime):
    resolved = _params(params, runtime)
    module = TensorConcat(dim=int(resolved.get("dim", -1)))
    return build_node(
        op_name="concat",
        module=module,
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("residual")
def build_residual(inputs, params, runtime):
    resolved = _params(params, runtime)
    return build_node(
        op_name="residual",
        module=ResidualAdd(),
        inputs=inputs,
        params=resolved,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("root_model")
def build_root_model(inputs, params, runtime):
    """Compile symbolic builder nodes into the final executable nn.Module.

    ``root_model`` is intentionally a normal registry operation. DagProcessor
    resolves and invokes it exactly like every other model-builder node; the
    only distinction is that its output type is the final ``DagTorchModel``.
    """
    resolved = _params(params, runtime)
    input_order = resolved.get("input_order")
    if input_order is not None and not isinstance(input_order, list):
        raise TypeError("root_model params.input_order must be a list")

    output = inputs["output"]
    if not isinstance(output, (BuiltModelNode, ModelInput)):
        raise TypeError("root_model output must be BuiltModelNode or ModelInput")

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
        visiting.add(node.node_id)
        for dependency in _iter_dependencies(node.inputs):
            visit(dependency)
        visiting.remove(node.node_id)
        completed.add(node.node_id)
        ordered.append(node)

    if isinstance(output, BuiltModelNode):
        visit(output)

    discovered_inputs: list[str] = []
    _collect_external_inputs(output, discovered_inputs)

    if input_order is None:
        effective_inputs = discovered_inputs
    else:
        effective_inputs = list(input_order)
        if set(effective_inputs) != set(discovered_inputs):
            raise DagConfigError(
                "root_model input_order does not match discovered model inputs: "
                f"configured={effective_inputs}, discovered={discovered_inputs}"
            )

    modules = {node.node_id: node.module for node in ordered}
    execution_plan = [
        ModelExecutionNode(
            node_id=node.node_id,
            op_name=node.op_name,
            inputs=_compile_ref(node.inputs),
        )
        for node in ordered
    ]
    metadata = {
        node.node_id: {
            "op": node.op_name,
            "params": dict(node.params),
        }
        for node in ordered
    }

    return DagTorchModel(
        modules=modules,
        execution_plan=execution_plan,
        output_ref=_compile_ref(output),
        input_names=effective_inputs,
        node_metadata=metadata,
    )
