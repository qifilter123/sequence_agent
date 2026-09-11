from __future__ import annotations

from numbers import Number
from typing import Any

import torch
import torch.nn as nn

from model_diagnostic.dag.dag_processor import DagConfigError
from model_diagnostic.dag.model_builder_registry import (
    BUILDER_FIELDS_KEY,
    MODEL_BUILDER_REGISTRY,
    ModelExecutionNode,
    ModelInputRef,
    ModelNodeRef,
    build_node,
    compile_model_graph,
)


class ContinuousTimeEmbedding(nn.Module):
    """Learnable sinusoidal embedding of one continuous feature in x_full."""

    def __init__(self, out_dim: int, source_index: int = 0) -> None:
        super().__init__()
        if out_dim <= 0 or out_dim % 2 != 0:
            raise ValueError("out_dim must be a positive even integer")
        self.out_dim = int(out_dim)
        self.source_index = int(source_index)
        self.omega = nn.Parameter(torch.randn(self.out_dim // 2))
        self.phi = nn.Parameter(torch.randn(self.out_dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x[..., self.source_index]
        scaled = dt.unsqueeze(-1) * self.omega + self.phi
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class GRUTransform(nn.Module):
    """GRU node whose forward result is the sequence output tensor only."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        batch_first: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            batch_first=bool(batch_first),
            num_layers=int(num_layers),
            bidirectional=bool(bidirectional),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class TensorConcat(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, *fields: torch.Tensor) -> torch.Tensor:
        if not fields:
            raise ValueError("concat requires at least one tensor field")
        return torch.cat(fields, dim=self.dim)


class ResidualAdd(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if x.shape != residual.shape:
            raise ValueError(
                "Residual tensors must have identical shapes: "
                f"x={tuple(x.shape)}, residual={tuple(residual.shape)}"
            )
        return x + residual


class DagTorchModel(nn.Module):
    """Executable PyTorch model assembled by the model-builder DAG.

    DagProcessor is used only at build time. Once constructed, normal PyTorch
    training/inference calls forward() directly and executes this compiled plan.
    """

    def __init__(
        self,
        *,
        modules: dict[str, nn.Module],
        execution_plan: list[ModelExecutionNode],
        output_ref: Any,
        input_names: list[str],
        node_metadata: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__()
        self.model_nodes = nn.ModuleDict(modules)
        self._execution_plan = tuple(execution_plan)
        self._output_ref = output_ref
        self._input_names = tuple(input_names)
        self.node_metadata = node_metadata

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Any:
        if len(args) > len(self._input_names):
            raise TypeError(
                f"Model expects at most {len(self._input_names)} positional inputs; "
                f"got {len(args)}"
            )

        values: dict[str, Any] = {}
        for name, value in zip(self._input_names, args):
            values[name] = value

        for name, value in kwargs.items():
            if name not in self._input_names:
                raise TypeError(f"Unexpected model input '{name}'")
            if name in values:
                raise TypeError(f"Model input '{name}' supplied more than once")
            values[name] = value

        missing = [name for name in self._input_names if name not in values]
        if missing:
            raise TypeError("Missing model input(s): " + ", ".join(missing))

        for node in self._execution_plan:
            resolved = _resolve_runtime_refs(node.inputs, values)
            module = self.model_nodes[node.node_id]

            if BUILDER_FIELDS_KEY in resolved:
                # Builder-DAG reserved contract:
                # inputs.fields is an ordered list expanded positionally.
                if set(resolved) != {BUILDER_FIELDS_KEY}:
                    raise RuntimeError(
                        f"Compiled builder node '{node.node_id}' mixes reserved "
                        "input 'fields' with named inputs"
                    )
                fields = resolved[BUILDER_FIELDS_KEY]
                if not isinstance(fields, list) or not fields:
                    raise RuntimeError(
                        f"Compiled builder node '{node.node_id}' reserved input "
                        "'fields' must be a non-empty ordered list"
                    )
                node_output = module(*fields)
            else:
                # Named builder inputs are semantic keyword bindings.
                node_output = module(**resolved)

            values[node.node_id] = node_output

        return _resolve_runtime_refs(self._output_ref, values)


def _resolve_runtime_refs(spec: Any, values: dict[str, Any]) -> Any:
    if isinstance(spec, ModelInputRef):
        return values[spec.name]
    if isinstance(spec, ModelNodeRef):
        return values[spec.node_id]
    if isinstance(spec, list):
        return [_resolve_runtime_refs(item, values) for item in spec]
    if isinstance(spec, dict):
        return {key: _resolve_runtime_refs(value, values) for key, value in spec.items()}
    return spec


# ---------------------------------------------------------------------------
# Model-builder DAG operations.
# Importing this module registers all model operations, exactly like
# dag_feature_ops.py populates FEATURE_DAG_REGISTRY.
# ---------------------------------------------------------------------------


@MODEL_BUILDER_REGISTRY.register("add_numbers")
def add_numbers(inputs, params, runtime):
    """Build-time binary numeric addition for derived model parameters."""
    del runtime
    if inputs:
        raise ValueError("add accepts no DAG data inputs; use params/cfg_params/derived_params")
    if set(params) != {"left", "right"}:
        raise ValueError("add requires exactly two parameters: 'left' and 'right'")

    left = params["left"]
    right = params["right"]
    for name, value in (("left", left), ("right", right)):
        if isinstance(value, bool) or not isinstance(value, Number):
            raise TypeError(f"add parameter '{name}' must be numeric")
    return left + right


@MODEL_BUILDER_REGISTRY.register("build_temporal_embedding")
def build_temporal_embedding(inputs, params, runtime):
    module = ContinuousTimeEmbedding(
        out_dim=int(params["out_dim"]),
        source_index=int(params.get("source_index", 0)),
    )
    return build_node(
        op_name="build_temporal_embedding",
        module=module,
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_linear_transform")
def build_linear_transform(inputs, params, runtime):
    module = nn.Linear(
        int(params["input_dim"]),
        int(params["output_dim"]),
        bias=bool(params.get("bias", True)),
    )
    return build_node(
        op_name="build_linear_transform",
        module=module,
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_activation")
def build_activation(inputs, params, runtime):
    name = str(params.get("activation_name", "relu")).lower()
    builders = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    if name not in builders:
        raise ValueError(f"Unsupported activation_name '{name}'")
    return build_node(
        op_name="build_activation",
        module=builders[name](),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_dropout")
def build_dropout(inputs, params, runtime):
    return build_node(
        op_name="build_dropout",
        module=nn.Dropout(float(params["dropout_rate"])),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_gru")
def build_gru(inputs, params, runtime):
    module = GRUTransform(
        input_dim=int(params["input_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        batch_first=bool(params.get("batch_first", True)),
        num_layers=int(params.get("num_layers", 1)),
        bidirectional=bool(params.get("bidirectional", False)),
    )
    return build_node(
        op_name="build_gru",
        module=module,
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_layer_norm")
def build_layer_norm(inputs, params, runtime):
    return build_node(
        op_name="build_layer_norm",
        module=nn.LayerNorm(int(params["normalized_shape"])),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_concat")
def build_concat(inputs, params, runtime):
    return build_node(
        op_name="build_concat",
        module=TensorConcat(dim=int(params.get("dim", -1))),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_residual")
def build_residual(inputs, params, runtime):
    return build_node(
        op_name="build_residual",
        module=ResidualAdd(),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_root_model")
def build_root_model(inputs, params, runtime):
    """Compile the symbolic graph into the final executable nn.Module.

    root_model is a normal registered DAG node. It adds no model computation.
    Its flat input mapping names the exposed model output(s), so it intentionally
    uses named inputs rather than the Builder-DAG reserved ``fields`` form.
    """
    if params:
        raise DagConfigError(
            "root_model does not accept params; its contract is derived from DAG inputs"
        )
    if BUILDER_FIELDS_KEY in inputs:
        raise DagConfigError(
            "root_model cannot use reserved input 'fields'; root input keys define "
            "the exposed model output names"
        )

    compiled = compile_model_graph(inputs)
    return DagTorchModel(
        modules=compiled.modules,
        execution_plan=compiled.execution_plan,
        output_ref=compiled.output_ref,
        input_names=compiled.input_names,
        node_metadata=compiled.node_metadata,
    )


def extract_last_embedding(
    encoder_output: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard post-encoder helper for callers that need one vector/sequence."""
    valid_len = torch.clamp(mask.sum(dim=1).long(), min=1)
    batch_indices = torch.arange(encoder_output.size(0), device=encoder_output.device)
    last = encoder_output[batch_indices, valid_len - 1]
    return torch.nn.functional.normalize(last, p=2, dim=-1)
