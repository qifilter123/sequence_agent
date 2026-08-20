from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


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

    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not isinstance(tensors, list) or not tensors:
            raise ValueError("concat requires a non-empty tensor list")
        return torch.cat(tensors, dim=self.dim)


class ResidualAdd(nn.Module):
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if x.shape != residual.shape:
            raise ValueError(
                "Residual tensors must have identical shapes: "
                f"x={tuple(x.shape)}, residual={tuple(residual.shape)}"
            )
        return x + residual


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


class DagTorchModel(nn.Module):
    """Executable PyTorch model assembled by the model-builder DAG.

    DagProcessor is used only at build time. Once constructed, normal PyTorch
    training/inference calls ``forward`` directly and executes this compiled
    model plan without re-reading YAML or dispatching builder operations.
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

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
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
            if len(resolved) == 1:
                node_output = module(next(iter(resolved.values())))
            else:
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


class NextStepPredictionHelper(nn.Module):
    """Temporary prediction layer until prediction + loss move to their DAG."""

    def __init__(
        self,
        hidden_dim: int,
        sw_classes: int,
        is_new_classes: int,
    ) -> None:
        super().__init__()
        self.v_head = nn.Linear(hidden_dim, 1)
        self.sw_head = nn.Linear(hidden_dim, sw_classes)
        self.amt_head = nn.Linear(hidden_dim, 1)
        self.is_new_head = nn.Linear(hidden_dim, is_new_classes)

    def forward(self, encoder_output: torch.Tensor) -> dict[str, torch.Tensor]:
        h_head = encoder_output[:, :-1, :]
        return {
            "v_pred": self.v_head(h_head).squeeze(-1),
            "sw_logits": self.sw_head(h_head),
            "amt_pred": self.amt_head(h_head).squeeze(-1),
            "is_new_logits": self.is_new_head(h_head),
        }


def extract_last_embedding(
    encoder_output: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard post-encoder helper for callers that need one vector/sequence."""
    valid_len = torch.clamp(mask.sum(dim=1).long(), min=1)
    batch_indices = torch.arange(encoder_output.size(0), device=encoder_output.device)
    last = encoder_output[batch_indices, valid_len - 1]
    return torch.nn.functional.normalize(last, p=2, dim=-1)
