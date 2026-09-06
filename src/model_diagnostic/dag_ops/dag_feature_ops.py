from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from model_diagnostic import generic_feature_util as feature_util
from model_diagnostic.dag.operation_registry import OperationRegistry


FEATURE_DAG_REGISTRY = OperationRegistry()


@FEATURE_DAG_REGISTRY.register("extract_sequence_fields")
def extract_sequence_fields(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return extracted fields; the DAG outputs declaration supplies the alias.

    This operation returns the field mapping itself, not a DAG-level
    ``{"raw": fields}`` envelope. ``outputs: {raw: extracted_batch}`` owns that
    public return shape.
    """
    fields = inputs.get("fields")

    if not isinstance(fields, dict) or not fields:
        raise ValueError(
            "extract_sequence_fields requires non-empty 'fields' mapping"
        )

    if "device" not in params:
        raise ValueError(
            "extract_sequence_fields requires cfg_params binding for 'device'"
        )
    device = params["device"]

    shape_tensor = next(iter(fields.values()))
    if not isinstance(shape_tensor, torch.Tensor):
        raise TypeError(
            "extract_sequence_fields fields must contain torch.Tensor values"
        )
    if shape_tensor.ndim < 2:
        raise ValueError(
            "First configured sequence field must have at least 2 dimensions; "
            f"got shape={tuple(shape_tensor.shape)}"
        )
    batch_size, seq_len = shape_tensor.shape[:2]

    raw: dict[str, torch.Tensor] = {}
    for field_name, source in fields.items():
        if not isinstance(source, torch.Tensor):
            raise TypeError(
                f"Configured field '{field_name}' must be a torch.Tensor; "
                f"got {type(source).__name__}"
            )
        if source.numel() != batch_size * seq_len:
            raise ValueError(
                f"Configured field '{field_name}' cannot be reshaped to "
                f"({batch_size}, {seq_len}); shape={tuple(source.shape)}"
            )
        raw[field_name] = source.to(device).view(batch_size, seq_len)

    return raw


@FEATURE_DAG_REGISTRY.register("field_to_history")
def field_to_history(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    inverse = bool(params.get("inverse", False))
    return feature_util.fields_to_history(
        [inputs["field"]],
        [inverse],
        inputs["mask"],
    )[0]


@FEATURE_DAG_REGISTRY.register("dt_to_pre")
def dt_to_pre(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    # ``inputs`` is the transform DAG's original raw input mapping because this
    # node omits an explicit YAML inputs declaration.
    return feature_util.dt_to_pre(inputs["dt"])


@FEATURE_DAG_REGISTRY.register("amt_to_pre")
def amt_to_pre(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    # ``inputs`` is the transform DAG's original raw input mapping because this
    # node omits an explicit YAML inputs declaration.
    return feature_util.amt_to_pre(inputs["amount"])


@FEATURE_DAG_REGISTRY.register("amt_norm")
def amt_norm(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    """Normalize amount using values already resolved from cfg_params."""
    required = ("mean", "deviation", "clip")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(
            "amt_norm requires cfg_params for: " + ", ".join(missing)
        )

    # Keep generic_feature_util as the numerical source of truth while avoiding
    # any runtime-cfg dependency inside the DAG operation itself.
    cfg_view = SimpleNamespace(
        amt_mean=params["mean"],
        amt_dv=params["deviation"],
        amt_clip_val=params["clip"],
    )
    return feature_util.amt_norm(inputs["amount"], cfg_view)


@FEATURE_DAG_REGISTRY.register("switch_decay")
def switch_decay(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    """Compute switch decay using tau already resolved from cfg_params."""
    if "tau" not in params:
        raise ValueError("switch_decay requires cfg_params binding for 'tau'")

    cfg_view = SimpleNamespace(tau_sw=params["tau"])
    return feature_util.get_decay_factor(cfg_view, inputs["dt"])


@FEATURE_DAG_REGISTRY.register("stable_log")
def stable_log(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    return feature_util.compute_stable_log(
        inputs["field"],
        params.get("base", "log1p"),
    )


@FEATURE_DAG_REGISTRY.register("multiply")
def multiply(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    return inputs["left"] * inputs["right"]


@FEATURE_DAG_REGISTRY.register("stack_fields")
def stack_fields(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    fields = inputs["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError("stack_fields requires a non-empty fields input list")

    dim = int(params.get("dim", -1))
    result = torch.stack(fields, dim=dim)

    if params.get("apply_mask", False):
        mask = inputs.get("mask")
        if mask is None:
            raise ValueError("stack_fields apply_mask=true requires a mask input")
        result = result * mask.unsqueeze(dim)

    return result


@FEATURE_DAG_REGISTRY.register("build_dict")
def build_dict(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return dict(inputs)
