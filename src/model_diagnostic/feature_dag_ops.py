from __future__ import annotations

from typing import Any

import torch

from model_diagnostic import generic_feature_util as feature_util
from model_diagnostic.dag.operation_registry import OperationRegistry


FEATURE_DAG_REGISTRY = OperationRegistry()


def _require_cfg(runtime: dict[str, Any]):
    cfg = runtime.get("cfg")
    if cfg is None:
        raise ValueError("Feature DAG operation requires runtime['cfg']")
    return cfg


@FEATURE_DAG_REGISTRY.register("extract_sequence_fields")
def extract_sequence_fields(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Move selected sequence fields to the configured device and reshape them.

    The field selection/order comes directly from the node's top-level YAML
    ``inputs`` list. DagProcessor converts that list into an insertion-ordered
    ``{field_name: tensor}`` mapping before invoking this operation.

    ``cfg_params: {device: device}`` is resolved by DagProcessor, so this op
    receives the actual device value as ``params['device']`` and does not need
    to know how runtime cfg lookup works.
    """
    if not inputs:
        raise ValueError("extract_sequence_fields requires at least one input field")

    if "device" not in params:
        raise ValueError(
            "extract_sequence_fields requires cfg_params binding for 'device'"
        )
    device = params["device"]

    shape_tensor = next(iter(inputs.values()))
    if not isinstance(shape_tensor, torch.Tensor):
        raise TypeError("extract_sequence_fields inputs must be torch.Tensor values")
    if shape_tensor.ndim < 2:
        raise ValueError(
            "First configured sequence field must have at least 2 dimensions; "
            f"got shape={tuple(shape_tensor.shape)}"
        )
    batch_size, seq_len = shape_tensor.shape[:2]

    raw: dict[str, torch.Tensor] = {}
    for field_name, source in inputs.items():
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

    return {"raw": raw}


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
    return feature_util.dt_to_pre(inputs["dt"])


@FEATURE_DAG_REGISTRY.register("amt_norm")
def amt_norm(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    cfg = _require_cfg(runtime)
    mean_key = params.get("mean_config_key", "amt_mean")
    deviation_key = params.get("deviation_config_key", "amt_dv")
    clip_key = params.get("clip_config_key", "amt_clip_val")

    for key in (mean_key, deviation_key, clip_key):
        if not hasattr(cfg, key):
            raise AttributeError(f"Config does not define required attribute '{key}'")

    if (mean_key, deviation_key, clip_key) == ("amt_mean", "amt_dv", "amt_clip_val"):
        return feature_util.amt_norm(inputs["amount"], cfg)

    proxy = type("AmtNormConfig", (), {
        "amt_mean": getattr(cfg, mean_key),
        "amt_dv": getattr(cfg, deviation_key),
        "amt_clip_val": getattr(cfg, clip_key),
    })()
    return feature_util.amt_norm(inputs["amount"], proxy)


@FEATURE_DAG_REGISTRY.register("amt_to_pre")
def amt_to_pre(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    return feature_util.amt_to_pre(inputs["amount"])


@FEATURE_DAG_REGISTRY.register("switch_decay")
def switch_decay(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    cfg = _require_cfg(runtime)
    tau_key = params.get("tau_config_key", "tau_sw")
    if not hasattr(cfg, tau_key):
        raise AttributeError(f"Config does not define required attribute '{tau_key}'")

    if tau_key == "tau_sw":
        return feature_util.get_decay_factor(cfg, inputs["dt"])

    proxy = type("SwitchDecayConfig", (), {"tau_sw": getattr(cfg, tau_key)})()
    return feature_util.get_decay_factor(proxy, inputs["dt"])


@FEATURE_DAG_REGISTRY.register("stable_log")
def stable_log(
    inputs: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
) -> torch.Tensor:
    return feature_util.compute_stable_log(
        inputs["value"],
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
