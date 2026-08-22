from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "pytorch_static_model_structure"

# Curated, static, JSON-friendly attributes that are useful to an AI agent.
# This deliberately avoids dumping module.__dict__.
_STATIC_ATTRIBUTE_NAMES = (
    # Common custom/model dimensions
    "out_dim",
    "hidden_dim",
    "base_dim",
    "time_emb_dim",
    "combo_dim",
    "sw_classes",
    "is_new_classes",
    "use_residual",
    # Linear / recurrent
    "in_features",
    "out_features",
    "input_size",
    "hidden_size",
    "num_layers",
    "batch_first",
    "bidirectional",
    "dropout",
    "p",
    "bias",
    # Normalization / activation
    "normalized_shape",
    "eps",
    "elementwise_affine",
    "inplace",
    # Embedding
    "num_embeddings",
    "embedding_dim",
    "padding_idx",
    # Convolution
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
)


def _default_diagnostics() -> Dict[str, bool]:
    """Create a fresh diagnostics config for one module."""
    return {
        "enabled": False,
        "activation": False,
        "gradient": False,
        "parameter": False,
        "update": False,
        "backward_flow": False,
    }


def _class_path(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _parent_path(module_path: str) -> Optional[str]:
    if module_path == "":
        return None
    if "." not in module_path:
        return ""
    return module_path.rsplit(".", 1)[0]


def _local_name(module_path: str) -> str:
    return "<root>" if module_path == "" else module_path.rsplit(".", 1)[-1]


def _depth(module_path: str) -> int:
    return 0 if module_path == "" else module_path.count(".") + 1


def _json_safe(value: Any) -> Optional[Any]:
    """
    Convert a curated static attribute to a compact JSON-safe value.
    Returns None for unsupported/complex runtime objects.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, tuple):
        converted = [_json_safe(v) for v in value]
        return converted if all(v is not None for v in converted) else None

    if isinstance(value, list):
        converted = [_json_safe(v) for v in value]
        return converted if all(v is not None for v in converted) else None

    # For modules such as nn.Linear, bias is a Parameter or None.
    if isinstance(value, nn.Parameter):
        return True

    return None


def _extract_static_attributes(module: nn.Module) -> Dict[str, Any]:
    """Extract only a small, useful set of declared static attributes."""
    attributes: Dict[str, Any] = {}

    for name in _STATIC_ATTRIBUTE_NAMES:
        if not hasattr(module, name):
            continue

        raw = getattr(module, name)

        # Special-case bias so Linear/Conv are represented as presence/absence,
        # while RNN modules keep their native boolean bias flag.
        if name == "bias":
            if isinstance(raw, nn.Parameter):
                attributes[name] = True
                continue
            if raw is None:
                attributes[name] = False
                continue

        value = _json_safe(raw)
        if value is not None:
            attributes[name] = value

    return attributes


def _ancestor_paths(module_path: str):
    """Yield a module path and all of its ancestors through the root path ""."""
    current: Optional[str] = module_path
    while current is not None:
        yield current
        current = _parent_path(current)


def _collect_parameter_and_buffer_stats(model: nn.Module):
    """
    Collect direct/subtree parameter and buffer element counts in one pass.

    Parameter/buffer identities are de-duplicated inside each subtree, so shared
    tensors are not double-counted for a given module path. remove_duplicate=False
    preserves alias paths when shared modules are registered in multiple places.
    """
    direct_param_ids: Dict[str, set[int]] = defaultdict(set)
    subtree_param_ids: Dict[str, set[int]] = defaultdict(set)
    param_meta: Dict[int, tuple[int, bool]] = {}

    for full_name, param in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        owner = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        pid = id(param)
        param_meta[pid] = (param.numel(), param.requires_grad)
        direct_param_ids[owner].add(pid)
        for path in _ancestor_paths(owner):
            subtree_param_ids[path].add(pid)

    direct_buffer_ids: Dict[str, set[int]] = defaultdict(set)
    subtree_buffer_ids: Dict[str, set[int]] = defaultdict(set)
    buffer_meta: Dict[int, int] = {}

    for full_name, buffer in model.named_buffers(
        recurse=True,
        remove_duplicate=False,
    ):
        owner = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        bid = id(buffer)
        buffer_meta[bid] = buffer.numel()
        direct_buffer_ids[owner].add(bid)
        for path in _ancestor_paths(owner):
            subtree_buffer_ids[path].add(bid)

    def param_counts(ids: set[int]) -> Dict[str, int]:
        return {
            "elements": sum(param_meta[pid][0] for pid in ids),
            "trainable_elements": sum(
                param_meta[pid][0] for pid in ids if param_meta[pid][1]
            ),
        }

    def buffer_count(ids: set[int]) -> int:
        return sum(buffer_meta[bid] for bid in ids)

    return {
        "direct_param_ids": direct_param_ids,
        "subtree_param_ids": subtree_param_ids,
        "param_counts": param_counts,
        "direct_buffer_ids": direct_buffer_ids,
        "subtree_buffer_ids": subtree_buffer_ids,
        "buffer_count": buffer_count,
    }


def _build_structure_id(model_record: Dict[str, Any], modules: List[Dict[str, Any]]) -> str:
    """
    Build a deterministic structural fingerprint.

    Diagnostics flags are intentionally excluded so Step 2 can edit them
    without changing the identity of the underlying static model structure.
    """
    payload = {
        "model_class_path": model_record["class_path"],
        "modules": [
            {
                "module_path": m["module_path"],
                "class_path": m["class_path"],
                "parent_path": m["parent_path"],
                "children": m["children"],
                "attributes": m["attributes"],
                "parameters": m["parameters"],
                "buffers": m["buffers"],
                "shared": m["shared"],
                "aliases": m["aliases"],
            }
            for m in modules
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_json_atomic(
    payload: Dict[str, Any],
    output_path: Union[str, Path],
    *,
    indent: int = 2,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.write("\n")

    tmp_path.replace(output_path)


def render_model_structure(
    model: nn.Module,
    *,
    model_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Render a static JSON-serializable description of a PyTorch nn.Module tree.

    Scope of Step 1:
      - module hierarchy
      - stable module paths
      - module class/type information
      - selected declared static attributes
      - direct/subtree parameter and buffer counts
      - shared-module aliases
      - per-module diagnostics flags, all defaulting to False

    Explicitly NOT included:
      - forward/execution path
      - activations or gradients
      - actual runtime input/output shapes
      - hooks
      - diagnostic policy decisions

    The same returned structure can be:
      1) written as JSON and given to an AI agent,
      2) manually edited in Step 2 to enable diagnostics,
      3) reused later by DiagnosticProbeManager for module registration.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    # remove_duplicate=False preserves all registered paths when the same
    # module instance is referenced from more than one place.
    named_modules = list(model.named_modules(remove_duplicate=False))

    paths_by_instance: Dict[int, List[str]] = defaultdict(list)
    for module_path, module in named_modules:
        paths_by_instance[id(module)].append(module_path)

    discovered_paths = {path for path, _ in named_modules}
    children_by_parent: Dict[str, List[str]] = defaultdict(list)

    for module_path, _ in named_modules:
        if module_path == "":
            continue

        parent = _parent_path(module_path)
        if parent not in discovered_paths:
            raise RuntimeError(
                f"Invalid module hierarchy: parent '{parent}' for "
                f"'{module_path}' was not discovered"
            )
        children_by_parent[parent].append(module_path)

    tensor_stats = _collect_parameter_and_buffer_stats(model)

    module_records: List[Dict[str, Any]] = []

    for module_path, module in named_modules:
        aliases = paths_by_instance[id(module)]
        canonical_path = aliases[0]

        children = children_by_parent.get(module_path, [])

        record = {
            "module_path": module_path,
            "name": _local_name(module_path),
            "class_name": module.__class__.__name__,
            "class_path": _class_path(module),
            "parent_path": _parent_path(module_path),
            "depth": _depth(module_path),
            "children": list(children),
            "is_leaf": len(children) == 0,
            "attributes": _extract_static_attributes(module),
            "parameters": {
                "direct": tensor_stats["param_counts"](
                    tensor_stats["direct_param_ids"][module_path]
                ),
                "subtree": tensor_stats["param_counts"](
                    tensor_stats["subtree_param_ids"][module_path]
                ),
            },
            "buffers": {
                "direct_elements": tensor_stats["buffer_count"](
                    tensor_stats["direct_buffer_ids"][module_path]
                ),
                "subtree_elements": tensor_stats["buffer_count"](
                    tensor_stats["subtree_buffer_ids"][module_path]
                ),
            },
            "shared": len(aliases) > 1,
            "canonical_path": canonical_path,
            "aliases": [p for p in aliases if p != module_path],
            "diagnostics": _default_diagnostics(),
        }

        module_records.append(record)

    root_param_counts = tensor_stats["param_counts"](
        tensor_stats["subtree_param_ids"][""]
    )
    root_buffer_elements = tensor_stats["buffer_count"](
        tensor_stats["subtree_buffer_ids"][""]
    )

    model_record = {
        "name": model_name or model.__class__.__name__,
        "class_name": model.__class__.__name__,
        "class_path": _class_path(model),
        "root_module_path": "",
        "module_count": len(module_records),
        "total_parameters": root_param_counts["elements"],
        "trainable_parameters": root_param_counts["trainable_elements"],
        "total_buffer_elements": root_buffer_elements,
    }

    structure: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "framework": {
            "name": "pytorch",
            "version": torch.__version__,
        },
        "model": model_record,
        "modules": module_records,
    }

    structure["structure_id"] = _build_structure_id(
        model_record,
        module_records,
    )

    if output_path is not None:
        _write_json_atomic(structure, output_path)

    return structure


class StaticModelStructureMixin:
    """
    Optional mixin/wrapper for DiagnosticProbeManager.

    A ProbeManager can inherit this mixin or copy the method directly.
    It assumes `self.model` is a torch.nn.Module.
    """

    model: nn.Module
    model_structure: Optional[Dict[str, Any]] = None

    def render_model_structure(
        self,
        *,
        model_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        self.model_structure = render_model_structure(
            self.model,
            model_name=model_name,
            output_path=output_path,
        )
        return self.model_structure
