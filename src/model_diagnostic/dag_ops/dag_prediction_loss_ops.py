from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_diagnostic.dag.model_builder_registry import (
    MODEL_BUILDER_REGISTRY,
    build_node,
)


class SequenceSlice(nn.Module):
    """Slice one tensor dimension using normal Python slice semantics."""

    def __init__(
        self,
        *,
        dim: int = 1,
        start: int | None = None,
        end: int | None = None,
        step: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.start = start if start is None else int(start)
        self.end = end if end is None else int(end)
        self.step = step if step is None else int(step)
        if self.step == 0:
            raise ValueError("sequence_slice step cannot be zero")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if dim < 0 or dim >= x.ndim:
            raise ValueError(
                f"sequence_slice dim {self.dim} is invalid for tensor rank {x.ndim}"
            )
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(self.start, self.end, self.step)
        return x[tuple(slices)]


class TensorSqueeze(nn.Module):
    def __init__(self, *, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


class DictFieldSlice(nn.Module):
    """Select one tensor field from a mapping, then slice one dimension."""

    def __init__(
        self,
        *,
        field: str,
        dim: int = 1,
        start: int | None = None,
        end: int | None = None,
        step: int | None = None,
    ) -> None:
        super().__init__()
        if not field:
            raise ValueError("dict_field_slice requires a non-empty field")
        self.field = str(field)
        self.slice = SequenceSlice(dim=dim, start=start, end=end, step=step)

    def forward(self, source: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.field not in source:
            raise KeyError(f"Missing source field '{self.field}'")
        return self.slice(source[self.field])


class NextStepPairMask(nn.Module):
    """Build the next-step validity mask: mask[:, :-1] * mask[:, 1:]."""

    def __init__(self, *, field: str = "mask") -> None:
        super().__init__()
        self.field = str(field)

    def forward(self, source: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.field not in source:
            raise KeyError(f"Missing source field '{self.field}'")
        mask = source[self.field]
        if mask.ndim < 2:
            raise ValueError("next_step_pair_mask expects mask rank >= 2")
        return mask[:, :-1] * mask[:, 1:]


class BinaryFieldsToClass(nn.Module):
    """Pack binary tensor fields into one integer class id."""

    def __init__(
        self,
        *,
        fields: list[str],
        weights: list[int],
        dim: int = 1,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        super().__init__()
        if not fields or len(fields) != len(weights):
            raise ValueError("binary_fields_to_class requires equal non-empty fields/weights")
        self.fields = tuple(str(field) for field in fields)
        self.weights = tuple(int(weight) for weight in weights)
        self.dim = int(dim)
        self.start = start if start is None else int(start)
        self.end = end if end is None else int(end)

    def forward(self, source: Mapping[str, torch.Tensor]) -> torch.Tensor:
        packed = None
        slicer = SequenceSlice(dim=self.dim, start=self.start, end=self.end)
        for field, weight in zip(self.fields, self.weights):
            if field not in source:
                raise KeyError(f"Missing source field '{field}'")
            value = slicer(source[field]) * weight
            packed = value if packed is None else packed + value
        return packed.long()


class StackDictFields(nn.Module):
    """Slice selected tensor fields from a mapping and stack them."""

    def __init__(
        self,
        *,
        fields: list[str],
        slice_dim: int = 1,
        start: int | None = None,
        end: int | None = None,
        stack_dim: int = -1,
    ) -> None:
        super().__init__()
        if not fields:
            raise ValueError("stack_dict_fields requires at least one field")
        self.fields = tuple(str(field) for field in fields)
        self.slice_dim = int(slice_dim)
        self.start = start if start is None else int(start)
        self.end = end if end is None else int(end)
        self.stack_dim = int(stack_dim)

    def forward(self, source: Mapping[str, torch.Tensor]) -> torch.Tensor:
        slicer = SequenceSlice(
            dim=self.slice_dim,
            start=self.start,
            end=self.end,
        )
        values = []
        for field in self.fields:
            if field not in source:
                raise KeyError(f"Missing source field '{field}'")
            values.append(slicer(source[field]))
        return torch.stack(values, dim=self.stack_dim)


def _valid_tokens(mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    return mask.sum() + float(epsilon)


class MaskedSmoothL1Mean(nn.Module):
    def __init__(self, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(prediction, target, reduction="none") * mask
        return loss.sum() / _valid_tokens(mask, self.epsilon)


class MaskedCrossEntropyMean(nn.Module):
    """Masked CE for sequence logits shaped [B, L, C]."""

    def __init__(self, *, class_dim: int = -1, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.class_dim = int(class_dim)
        self.epsilon = float(epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ce_input = logits.movedim(self.class_dim, 1)
        raw = F.cross_entropy(ce_input, target, reduction="none")
        loss = raw * mask
        return loss.sum() / _valid_tokens(mask, self.epsilon)


class MaskedMultilabelBCEMean(nn.Module):
    """Masked BCE-with-logits with label dimension summed before token mean."""

    def __init__(self, *, label_dim: int = -1, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.label_dim = int(label_dim)
        self.epsilon = float(epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = raw.sum(dim=self.label_dim) * mask
        return loss.sum() / _valid_tokens(mask, self.epsilon)


class WeightedSum(nn.Module):
    def __init__(self, *, weights: dict[str, float]) -> None:
        super().__init__()
        if not weights:
            raise ValueError("weighted_sum requires at least one weight")
        self.weights = {str(name): float(value) for name, value in weights.items()}

    def forward(self, **values: torch.Tensor) -> torch.Tensor:
        if set(values) != set(self.weights):
            raise ValueError(
                "weighted_sum inputs must exactly match configured weights: "
                f"inputs={sorted(values)}, weights={sorted(self.weights)}"
            )
        total = None
        for name, weight in self.weights.items():
            term = values[name] * weight
            total = term if total is None else total + term
        return total


class Scale(nn.Module):
    def __init__(self, *, factor: float) -> None:
        super().__init__()
        self.factor = float(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


@MODEL_BUILDER_REGISTRY.register("build_sequence_slice")
def build_sequence_slice(inputs, params, runtime):
    return build_node(
        op_name="build_sequence_slice",
        module=SequenceSlice(
            dim=int(params.get("dim", 1)),
            start=params.get("start"),
            end=params.get("end"),
            step=params.get("step"),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_squeeze")
def build_squeeze(inputs, params, runtime):
    return build_node(
        op_name="build_squeeze",
        module=TensorSqueeze(dim=int(params["dim"])),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_dict_field_slice")
def build_dict_field_slice(inputs, params, runtime):
    return build_node(
        op_name="build_dict_field_slice",
        module=DictFieldSlice(
            field=str(params["field"]),
            dim=int(params.get("dim", 1)),
            start=params.get("start"),
            end=params.get("end"),
            step=params.get("step"),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_next_step_pair_mask")
def build_next_step_pair_mask(inputs, params, runtime):
    return build_node(
        op_name="build_next_step_pair_mask",
        module=NextStepPairMask(field=str(params.get("field", "mask"))),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_binary_fields_to_class")
def build_binary_fields_to_class(inputs, params, runtime):
    return build_node(
        op_name="build_binary_fields_to_class",
        module=BinaryFieldsToClass(
            fields=list(params["fields"]),
            weights=list(params["weights"]),
            dim=int(params.get("dim", 1)),
            start=params.get("start"),
            end=params.get("end"),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_stack_dict_fields")
def build_stack_dict_fields(inputs, params, runtime):
    return build_node(
        op_name="build_stack_dict_fields",
        module=StackDictFields(
            fields=list(params["fields"]),
            slice_dim=int(params.get("slice_dim", 1)),
            start=params.get("start"),
            end=params.get("end"),
            stack_dim=int(params.get("stack_dim", -1)),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_masked_smooth_l1_mean")
def build_masked_smooth_l1_mean(inputs, params, runtime):
    return build_node(
        op_name="build_masked_smooth_l1_mean",
        module=MaskedSmoothL1Mean(epsilon=float(params.get("epsilon", 1e-6))),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_masked_cross_entropy_mean")
def build_masked_cross_entropy_mean(inputs, params, runtime):
    return build_node(
        op_name="build_masked_cross_entropy_mean",
        module=MaskedCrossEntropyMean(
            class_dim=int(params.get("class_dim", -1)),
            epsilon=float(params.get("epsilon", 1e-6)),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_masked_multilabel_bce_mean")
def build_masked_multilabel_bce_mean(inputs, params, runtime):
    return build_node(
        op_name="build_masked_multilabel_bce_mean",
        module=MaskedMultilabelBCEMean(
            label_dim=int(params.get("label_dim", -1)),
            epsilon=float(params.get("epsilon", 1e-6)),
        ),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_weighted_sum")
def build_weighted_sum(inputs, params, runtime):
    input_names = set(inputs)
    expected_params = {f"{name}_weight" for name in input_names}
    if set(params) != expected_params:
        raise ValueError(
            "weighted_sum requires one '<input>_weight' parameter per input: "
            f"expected={sorted(expected_params)}, got={sorted(params)}"
        )
    weights = {name: params[f"{name}_weight"] for name in inputs}
    return build_node(
        op_name="build_weighted_sum",
        module=WeightedSum(weights=weights),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )


@MODEL_BUILDER_REGISTRY.register("build_scale")
def build_scale(inputs, params, runtime):
    if set(params) != {"factor"}:
        raise ValueError("scale requires exactly one parameter: 'factor'")
    return build_node(
        op_name="build_scale",
        module=Scale(factor=float(params["factor"])),
        inputs=inputs,
        params=params,
        runtime=runtime,
    )
