import numpy as np
from transformer.model_cfg import *


def prepare_model_arrays(
    input_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned model inputs and labels from ``collect_np`` output."""
    required = {"features", "sequence_position", "vwap"}
    missing = required.difference(input_data)
    if missing:
        raise KeyError(f"input_data is missing: {sorted(missing)}")

    features = np.asarray(input_data["features"])
    positions = np.asarray(input_data["sequence_position"])
    if features.ndim != 3 or positions.shape != features.shape[:2]:
        raise ValueError("features and sequence_position have incompatible shapes")
    if len(features) != len(input_data["vwap"]):
        raise ValueError("features and vwap must share the sample dimension")

    deltas = _build_step_deltas(features)
    delta_positions = positions[:, 1:].astype(np.int64, copy=False)
    targets = _build_prediction_targets(input_data["vwap"])
    return deltas, delta_positions, targets


def _build_step_deltas(
    features: np.ndarray,
    price_index: int = PRICE_FEATURE_INDEX,
    volume_index: int = VOLUME_FEATURE_INDEX,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Return centered price/volume ratios for adjacent input bars.

    K raw bars produce K-1 transition tokens. The first bar is retained as the
    baseline for the first transition rather than inventing a previous value.
    """
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("features must have shape [samples, bars>=2, dimensions]")
    if (
        min(price_index, volume_index) < 0
        or max(price_index, volume_index) >= values.shape[2]
    ):
        raise ValueError("price or volume feature index is out of range")

    selected = values[..., [price_index, volume_index]].astype(np.float64, copy=False)
    if not np.isfinite(selected).all() or np.any(selected < 0):
        raise ValueError("price and volume values must be finite and non-negative")

    previous = selected[:, :-1]
    current = selected[:, 1:]
    denominator = previous + current
    if np.any(denominator <= epsilon):
        raise ValueError("adjacent price and volume pairs cannot both be zero")

    deltas = current / denominator - 0.5
    return deltas.astype(np.float32)


def _build_prediction_targets(
    vwap: np.ndarray,
    up_threshold: float = UP_RETURN_THRESHOLD,
    down_threshold: float = DOWN_RETURN_THRESHOLD,
) -> np.ndarray:
    """Map current/next VWAP pairs to UP, DOWN, or FLAT class indices."""
    prices = np.asarray(vwap, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] != 2:
        raise ValueError("vwap must have shape [samples, 2] as [current, next]")
    if not down_threshold < up_threshold:
        raise ValueError("down_threshold must be smaller than up_threshold")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("VWAP label values must be finite and positive")

    returns = prices[:, 1] / prices[:, 0] - 1.0
    targets = np.full(len(returns), FLAT_CLASS, dtype=np.int64)
    targets[returns >= up_threshold] = UP_CLASS
    targets[returns <= down_threshold] = DOWN_CLASS
    return targets