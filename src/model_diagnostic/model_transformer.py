"""Train a small non-causal Transformer on price/volume step changes."""

from __future__ import annotations

import copy
import math
import random
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from client.data_collector import FEATURE_NAMES, collect_np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "AMD"
START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 9, 15)
LOOKBACK_BARS = 26

UP_RETURN_THRESHOLD = 0.02
DOWN_RETURN_THRESHOLD = -0.02
CLASS_NAMES = ("UP", "DOWN", "FLAT")
UP_CLASS = 0
DOWN_CLASS = 1
FLAT_CLASS = 2

PRICE_FEATURE_INDEX = 0
VOLUME_FEATURE_INDEX = 1
INPUT_DIM = 2
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 64
DROPOUT = 0.1
POSITIONAL_BASE = 10_000.0

VALIDATION_FRACTION = 0.2
VALIDATION_GAP_SAMPLES = LOOKBACK_BARS
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
SHUFFLE_TRAINING = True
NUM_WORKERS = 0
RANDOM_SEED = 7
DEVICE = "auto"
EPSILON = 1e-12
MODEL_OUTPUT_PATH = Path(__file__).with_name("model_transformer.pt")


def build_step_deltas(
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


def build_sinusoidal_embedding(
    sequence_position: Tensor,
    d_model: int = D_MODEL,
    base: float = POSITIONAL_BASE,
) -> Tensor:
    """Build standard sinusoidal embeddings for logical sequence positions."""
    if sequence_position.ndim != 2:
        raise ValueError("sequence_position must have shape [batch, steps]")
    if d_model < 1 or base <= 1.0:
        raise ValueError("d_model must be positive and base must be greater than one")

    positions = sequence_position.to(dtype=torch.float32).unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=positions.dtype)
        * (-math.log(base) / d_model)
    )
    angles = positions * frequencies
    embedding = torch.zeros(
        (*sequence_position.shape, d_model),
        device=sequence_position.device,
        dtype=positions.dtype,
    )
    embedding[..., 0::2] = torch.sin(angles)
    embedding[..., 1::2] = torch.cos(angles[..., : d_model // 2])
    return embedding


def build_prediction_targets(
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


def build_input_layer(input_dim: int, d_model: int) -> nn.Module:
    """Project each price/volume transition into the model dimension."""
    return nn.Sequential(
        nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
    )


def build_transformer_encoder(
    d_model: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
) -> nn.Module:
    """Build a bidirectional encoder; no causal mask is used."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=num_layers,
        norm=nn.LayerNorm(d_model),
        enable_nested_tensor=False,
    )


def build_prediction_head(d_model: int, num_classes: int, dropout: float) -> nn.Module:
    """Convert the final contextual token into three class logits."""
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, num_classes),
    )


class StockTransformer(nn.Module):
    """Many-to-one Transformer for a fixed historical window."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_layer = build_input_layer(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = build_transformer_encoder(
            d_model, num_heads, num_layers, ffn_dim, dropout
        )
        self.prediction_head = build_prediction_head(
            d_model, len(CLASS_NAMES), dropout
        )

    def forward(self, step_deltas: Tensor, sequence_position: Tensor) -> Tensor:
        if step_deltas.ndim != 3 or step_deltas.shape[-1] != self.input_dim:
            raise ValueError(
                f"step_deltas must have shape [batch, steps, {self.input_dim}]"
            )
        if step_deltas.shape[:2] != sequence_position.shape:
            raise ValueError("step_deltas and sequence_position must be aligned")

        tokens = self.input_layer(step_deltas)
        positions = build_sinusoidal_embedding(sequence_position, self.d_model)
        encoded = self.encoder(self.input_dropout(tokens + positions))
        return self.prediction_head(encoded[:, -1])


def prediction_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return multiclass cross-entropy over UP, DOWN, and FLAT logits."""
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"logits must have shape [batch, {len(CLASS_NAMES)}]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch]")
    return F.cross_entropy(logits, targets)


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

    deltas = build_step_deltas(features)
    delta_positions = positions[:, 1:].astype(np.int64, copy=False)
    targets = build_prediction_targets(input_data["vwap"])
    return deltas, delta_positions, targets


def split_datasets(
    deltas: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
) -> tuple[TensorDataset, TensorDataset]:
    """Create chronological train/validation datasets with a purged boundary."""
    count = len(targets)
    if len(deltas) != count or len(positions) != count:
        raise ValueError("all model arrays must share the sample dimension")
    split = int(count * (1.0 - VALIDATION_FRACTION))
    train_end = split - VALIDATION_GAP_SAMPLES
    if train_end < 1 or split >= count:
        raise ValueError(
            "not enough samples for the configured validation split and gap"
        )

    def dataset(start: int, end: int) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(deltas[start:end]),
            torch.from_numpy(positions[start:end]),
            torch.from_numpy(targets[start:end]),
        )

    return dataset(0, train_end), dataset(split, count)


def build_data_loaders(
    training_data: TensorDataset,
    validation_data: TensorDataset,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    common: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }
    return (
        DataLoader(
            training_data,
            shuffle=SHUFFLE_TRAINING,
            generator=generator,
            **common,
        ),
        DataLoader(validation_data, shuffle=False, **common),
    )


def resolve_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def run_epoch(
    model: StockTransformer,
    batches: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch and return loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for deltas, positions, targets in batches:
            deltas = deltas.to(device)
            positions = positions.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(deltas, positions)
            loss = prediction_loss(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    return total_loss / total_samples, total_correct / total_samples


def train_model(
    training_batches: DataLoader,
    validation_batches: DataLoader,
    device: torch.device,
) -> StockTransformer:
    model = StockTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, training_batches, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_batches, device
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    return model


def save_model(model: StockTransformer) -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "lookback_bars": LOOKBACK_BARS,
            "model_config": {
                "input_dim": INPUT_DIM,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "ffn_dim": FFN_DIM,
                "dropout": DROPOUT,
            },
            "label_config": {
                "up_return_threshold": UP_RETURN_THRESHOLD,
                "down_return_threshold": DOWN_RETURN_THRESHOLD,
            },
        },
        MODEL_OUTPUT_PATH,
    )


def load_input_data() -> dict[str, np.ndarray]:

    if tuple(FEATURE_NAMES[:2]) != ("vwap_all", "volume_all"):
        raise ValueError("expected vwap_all and volume_all as the first two features")
    return collect_np(SYMBOL, START_DATE, END_DATE, LOOKBACK_BARS)


def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def main() -> None:
    set_random_seed()
    input_data = load_input_data()
    if len(input_data["features"]) == 0:
        raise ValueError("data_collector returned no training samples")

    deltas, positions, targets = prepare_model_arrays(input_data)
    training_data, validation_data = split_datasets(deltas, positions, targets)
    training_batches, validation_batches = build_data_loaders(
        training_data, validation_data
    )
    device = resolve_device()
    print(
        f"device={device} train_samples={len(training_data)} "
        f"validation_samples={len(validation_data)}"
    )
    model = train_model(training_batches, validation_batches, device)
    save_model(model)
    print(f"saved_model={MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
