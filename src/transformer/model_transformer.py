"""Train a small non-causal Transformer on price/volume step changes."""

from __future__ import annotations

import math


from datetime import date

from torch import Tensor, nn
from torch.nn import functional as F

from transformer.batch_util import *
from transformer.eval_util import *
from transformer.model_util import *


def _build_sinusoidal_embedding(
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


def _build_input_layer(input_dim: int, d_model: int) -> nn.Module:
    """Project each price/volume transition into the model dimension."""
    return nn.Sequential(
        nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
    )


def _build_transformer_encoder(
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


def _build_prediction_head(d_model: int, num_classes: int, dropout: float) -> nn.Module:
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
        self.input_layer = _build_input_layer(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = _build_transformer_encoder(
            d_model, num_heads, num_layers, ffn_dim, dropout
        )
        self.prediction_head = _build_prediction_head(
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
        positions = _build_sinusoidal_embedding(sequence_position, self.d_model)
        encoded = self.encoder(self.input_dropout(tokens + positions))
        return self.prediction_head(encoded[:, -1])


def _prediction_loss(logits: Tensor, targets: Tensor) -> Tensor:

    """Return multiclass cross-entropy over UP, DOWN, and FLAT logits."""
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"logits must have shape [batch, {len(CLASS_NAMES)}]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch]")
    return F.cross_entropy(logits, targets)


def _run_epoch(
    model: StockTransformer,
    batches: DataLoader,
    device: torch.device,
    total_prediction_metrics: dict[str, int],
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch and accumulate predicted class counts in the given dict."""
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0

    for class_name in CLASS_NAMES:
        total_prediction_metrics.setdefault(class_name, 0)
        total_prediction_metrics.setdefault(f"{class_name}_TARGET", 0)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for deltas, positions, targets in batches:
            deltas[..., 0] *= 100
            deltas = deltas.to(device)
            positions = positions.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(deltas, positions)

            batch_prediction_metrics = prediction_metrics_from_logits(logits, targets, training)

            for class_name, count in batch_prediction_metrics.items():
                total_prediction_metrics[class_name] += count

            loss = _prediction_loss(logits, targets)
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


def _run_model(
    model,
    batches: DataLoader,
    device: torch.device,
    is_training: bool
) -> StockTransformer:

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    epochs = EPOCHS if is_training else 1
    for epoch in range(0, epochs):

        prediction_metrics = {class_name: 0 for class_name in CLASS_NAMES}
        train_loss, train_accuracy = _run_epoch(
            model,
            batches,
            device,
            prediction_metrics,
            optimizer if is_training else None
        )

        if is_training:
            print(f"epoch={epoch:02d} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f} ")
        print(f"  train_predictions={prediction_metrics} ")
    return model


def run_model(device, model, start, end, symbol, is_training: bool):

    batches = load_batches(symbol, start, end, LOOKBACK_BARS, is_training)
    model = _run_model(model, batches, device, is_training)

    return model


SYMBOL = "AMD"

def train(device) -> None:

    model = StockTransformer().to(device)
    # Train
    model = run_model(device, model, date(2020, 1, 1), date(2024, 9, 15), SYMBOL, True)
    save_model(model)
    print(f"saved_model={MODEL_OUTPUT_PATH}")


def evaluate(device) -> None:

    model = load_model(MODEL_OUTPUT_PATH, device)
    run_model(device, model, date(2024, 11, 1), date(2025, 2, 1), SYMBOL, False)


if __name__ == "__main__":

    set_random_seed()
    device = resolve_device()
    train(device)
    evaluate(device)





