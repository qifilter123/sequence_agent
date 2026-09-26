import numpy as np
import random
import torch
from typing import Any

from client.data_collector import load_input_data
from torch.utils.data import DataLoader, TensorDataset
from transformer.feature_util import *

from transformer.model_cfg import *

def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def resolve_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def _build_data_loaders(
    data: TensorDataset,
    is_training: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    common: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }

    return DataLoader(
            data,
            shuffle=is_training,
            generator=generator,
            **common,
        )


def load_batches(sym, start, end, lookback_bars, is_training: bool) -> DataLoader:
    input_data = load_input_data(sym, start, end, lookback_bars)
    deltas, positions, targets = prepare_model_arrays(input_data)
    data = _gen_datasets(deltas, positions, targets)
    print(f"samples={len(data)} ")
    return _build_data_loaders(data, is_training)


def _gen_datasets(
    deltas: np.ndarray,
    positions: np.ndarray,
    targets: np.ndarray,
) -> tuple[TensorDataset, TensorDataset]:

    def dataset(start: int) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(deltas[start:]),
            torch.from_numpy(positions[start:]),
            torch.from_numpy(targets[start:]),
        )

    return dataset(0)
