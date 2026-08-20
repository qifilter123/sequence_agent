from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any
import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

_META_KEYS = {"num_trx", "trx_scenario", "record_is_fraud"}

def sample_txn_batch(data, txn_index, n_txn):

    cids = random.sample(list(txn_index.keys()), min(n_txn, len(txn_index)))
    rows, group = [], []
    for g, c in enumerate(cids):
        for r in txn_index[c]:
            rows.append(r)
            group.append(g)
    idx = torch.tensor(rows, dtype=torch.long)
    batch = {}
    for k, v in data.items():
        if k in _META_KEYS:
            continue
        batch[k] = [v[i] for i in rows] if k == "record_scenario" else v[idx]
    return batch, torch.tensor(group, dtype=torch.long), len(cids)