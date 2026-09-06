import random
import torch

_META_KEYS = {"num_trx", "trx_scenario", "record_is_fraud"}

def sample_txn_batch(data, txn_index, n_txn):

    cids = random.sample(list(txn_index.keys()), min(n_txn, len(txn_index)))
    rows = []
    for g, c in enumerate(cids):
        for r in txn_index[c]:
            rows.append(r)

    idx = torch.tensor(rows, dtype=torch.long)
    batch = {}
    for k, v in data.items():
        if k in _META_KEYS:
            continue
        batch[k] = [v[i] for i in rows] if k == "record_scenario" else v[idx]
    return batch
