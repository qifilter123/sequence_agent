from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from model_diagnostic import generic_seq_generator as seq_gen
from model_diagnostic.batch_util import _META_KEYS
from model_diagnostic.dag.hdbscan_registry import HDBSCAN_DAG_REGISTRY
from model_diagnostic.model_trainer import build_full_features


_BENIGN_SEMANTIC_MAP = {
    "Traveler": "NORMAL_TRAVELER",
    "Shopper": "NORMAL_SHOPPER",
    "Upgrader": "NORMAL_UPGRADER",
}
_BENIGN_PURITY_GATE = 0.80
_ANCHOR_NAME = {0: "bca", 1: "em", 2: "fp", 3: "sa"}


def _require_runtime(runtime: dict[str, Any], name: str) -> Any:
    value = runtime.get(name)
    if value is None:
        raise ValueError(f"HDBSCAN DAG operation requires runtime['{name}']")
    return value


def _require_positive_int(params: dict[str, Any], name: str) -> int:
    value = int(params[name])
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _infer_semantic_type(
    p_fraud: float,
    scen_counter: Counter,
    size: int,
    fraud_threshold: float,
) -> str:
    if p_fraud >= fraud_threshold:
        return "FRAUD"
    if not scen_counter:
        return "NORMAL_BASELINE"
    top_sc, top_n = scen_counter.most_common(1)[0]
    if top_n / max(1, size) >= _BENIGN_PURITY_GATE and top_sc in _BENIGN_SEMANTIC_MAP:
        return _BENIGN_SEMANTIC_MAP[top_sc]
    return "NORMAL_BASELINE"


def _best_f1(y_true: np.ndarray, y_scores: np.ndarray):
    roc = (
        float(roc_auc_score(y_true, y_scores))
        if y_true.sum() and (y_true == 0).sum()
        else float("nan")
    )
    pr = float(average_precision_score(y_true, y_scores))
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    best_th = float(thr[best]) if best < len(thr) else float(thr[-1] if len(thr) else 0.0)
    y_pred = (y_scores >= best_th).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    return roc, pr, float(f1[best]), best_th, float(prec[best]), float(rec[best]), acc, y_pred


@HDBSCAN_DAG_REGISTRY.register("generate_anchored_dataset")
def generate_anchored_dataset(inputs, params, runtime):
    """Generate anchored records with one shared positioning contract for build/eval.

    HDBSCAN intentionally does not pass an ``is_training`` flag.  Sequence
    positioning is owned by the sequence generator/modeling pipeline and is the
    same for centroid construction and evaluation.
    """
    del inputs
    model_cfg = _require_runtime(runtime, "model_cfg")
    num_trx = _require_positive_int(params, "num_trx")
    data = seq_gen.make_anchored_dataset(current_cfg=model_cfg, num_trx=num_trx)
    if not isinstance(data, dict):
        raise TypeError("make_anchored_dataset must return a dictionary")
    if "dt" not in data or len(data["dt"]) == 0:
        raise RuntimeError("no records generated")
    return data


@HDBSCAN_DAG_REGISTRY.register("extract_record_embeddings")
@torch.no_grad()
def extract_record_embeddings(inputs, params, runtime):
    """Build one normalized embedding per anchored record."""
    data = inputs["dataset"]
    model = _require_runtime(runtime, "model")
    model_cfg = _require_runtime(runtime, "model_cfg")

    pooling = str(params.get("pooling", "last_seq_embedding"))
    normalization = str(params.get("normalization", "l2"))
    if pooling != "last_seq_embedding":
        raise ValueError(
            f"Unsupported pooling '{pooling}'; current implementation supports only last_seq_embedding"
        )
    if normalization != "l2":
        raise ValueError(
            f"Unsupported normalization '{normalization}'; current implementation supports only l2"
        )

    batch_size = _require_positive_int(params, "batch_size")
    n = len(data["dt"])
    embs: list[torch.Tensor] = []

    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        batch = {
            key: value[sl]
            for key, value in data.items()
            if key not in _META_KEYS
        }
        # Embedding extraction is always inference-style feature/model execution.
        x_full, raw = build_full_features(batch, model_cfg, is_training=False)
        encoder_output = model(x_full)

        mask = raw["mask"]
        valid_len = torch.clamp(mask.sum(dim=1).long(), min=1)
        batch_indices = torch.arange(encoder_output.size(0), device=encoder_output.device)
        emb = encoder_output[batch_indices, valid_len - 1]
        emb = F.normalize(emb, p=2, dim=-1)
        embs.append(emb.cpu())

    X_all = torch.cat(embs, dim=0).numpy().astype(np.float32)
    return {
        "X_all": X_all,
        "trx_id_all": _as_numpy(data["trx_id"], dtype=np.int64).copy(),
        "record_scenario_all": list(data["record_scenario"]),
        "record_is_fraud_all": _as_numpy(data["record_is_fraud"], dtype=np.int32),
        "valid_len_all": _as_numpy(data["valid_len"], dtype=np.int64),
        "anchor_type_all": _as_numpy(data["anchor_type"], dtype=np.int64),
        "num_trx": int(data["num_trx"]),
        "trx_scenario": list(data["trx_scenario"]),
        "seq_dt": _as_numpy(data["dt"]),
        "seq_amount": _as_numpy(data["amount"]),
        "seq_sw_ip": _as_numpy(data["sw_ip"]),
        "seq_sw_email": _as_numpy(data["sw_email"]),
        "seq_sw_fp": _as_numpy(data["sw_fp"]),
        "seq_sw_bill": _as_numpy(data["sw_bill"]),
        "seq_sw_ship": _as_numpy(data["sw_ship"]),
        "rec_anchor_type": _as_numpy(data["anchor_type"], dtype=np.int64),
        "rec_valid_len": _as_numpy(data["valid_len"], dtype=np.int64),
        "rec_side": _as_numpy(data["record_pair_side"], dtype=np.int64),
        "rec_scenario_all": list(data["record_scenario"]),
        "rec_is_fraud_all": _as_numpy(data["record_is_fraud"], dtype=np.int32),
    }


@HDBSCAN_DAG_REGISTRY.register("apply_record_length_gate")
def apply_record_length_gate(inputs, params, runtime):
    """Create the HDBSCAN-fit record view while preserving ungated X_all."""
    del runtime
    embed = inputs["embeddings"]
    min_len_gate = int(params["min_len_gate"])
    if min_len_gate < 0:
        raise ValueError("min_len_gate must be >= 0")

    valid_len_all = embed["valid_len_all"]
    keep = valid_len_all > min_len_gate
    scenarios_all = embed["record_scenario_all"]

    result = dict(embed)
    result.update(
        {
            "X": embed["X_all"][keep],
            "record_scenario": [s for s, selected in zip(scenarios_all, keep) if selected],
            "is_anom": embed["record_is_fraud_all"][keep],
            "trx_id": embed["trx_id_all"][keep],
            "valid_len": valid_len_all[keep],
            "anchor_type": embed["anchor_type_all"][keep],
            "dropped_by_len_gate": int((~keep).sum()),
        }
    )

    current_scenarios = result["trx_scenario"]
    result["cid_is_anom"] = np.array(
        [1 if seq_gen._is_anomaly(sc) else 0 for sc in current_scenarios],
        dtype=np.int32,
    )
    return result


@HDBSCAN_DAG_REGISTRY.register("hdbscan_fit")
def hdbscan_fit(inputs, params, runtime):
    """Fit HDBSCAN using record geometry only.  No fraud labels are read here."""
    del runtime
    X = inputs["records"]["X"]
    if len(X) == 0:
        raise RuntimeError("no records remain after min_len_gate")

    clusterer = HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
        cluster_selection_method=str(params["cluster_selection_method"]),
        metric=str(params["metric"]),
        copy=False,
    )
    labels = clusterer.fit_predict(X).astype(np.int64)
    return {
        "labels": labels,
        "cluster_count": int(sum(1 for label in set(labels.tolist()) if label != -1)),
        "noise_count": int((labels == -1).sum()),
    }


@HDBSCAN_DAG_REGISTRY.register("build_cluster_geometry")
def build_cluster_geometry(inputs, params, runtime):
    """Build centroids/radii from HDBSCAN assignments without fraud-label semantics."""
    del runtime
    records = inputs["records"]
    labels = inputs["clustering"]["labels"]
    X = records["X"]
    scenarios = records["record_scenario"]
    anchor_type = records["anchor_type"]
    pct = float(params["align_distance_percentile"])
    if not 0.0 <= pct <= 100.0:
        raise ValueError("align_distance_percentile must be in [0, 100]")

    geometry: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        centroid = X[idx].mean(axis=0).astype(np.float32)
        distances = np.linalg.norm(X[idx] - centroid, axis=1)
        scen_counter = Counter(scenarios[i] for i in idx)
        anchor_counter = Counter(_ANCHOR_NAME[int(anchor_type[i])] for i in idx)
        dominant_anchor, dominant_count = anchor_counter.most_common(1)[0]
        anchor_purity = dominant_count / len(idx)
        geometry.append(
            {
                "label": int(label),
                "size": int(len(idx)),
                "centroid": centroid,
                "max_align_distance": float(np.percentile(distances, pct)),
                "median_align_distance": float(np.percentile(distances, 50.0)),
                "anchor_purity": float(anchor_purity),
                "anchor_dominant": f"{dominant_anchor}:{anchor_purity:.0%}",
                "anchor_mix": anchor_counter.most_common(4),
                "dominant_scenarios": scen_counter.most_common(3),
                "_member_indices": idx,
            }
        )
    return geometry


@HDBSCAN_DAG_REGISTRY.register("annotate_clusters")
def annotate_clusters(inputs, params, runtime):
    """Add label-aware FRAUD/NORMAL annotations to already-formed clusters."""
    del runtime
    geometry = inputs["geometry"]
    records = inputs["records"]
    is_anom = records["is_anom"]
    fraud_threshold = float(params["fraud_purity_threshold"])
    if not 0.0 <= fraud_threshold <= 1.0:
        raise ValueError("fraud_purity_threshold must be in [0, 1]")

    annotated: list[dict[str, Any]] = []
    for item in geometry:
        idx = item["_member_indices"]
        n_anom = int(is_anom[idx].sum())
        p_fraud = n_anom / len(idx)
        scen_counter = Counter(dict(item["dominant_scenarios"]))
        cluster = {k: v for k, v in item.items() if k != "_member_indices"}
        cluster.update(
            {
                "n_anomaly": n_anom,
                "n_normal": int(len(idx) - n_anom),
                "p_fraud": float(p_fraud),
                "purity": float(max(p_fraud, 1.0 - p_fraud)),
                "assigned": "FRAUD" if p_fraud >= fraud_threshold else "NORMAL",
                "semantic_type": _infer_semantic_type(
                    p_fraud,
                    scen_counter,
                    len(idx),
                    fraud_threshold,
                ),
            }
        )
        annotated.append(cluster)
    return annotated


@HDBSCAN_DAG_REGISTRY.register("build_noise_diagnostics")
def build_noise_diagnostics(inputs, params, runtime):
    """Summarize HDBSCAN noise; fraud counts are evaluation diagnostics only."""
    del params, runtime
    records = inputs["records"]
    labels = inputs["clustering"]["labels"]
    noise_idx = np.where(labels == -1)[0]
    noise_counter = Counter(records["record_scenario"][i] for i in noise_idx)
    return {
        "count": int(len(noise_idx)),
        "n_anomaly": int(records["is_anom"][noise_idx].sum()),
        "top_scenarios": noise_counter.most_common(6),
    }


@HDBSCAN_DAG_REGISTRY.register("select_fraud_centroids")
def select_fraud_centroids(inputs, params, runtime):
    del runtime
    clusters = inputs["clusters"]
    min_purity = float(params["min_centroid_purity"])
    if not 0.0 <= min_purity <= 1.0:
        raise ValueError("min_centroid_purity must be in [0, 1]")
    pure = [c for c in clusters if c["purity"] >= min_purity]
    fraud = [c for c in pure if c["assigned"] == "FRAUD"]
    return {
        "pure_clusters": pure,
        "fraud_clusters": fraud,
        "pure_cluster_count": len(pure),
        "fraud_centroid_count": len(fraud),
    }


@HDBSCAN_DAG_REGISTRY.register("fraud_centroid_alignment")
def fraud_centroid_alignment(inputs, params, runtime):
    del runtime
    records = inputs["records"]
    selected = inputs["fraud_centroids"]
    aggregation = str(params.get("transaction_aggregation", "any"))
    if aggregation != "any":
        raise ValueError("current fraud_centroid_alignment supports only transaction_aggregation='any'")

    fraud_clusters = selected["fraud_clusters"]
    if not fraud_clusters:
        return {
            "available": False,
            "reason": "No FRAUD centroids passed the purity gate.",
            "pure_cluster_count": selected["pure_cluster_count"],
            "fraud_centroid_count": 0,
            "transactions_aligned": 0,
            "fraud_aligned": 0,
            "benign_aligned": 0,
            "per_scenario": [],
        }

    X = records["X"]
    centroids = np.stack([c["centroid"] for c in fraud_clusters], axis=0)
    radii = np.array([c["max_align_distance"] for c in fraud_clusters], dtype=np.float32)
    distances = np.stack([np.linalg.norm(X - centroid, axis=1) for centroid in centroids], axis=1)
    nearest = np.argmin(distances, axis=1)
    d_min = distances[np.arange(len(X)), nearest]
    aligned = d_min <= radii[nearest]

    num_trx = records["num_trx"]
    txn_aligned = np.zeros(num_trx, dtype=bool)
    for index, trx_id in enumerate(records["trx_id"]):
        if aligned[index]:
            txn_aligned[int(trx_id)] = True

    y = records["cid_is_anom"]
    scenarios = np.asarray(records["trx_scenario"])
    per_scenario = []
    for scenario in sorted(set(scenarios.tolist())):
        mask = scenarios == scenario
        if not mask.any():
            continue
        per_scenario.append(
            {
                "scenario": str(scenario),
                "n": int(mask.sum()),
                "is_fraud": bool(seq_gen._is_anomaly(scenario)),
                "role": "detect" if seq_gen._is_anomaly(scenario) else "FP-rate",
                "alignment_rate": float(txn_aligned[mask].mean()),
            }
        )

    return {
        "available": True,
        "pure_cluster_count": selected["pure_cluster_count"],
        "fraud_centroid_count": selected["fraud_centroid_count"],
        "transactions_total": int(num_trx),
        "transactions_aligned": int(txn_aligned.sum()),
        "fraud_aligned": int((txn_aligned & (y == 1)).sum()),
        "benign_aligned": int((txn_aligned & (y == 0)).sum()),
        "per_scenario": per_scenario,
    }


@HDBSCAN_DAG_REGISTRY.register("nearest_centroid_assignment")
def nearest_centroid_assignment(inputs, params, runtime):
    """Assign every ungated record (X_all) to its nearest built centroid."""
    del params, runtime
    records = inputs["records"]
    clusters = inputs["clusters"]
    if not clusters:
        raise RuntimeError("HDBSCAN produced no clusters; centroid assignment is unavailable")

    X_all = records["X_all"]
    all_centroids = np.stack([c["centroid"] for c in clusters], axis=0)
    distances = np.stack(
        [np.linalg.norm(X_all - centroid, axis=1) for centroid in all_centroids],
        axis=1,
    )
    nearest = np.argmin(distances, axis=1)
    return {
        "nearest": nearest,
        "all_centroids": all_centroids,
        "cluster_types": [c.get("semantic_type", "") for c in clusters],
    }


@HDBSCAN_DAG_REGISTRY.register("transaction_centroid_spread")
def transaction_centroid_spread(inputs, params, runtime):
    del runtime
    records = inputs["records"]
    assignment = inputs["assignment"]
    aggregation = str(params.get("aggregation", "max_pairwise_distance"))
    if aggregation != "max_pairwise_distance":
        raise ValueError("current centroid spread supports only aggregation='max_pairwise_distance'")

    nearest = assignment["nearest"]
    all_centroids = assignment["all_centroids"]
    trx_ids = records["trx_id_all"]
    num_trx = records["num_trx"]

    txn_to_centroids: dict[int, set[int]] = defaultdict(set)
    for index, trx_id in enumerate(trx_ids):
        txn_to_centroids[int(trx_id)].add(int(nearest[index]))

    spread = np.zeros(num_trx, dtype=np.float32)
    for trx_id, centroid_ids in txn_to_centroids.items():
        if len(centroid_ids) < 2:
            continue
        idx = list(centroid_ids)
        centroids = all_centroids[idx]
        rows, cols = np.triu_indices(len(idx), k=1)
        spread[trx_id] = float(np.linalg.norm(centroids[rows] - centroids[cols], axis=1).max())

    # Kept for compatibility with existing diagnostics; suppression is disabled.
    return {
        "spread": spread,
        "txn_suppressed": np.zeros(num_trx, dtype=bool),
    }


@HDBSCAN_DAG_REGISTRY.register("binary_score_metrics")
def binary_score_metrics(inputs, params, runtime):
    del runtime
    records = inputs["records"]
    spread_result = inputs["spread_result"]
    threshold_strategy = str(params.get("threshold_strategy", "best_f1"))
    if threshold_strategy != "best_f1":
        raise ValueError("current evaluation supports only threshold_strategy='best_f1'")

    y = records["cid_is_anom"]
    scores = spread_result["spread"]
    suppressed = spread_result["txn_suppressed"]
    roc, pr, f1, threshold, precision, recall, accuracy, y_pred = _best_f1(y, scores)

    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fraud_detection_rate = tp / max(1, tp + fn)
    benign_fp_rate = fp / max(1, fp + tn)

    scenarios = np.asarray(records["trx_scenario"])
    per_scenario = []
    order = sorted(
        set(scenarios.tolist()),
        key=lambda name: -float(scores[scenarios == name].mean()) if (scenarios == name).any() else 0.0,
    )
    for scenario in order:
        mask = scenarios == scenario
        if not mask.any():
            continue
        is_fraud = bool(seq_gen._is_anomaly(scenario))
        per_scenario.append(
            {
                "scenario": str(scenario),
                "n": int(mask.sum()),
                "is_fraud": is_fraud,
                "role": "detect" if is_fraud else "FP-rate",
                "mean_centroid_spread": float(scores[mask].mean()),
                "flag_rate": float(y_pred[mask].mean()),
                "suppressed_rate": float(suppressed[mask].mean()),
            }
        )

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "best_f1": f1,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fraud_detection_rate": fraud_detection_rate,
        "benign_fp_rate": benign_fp_rate,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "transactions_total": int(records["num_trx"]),
        "fraud_transactions": int(y.sum()),
        "benign_transactions": int((y == 0).sum()),
    }
    return {
        "y_pred": y_pred,
        "threshold": threshold,
        "metrics": metrics,
        "per_scenario": per_scenario,
    }
