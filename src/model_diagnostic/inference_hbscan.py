"""DAG-driven label-free HDBSCAN clustering + label-aware evaluation.

The HDBSCAN fit/geometry path is label-free. Fraud labels are introduced only
when formed clusters are annotated/evaluated. The same anchored-sequence
positioning semantics are used for centroid construction and evaluation.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_diagnostic.cfg_base import CFG2
from model_diagnostic.dag.dag_processor import DagProcessor
from model_diagnostic.dag.hdbscan_registry import HDBSCAN_DAG_REGISTRY
from model_diagnostic.model_trainer import load_trained_model, set_seed
from model_diagnostic import hbscan_report_helper
from model_diagnostic.cfg_base import HBSCAN4Config

from model_diagnostic.dag_ops import dag_hdbscan_ops as _dag_hdbscan_ops # Must keep, registering ops into registry.

_HDBSCAN_DAG_RUNTIME = None

def _get_hdbscan_dag_runtime():
    """Lazily load both HDBSCAN runtime DAG configs once per experiment."""
    global _HDBSCAN_DAG_RUNTIME
    if _HDBSCAN_DAG_RUNTIME is None:
        processor = DagProcessor(HDBSCAN_DAG_REGISTRY)
        config_dir = Path(__file__).resolve().parent / "config"
        centroid_cfg = processor.load_config(config_dir / "hdbscan_centroid_build.yaml")
        evaluation_cfg = processor.load_config(config_dir / "hdbscan_evaluation.yaml")
        _HDBSCAN_DAG_RUNTIME = (processor, centroid_cfg, evaluation_cfg)
    return _HDBSCAN_DAG_RUNTIME


def reload_hdbscan_dag_runtime():
    """Reload HDBSCAN YAML after an Agent/experiment edits either DAG."""
    global _HDBSCAN_DAG_RUNTIME
    _HDBSCAN_DAG_RUNTIME = None
    return _get_hdbscan_dag_runtime()


def _dag_runtime(model, model_cfg: CFG2, hcfg: HBSCAN4Config) -> dict[str, Any]:
    # Generic DagProcessor cfg_params resolve from runtime['cfg']; for this DAG,
    # that cfg is the HDBSCAN experiment config. The trained model and its CFG2
    # remain explicit runtime objects with separate names.
    return {
        "cfg": hcfg,
        "model": model,
        "model_cfg": model_cfg,
    }


def _op(name: str, inputs: dict[str, Any], params: dict[str, Any], runtime: dict[str, Any]):
    return HDBSCAN_DAG_REGISTRY.get(name)(inputs, params, runtime)


def collect_record_embeddings(
    model,
    cfg: CFG2,
    hcfg: HBSCAN4Config,
    is_training: Optional[bool] = None,
) -> Dict[str, Any]:
    """Backward-compatible helper for callers that previously passed is_training.

    ``is_training`` is intentionally ignored. HDBSCAN centroid build and
    evaluation use the same sequence-positioning contract. Encoder feature
    extraction itself always runs with ``is_training=False`` inside the DAG op.
    """
    del is_training
    runtime = _dag_runtime(model, cfg, hcfg)
    dataset = _op(
        "generate_anchored_dataset",
        {},
        {"num_trx": hcfg.eval_num_trx},
        runtime,
    )
    embeddings = _op(
        "extract_record_embeddings",
        {"dataset": dataset},
        {
            "pooling": "last_seq_embedding",
            "normalization": "l2",
            "batch_size": hcfg.embed_batch,
        },
        runtime,
    )
    result = _op(
        "apply_record_length_gate",
        {"embeddings": embeddings},
        {"min_len_gate": hcfg.min_len_gate},
        runtime,
    )
    print(f"records embedded          : {len(result['X'])}   "
          f"(dropped {result['dropped_by_len_gate']} with len<= {hcfg.min_len_gate})")
    print(f"  incl. short (centroid spread): {len(result['X_all'])}   "
          f"(all records, no len gate)")
    print(f"  record-level anomaly    : {int(result['is_anom'].sum())} / {len(result['is_anom'])}")
    print(f"transactions (currents)   : {result['num_trx']}   "
          f"anomaly {int(result['cid_is_anom'].sum())} / {result['num_trx']}")
    return result


def run_hdbscan_v4(
    embed: Dict[str, Any],
    hcfg: HBSCAN4Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward-compatible wrapper over the centroid-build DAG operations."""
    runtime = {"cfg": hcfg}
    clustering = _op(
        "hdbscan_fit",
        {"records": embed},
        {
            "min_cluster_size": hcfg.min_cluster_size,
            "min_samples": hcfg.min_samples,
            "cluster_selection_epsilon": hcfg.cluster_selection_epsilon,
            "cluster_selection_method": hcfg.cluster_selection_method,
            "metric": hcfg.metric,
        },
        runtime,
    )
    geometry = _op(
        "build_cluster_geometry",
        {"records": embed, "clustering": clustering},
        {"align_distance_percentile": hcfg.align_distance_percentile},
        runtime,
    )
    clusters = _op(
        "annotate_clusters",
        {"geometry": geometry, "records": embed},
        {"fraud_purity_threshold": hcfg.fraud_purity_threshold},
        runtime,
    )
    noise_info = _op(
        "build_noise_diagnostics",
        {"records": embed, "clustering": clustering},
        {},
        runtime,
    )
    return clusters, noise_info


def evaluate_v4(
    embed: Dict[str, Any],
    clusters: List[Dict[str, Any]],
    hcfg: HBSCAN4Config,
) -> Dict[str, Any]:
    """Backward-compatible fraud-centroid alignment wrapper."""
    runtime = {"cfg": hcfg}
    selected = _op(
        "select_fraud_centroids",
        {"clusters": clusters},
        {"min_centroid_purity": hcfg.min_centroid_purity},
        runtime,
    )
    return _op(
        "fraud_centroid_alignment",
        {"records": embed, "fraud_centroids": selected},
        {"transaction_aggregation": "any"},
        runtime,
    )


def evaluate_intra_centroid_spread(
    embed: Dict[str, Any],
    clusters: List[Dict[str, Any]],
    hcfg: HBSCAN4Config,
) -> Dict[str, Any]:
    """Backward-compatible transaction centroid-spread wrapper."""
    runtime = {"cfg": hcfg}
    assignment = _op(
        "nearest_centroid_assignment",
        {"records": embed, "clusters": clusters},
        {},
        runtime,
    )
    spread_result = _op(
        "transaction_centroid_spread",
        {"records": embed, "assignment": assignment},
        {"aggregation": "max_pairwise_distance"},
        runtime,
    )
    evaluation = _op(
        "binary_score_metrics",
        {"records": embed, "spread_result": spread_result},
        {"threshold_strategy": "best_f1"},
        runtime,
    )
    return {
        "y_pred": evaluation["y_pred"],
        "spread": spread_result["spread"],
        "threshold": evaluation["threshold"],
        "nearest": assignment["nearest"],
        "all_centroids": assignment["all_centroids"],
        "cluster_types": assignment["cluster_types"],
        "metrics": evaluation["metrics"],
        "per_scenario": evaluation["per_scenario"],
    }


_ANCHOR_NAME = {0: "BCA", 1: "EM", 2: "FP", 3: "SA"}
_SIDE_NAME = {0: "main", 1: "contrast"}


def _print_record_steps(embed: Dict[str, Any], i: int, length: int) -> None:
    dt = embed["seq_dt"][i]
    amt = embed["seq_amount"][i]
    sw = {
        "ip": embed["seq_sw_ip"][i],
        "em": embed["seq_sw_email"][i],
        "fp": embed["seq_sw_fp"][i],
        "bill": embed["seq_sw_bill"][i],
        "ship": embed["seq_sw_ship"][i],
    }
    print(f"        {'step':>4s} {'dt(s)':>11s} {'amt':>9s}  switches")
    for step in range(length):
        flags = ",".join(name for name, value in sw.items() if value[step] > 0.5) or "-"
        print(f"        {step:4d} {dt[step]:11.1f} {amt[step]:9.2f}  {flags}")


def show_samples(
    embed: Dict[str, Any],
    num_samples: int,
    pattern_name: str,
    spread_info,
    want_caught: bool = True,
    show_steps: bool = True,
) -> None:
    """Existing debug/report helper retained outside the algorithm DAG."""
    num_trx = embed["num_trx"]
    scenarios = np.array(embed["trx_scenario"])
    fraud_transactions = spread_info["y_pred"]
    nearest = spread_info["nearest"]
    all_centroids = spread_info["all_centroids"]
    cluster_types = spread_info["cluster_types"]
    spread_value = spread_info["spread"]
    threshold = spread_info["threshold"]

    caught = np.zeros(num_trx, dtype=bool)
    ft = np.asarray(list(fraud_transactions)) if not isinstance(fraud_transactions, np.ndarray) else fraud_transactions
    if ft.dtype == bool and ft.shape[0] == num_trx:
        caught = ft
    elif ft.shape[0] == num_trx and ft.min() >= 0 and ft.max() <= 1:
        caught = ft.astype(bool)
    else:
        caught[ft.astype(int)] = True

    candidates = [
        trx_id for trx_id in range(num_trx)
        if (scenarios[trx_id] == pattern_name or scenarios[trx_id].startswith(pattern_name))
        and bool(caught[trx_id]) == want_caught
    ]
    tag = "PREDICTED-FRAUD" if want_caught else "PREDICTED-BENIGN"
    print(f"\n===== showSamples('{pattern_name}', {tag}) "
          f"({len(candidates)} available, showing up to {num_samples}) =====")
    if not candidates:
        print(f"  (no {tag} transactions for this pattern)")
        return

    trx_id_all = embed["trx_id_all"]
    for trx_id in candidates[:num_samples]:
        record_idx = np.where(trx_id_all == trx_id)[0]
        print(f"\n--- trx #{trx_id}  scenario={scenarios[trx_id]}  ({len(record_idx)} anchor-view records)"
              f"  spread={spread_value[trx_id]:.4f} (>= {threshold:.4f} => FRAUD) ---")
        view_label = {}
        for i in sorted(record_idx, key=lambda j: (int(embed["rec_side"][j]), int(embed["rec_anchor_type"][j]))):
            length = int(embed["rec_valid_len"][i])
            anchor = _ANCHOR_NAME.get(int(embed["rec_anchor_type"][i]), "?")
            side = _SIDE_NAME.get(int(embed["rec_side"][i]), "?")
            fraud = "FRAUD" if embed["rec_is_fraud_all"][i] else "benign"
            centroid_idx = int(nearest[i])
            view_label[i] = f"{anchor}/{side}"
            print(f"  [{anchor:3s} / {side:8s}]  len={length:2d}  record={fraud}"
                  f"  -> centroid c{centroid_idx:02d}({cluster_types[centroid_idx] or '?'})")
            if show_steps:
                _print_record_steps(embed, i, length)

        centroid_of = {i: int(nearest[i]) for i in record_idx}
        best = None
        for left in record_idx:
            for right in record_idx:
                if right <= left or centroid_of[left] == centroid_of[right]:
                    continue
                distance = float(
                    np.linalg.norm(all_centroids[centroid_of[left]] - all_centroids[centroid_of[right]])
                )
                if best is None or distance > best[0]:
                    best = (distance, left, right)
        if best is None:
            print("  >> all views share one centroid — spread driven at txn level "
                  "(no divergent pair; see suppression/threshold)")
        else:
            distance, left, right = best
            print(f"  >> FLAG driver: {view_label[left]} (c{centroid_of[left]:02d}) vs "
                  f"{view_label[right]} (c{centroid_of[right]:02d})  centroid-dist={distance:.4f}")


def build_hdbscan_centroids(model, cfg: CFG2, hcfg: HBSCAN4Config):
    """Phase 1: build HDBSCAN centroids with the shared sequence-positioning contract."""
    print("\n" + "=" * 60)
    print(" PHASE 1: BUILDING CENTROIDS")
    print("=" * 60)
    processor, centroid_dag, _ = _get_hdbscan_dag_runtime()
    result = processor.run(
        centroid_dag,
        inputs={},
        runtime=_dag_runtime(model, cfg, hcfg),
    )
    clusters = result["clusters"]
    noise_info = result["noise_info"]
    return clusters, noise_info


def evaluate_on_clusters(
    model,
    cfg: CFG2,
    hcfg: HBSCAN4Config,
    clusters: List[Dict[str, Any]],
):
    """Phase 2: evaluate alignment and centroid spread on already-built clusters."""
    print("\n" + "=" * 60)
    print(" PHASE 2: EVALUATION ON BUILT CLUSTERS")
    print("=" * 60)
    processor, _, evaluation_dag = _get_hdbscan_dag_runtime()
    result = processor.run(
        evaluation_dag,
        inputs={"clusters": clusters},
        runtime=_dag_runtime(model, cfg, hcfg),
    )

    embed_eval = result["record_embeddings"]
    assignment = result["nearest_centroid"]
    spread_result = result["centroid_spread"]
    evaluation = result["evaluation_metrics"]
    spread_info = {
        "y_pred": evaluation["y_pred"],
        "spread": spread_result["spread"],
        "threshold": evaluation["threshold"],
        "nearest": assignment["nearest"],
        "all_centroids": assignment["all_centroids"],
        "cluster_types": assignment["cluster_types"],
        "metrics": evaluation["metrics"],
        "per_scenario": evaluation["per_scenario"],
        "alignment": result["alignment"],
    }

    hbscan_report_helper.report_alignment(result["alignment"], hcfg)
    hbscan_report_helper.report_centroid_spread(spread_info)

    return embed_eval, spread_info


def _serialize_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact cluster summary without transporting centroid vectors."""
    return {
        "label": int(cluster["label"]),
        "size": int(cluster["size"]),
        "n_anomaly": int(cluster["n_anomaly"]),
        "n_normal": int(cluster["n_normal"]),
        "p_fraud": float(cluster["p_fraud"]),
        "purity": float(cluster["purity"]),
        "assigned": str(cluster["assigned"]),
        "semantic_type": str(cluster.get("semantic_type", "")),
        "anchor_purity": float(cluster.get("anchor_purity", 0.0)),
        "anchor_dominant": str(cluster.get("anchor_dominant", "")),
        "anchor_mix": [[str(name), int(count)] for name, count in cluster.get("anchor_mix", [])],
        "dominant_scenarios": [
            [str(name), int(count)] for name, count in cluster.get("dominant_scenarios", [])
        ],
        "max_align_distance": float(cluster.get("max_align_distance", 0.0)),
        "median_align_distance": float(cluster.get("median_align_distance", 0.0)),
    }


def _snapshot_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def run_hdbscan_evaluation(
    model,
    cfg: CFG2,
    hcfg: Optional[HBSCAN4Config] = None,
) -> Dict[str, Any]:
    """Run the two HDBSCAN DAGs safely against an in-memory trained model."""
    if model is None:
        raise ValueError("model is required")
    if cfg is None:
        raise ValueError("cfg is required")

    hcfg = hcfg or HBSCAN4Config()
    rng_state = _snapshot_rng_state()
    was_training = bool(model.training)

    try:
        set_seed(hcfg.seed)
        model.eval()

        clusters, noise_info = build_hdbscan_centroids(model, cfg, hcfg)
        hbscan_report_helper.report_clusters(clusters, noise_info)
        if not clusters:
            raise RuntimeError("HDBSCAN produced no non-noise clusters")

        embed_eval, spread_info = evaluate_on_clusters(model, cfg, hcfg, clusters)
        metrics = spread_info.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("centroid-spread evaluation did not produce metrics")

        return {
            "status": "completed",
            "evaluation": "hdbscan_centroid_spread_v4",
            "config": asdict(hcfg),
            "cluster_build": {
                "cluster_count": len(clusters),
                "fraud_cluster_count": sum(1 for c in clusters if c.get("assigned") == "FRAUD"),
                "normal_cluster_count": sum(1 for c in clusters if c.get("assigned") == "NORMAL"),
                "noise": {
                    "count": int(noise_info.get("count", 0)),
                    "n_anomaly": int(noise_info.get("n_anomaly", 0)),
                    "top_scenarios": [
                        [str(name), int(count)]
                        for name, count in noise_info.get("top_scenarios", [])
                    ],
                },
                "clusters": [_serialize_cluster(c) for c in clusters],
            },
            "alignment": spread_info.get("alignment", {}),
            "centroid_spread": metrics,
            "per_pattern": spread_info.get("per_scenario", []),
            "evaluation_dataset": {
                "transactions": int(embed_eval["num_trx"]),
                "record_count_with_len_gate": int(len(embed_eval["X"])),
                "record_count_all": int(len(embed_eval["X_all"])),
                "fraud_transactions": int(embed_eval["cid_is_anom"].sum()),
                "benign_transactions": int((embed_eval["cid_is_anom"] == 0).sum()),
            },
        }
    finally:
        model.train(was_training)
        _restore_rng_state(rng_state)


def main():
    cfg = CFG2()
    hcfg = HBSCAN4Config()
    set_seed(hcfg.seed)
    print("===== HDBSCAN v4 — label-free clustering + label-aware evaluation =====")
    model = load_trained_model(cfg)

    clusters, noise_info = build_hdbscan_centroids(model, cfg, hcfg)
    hbscan_report_helper.report_clusters(clusters, noise_info)
    evaluate_on_clusters(model, cfg, hcfg, clusters)


if __name__ == "__main__":
    main()
