"""DAG-driven label-free HDBSCAN clustering + label-aware evaluation.

The HDBSCAN fit/geometry path is label-free. Fraud labels are introduced only
when formed clusters are annotated/evaluated. The same anchored-sequence
positioning semantics are used for centroid construction and evaluation.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _evaluate_on_clusters(
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

        embed_eval, spread_info = _evaluate_on_clusters(model, cfg, hcfg, clusters)
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
    _evaluate_on_clusters(model, cfg, hcfg, clusters)


if __name__ == "__main__":
    main()
