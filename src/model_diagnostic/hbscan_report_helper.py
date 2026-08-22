from typing import Any, Dict, List, Optional, Tuple
from model_diagnostic.cfg_base import HBSCAN4Config
import numpy as np

def report_alignment(alignment: Dict[str, Any], hcfg: HBSCAN4Config) -> None:

    print("\n--- Alignment scoring (record → transaction MAX) ---")

    pure_count = alignment.get("pure_cluster_count", 0)
    fraud_centroid_count = alignment.get("fraud_centroid_count", 0)

    print(
        f"purity gate >= {hcfg.min_centroid_purity}: "
        f"kept {pure_count} clusters | FRAUD centroids: {fraud_centroid_count}"
    )

    if not alignment.get("available", False):
        print(f"no FRAUD centroids — cannot score: {alignment.get('reason', '')}")
        return

    print("\n[Centroid alignment coverage — analysis only]")
    print(
        f"  Transactions within FRAUD centroid radius: "
        f"{alignment['transactions_aligned']}/{alignment['transactions_total']}"
    )
    print(
        f"    fraud aligned: {alignment['fraud_aligned']}  |  "
        f"benign aligned: {alignment['benign_aligned']}"
    )

    print("\n--- Per-scenario alignment rate (* = fraud) ---")
    print(f"  {'scenario':24s} {'n':>4s} {'align%':>7s}   role")

    for row in alignment.get("per_scenario", []):
        tag = "*" if row["is_fraud"] else " "
        print(
            f" {tag}{row['scenario']:23s} "
            f"{row['n']:4d} "
            f"{row['alignment_rate']:7.1%}   "
            f"{row['role']}"
        )

def report_centroid_spread(spread_info: Dict[str, Any]) -> None:
    metrics = spread_info["metrics"]

    print("\n--- Intra-transaction centroid spread ---")

    print("\n[Centroid-spread score — transaction level]")
    print(
        f"  ROC-AUC {metrics['roc_auc']:.4f} | "
        f"PR-AUC {metrics['pr_auc']:.4f} | "
        f"best-F1 {metrics['best_f1']:.4f} "
        f"@ spread >= {metrics['threshold']:.4f}"
    )
    print(
        f"  precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f} | "
        f"accuracy {metrics['accuracy']:.4f}"
    )

    confusion = metrics["confusion"]

    print("\n--- Confusion @ best-F1 (centroid spread) ---")
    print("                 pred_fraud   pred_benign")
    print(
        f"  actual_fraud   {confusion['tp']:10d}   "
        f"{confusion['fn']:11d}"
    )
    print(
        f"  actual_benign  {confusion['fp']:10d}   "
        f"{confusion['tn']:11d}"
    )
    print(
        f"  fraud detection rate {metrics['fraud_detection_rate']:.1%}  |  "
        f"benign FP rate {metrics['benign_fp_rate']:.1%}"
    )

    print("\n--- Per-pattern centroid-spread (transaction level; * = fraud) ---")
    print(
        f"  {'scenario':24s} {'n':>4s} "
        f"{'mean_cs':>7s} {'flag%':>6s} {'supp%':>6s}   role"
    )

    for row in spread_info.get("per_scenario", []):
        tag = "*" if row["is_fraud"] else " "
        print(
            f" {tag}{row['scenario']:23s} "
            f"{row['n']:4d} "
            f"{row['mean_centroid_spread']:7.4f} "
            f"{row['flag_rate']:6.1%} "
            f"{row.get('suppressed_rate', 0.0):6.1%}   "
            f"{row['role']}"
        )

def report_clusters(clusters: List[Dict[str, Any]], noise_info: Dict[str, Any]) -> None:
    print("\n--- Cluster diagnostics ---")
    print(f"{'lbl':>4s} {'size':>5s} {'p_fraud':>7s} {'assign':>6s} "
          f"{'semantic_type':>18s} {'maxAln':>6s} {'medAln':>6s} {'anchor':>10s}  dominant scenarios")
    for c in sorted(clusters, key=lambda x: (-x["p_fraud"], -x["size"])):
        dom = ", ".join(f"{s}:{n}" for s, n in c["dominant_scenarios"])
        st = c.get("semantic_type", "NORMAL_BASELINE")
        anc = c.get("anchor_dominant", "?")
        print(f"{c['label']:4d} {c['size']:5d} {c['p_fraud']:7.3f} {c['assigned']:>6s} "
              f"{st:>18s} {c['max_align_distance']:6.3f} {c['median_align_distance']:6.3f} {anc:>10s}  {dom}")
    if clusters:
        purities = [c["anchor_purity"] for c in clusters]
        n_pure = sum(1 for p in purities if p >= 0.9)
        print(f"\n  [anchor-purity diag] {n_pure}/{len(clusters)} clusters are >=90% single-anchor "
              f"| mean anchor purity {np.mean(purities):.1%} "
              f"| min {min(purities):.1%} max {max(purities):.1%}")
        for c in sorted(clusters, key=lambda x: -x["anchor_purity"]):
            mix = ", ".join(f"{a}:{n}" for a, n in c["anchor_mix"])
            print(f"    c{c['label']:02d} anchor_purity={c['anchor_purity']:.0%}  mix=[{mix}]")
    top = ", ".join(f"{s}:{n}" for s, n in noise_info["top_scenarios"])
    print(f"noise: {noise_info['count']} pts  "
          f"(anomaly {noise_info['n_anomaly']})  top: {top}")

    benign = [
        c for c in clusters
        if c.get("semantic_type", "").startswith("NORMAL_")
        and c["semantic_type"] != "NORMAL_BASELINE"
    ]
    if benign:
        print(f"\n  [LLM hint] Auto-identified benign clusters ({len(benign)}):")
        for c in sorted(benign, key=lambda x: x["semantic_type"]):
            top_sc = c["dominant_scenarios"][0] if c["dominant_scenarios"] else ("?", 0)
            print(f"    label={c['label']:2d}  {c['semantic_type']:<20s}  "
                  f"size={c['size']}  purity={top_sc[1] / max(1, c['size']):.0%}  "
                  f"top={top_sc[0]}:{top_sc[1]}")