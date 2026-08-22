from typing import Any, Dict, List, Optional, Tuple
from model_diagnostic.cfg_base import HBSCAN4Config

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