"""encoder_eval_hbscan_4.py — label-free HDBSCAN on Seq-on-Graph record embeddings.
"""
import sys, os, math, random

sys.path.insert(0, "src")

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from model_diagnostic import generic_seq_generator as seq_gen
from model_diagnostic.encoder_model_train import (
    set_seed, load_trained_model, build_full_features,
)
from model_diagnostic.cfg_base import CFG2
from model_diagnostic.batch_util import _META_KEYS


@dataclass
class HBSCAN4Config:
    eval_num_trx: int = 3000
    embed_batch: int = 512
    min_cluster_size: int = 10
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "leaf"
    metric: str = "euclidean"

    use_density: bool = False
    density_lambda: float = 0.0

    fraud_purity_threshold: float = 0.5
    min_centroid_purity: float = 0.5
    align_distance_percentile: float = 95.0
    gray_zone_alpha: float = 1.5
    hard_distance_cap: float = 1.0
    min_len_gate: int = 1
    seed: int = 42
    show_n_samples: int = 3


@torch.no_grad()
def collect_record_embeddings(model, cfg: CFG2, hcfg: HBSCAN4Config, is_training: bool) -> Dict[str, Any]:
    print(f"\n--- Collecting per-record embeddings (is_training={is_training}) ---")
    # 根据传入的 is_training 标志调用 make_anchored_dataset
    data = seq_gen.make_anchored_dataset(cfg, num_trx=hcfg.eval_num_trx, is_training=is_training)
    n = len(data["dt"])
    if n == 0:
        raise RuntimeError("no records generated")

    embs: List[torch.Tensor] = []
    for s in range(0, n, hcfg.embed_batch):
        sl = slice(s, min(n, s + hcfg.embed_batch))
        batch = {k: (data[k][sl] if k != "record_scenario" else data["record_scenario"][sl])
                 for k in data if k not in _META_KEYS}
        # 提取 Embedding 时始终保持 is_training=False，确保前向传播没有 Dropout/Masking 干扰
        x_full, raw = build_full_features(batch, cfg, is_training=False)
        emb, _ = model.extract_embedding(x_full, raw["mask"])
        embs.append(emb.cpu())
    X = torch.cat(embs, dim=0).numpy().astype(np.float32)

    scenarios = list(data["record_scenario"])
    is_anom = data["record_is_fraud"].numpy().astype(np.int32)
    current_id = data["trx_id"].numpy().astype(np.int64)
    valid_len = data["valid_len"].numpy().astype(np.int64)
    anchor_type = data["anchor_type"].numpy().astype(np.int64)

    X_all = X
    trx_id_all = current_id.copy()

    keep = valid_len > hcfg.min_len_gate
    dropped = int((~keep).sum())
    X = X[keep]
    scenarios = [s for s, k in zip(scenarios, keep) if k]
    is_anom = is_anom[keep]
    current_id = current_id[keep]
    valid_len = valid_len[keep]
    anchor_type = anchor_type[keep]

    num_currents = int(data["num_trx"])
    current_scenarios = list(data["trx_scenario"])
    cid_is_anom = np.array(
        [1 if seq_gen._is_anomaly(current_scenarios[c]) else 0 for c in range(num_currents)],
        dtype=np.int32,
    )

    print(f"records embedded          : {len(X)}   (dropped {dropped} with len<= {hcfg.min_len_gate})")
    print(f"  incl. short (centroid spread): {len(X_all)}   (all records, no len gate)")
    print(f"  record-level anomaly    : {int(is_anom.sum())} / {len(is_anom)}")
    print(f"transactions (currents)   : {num_currents}   "
          f"anomaly {int(cid_is_anom.sum())} / {num_currents}")
    return {
        "X": X, "record_scenario": scenarios, "is_anom": is_anom,
        "trx_id": current_id, "valid_len": valid_len, "anchor_type": anchor_type,
        "num_trx": num_currents, "trx_scenario": current_scenarios,
        "cid_is_anom": cid_is_anom,
        "X_all": X_all, "trx_id_all": trx_id_all,
        "seq_dt": data["dt"].numpy(), "seq_amount": data["amount"].numpy(),
        "seq_sw_ip": data["sw_ip"].numpy(), "seq_sw_email": data["sw_email"].numpy(),
        "seq_sw_fp": data["sw_fp"].numpy(), "seq_sw_bill": data["sw_bill"].numpy(),
        "seq_sw_ship": data["sw_ship"].numpy(),
        "rec_anchor_type": data["anchor_type"].numpy(),
        "rec_valid_len": data["valid_len"].numpy(),
        "rec_side": data["record_pair_side"].numpy(),
        "rec_scenario_all": list(data["record_scenario"]),
        "rec_is_fraud_all": data["record_is_fraud"].numpy(),
    }


_BENIGN_SEMANTIC_MAP = {
    "Traveler": "NORMAL_TRAVELER",
    "Shopper": "NORMAL_SHOPPER",
    "Upgrader": "NORMAL_UPGRADER",
}
_BENIGN_PURITY_GATE = 0.80


def _infer_semantic_type(p_fraud: float, scen_counter: Counter,
                         size: int, fraud_threshold: float) -> str:
    if p_fraud >= fraud_threshold:
        return "FRAUD"
    if not scen_counter:
        return "NORMAL_BASELINE"
    top_sc, top_n = scen_counter.most_common(1)[0]
    if top_n / max(1, size) >= _BENIGN_PURITY_GATE and top_sc in _BENIGN_SEMANTIC_MAP:
        return _BENIGN_SEMANTIC_MAP[top_sc]
    return "NORMAL_BASELINE"


def _centroid(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X[idx].mean(axis=0).astype(np.float32)


def _align_distance(X: np.ndarray, idx: np.ndarray, centroid: np.ndarray, pct: float) -> float:
    d = np.linalg.norm(X[idx] - centroid, axis=1)
    return float(np.percentile(d, pct))


def run_hdbscan_v4(embed: Dict[str, Any], hcfg: HBSCAN4Config
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("\n--- Running record-level HDBSCAN (label-free fit) ---")
    print(f"min_cluster_size          = {hcfg.min_cluster_size}")
    print(f"min_samples               = {hcfg.min_samples}")
    print(f"cluster_selection_method  = {hcfg.cluster_selection_method}")
    print(f"metric                    = {hcfg.metric}  (euclidean on L2-norm == cosine)")

    X = embed["X"]
    scenarios = embed["record_scenario"]
    is_anom = embed["is_anom"]
    anchor_type = embed["anchor_type"]
    _ANCHOR_NAME = {0: "bca", 1: "em", 2: "fp", 3: "sa"}

    clusterer = HDBSCAN(
        min_cluster_size=hcfg.min_cluster_size,
        min_samples=hcfg.min_samples,
        cluster_selection_epsilon=hcfg.cluster_selection_epsilon,
        cluster_selection_method=hcfg.cluster_selection_method,
        metric=hcfg.metric,
    )
    labels = clusterer.fit_predict(X)

    unique = sorted(set(labels.tolist()))
    n_clusters = sum(1 for l in unique if l != -1)
    n_noise = int((labels == -1).sum())
    print(f"clusters extracted        : {n_clusters}   |   "
          f"noise: {n_noise}/{len(X)} ({100.0 * n_noise / max(1, len(X)):.2f}%)")

    clusters: List[Dict[str, Any]] = []
    for lbl in unique:
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        n_anom = int(is_anom[idx].sum())
        p_fraud = n_anom / len(idx)
        centroid = _centroid(X, idx)
        assigned = "FRAUD" if p_fraud >= hcfg.fraud_purity_threshold else "NORMAL"
        scen_counter = Counter(scenarios[i] for i in idx)
        semantic_type = _infer_semantic_type(p_fraud, scen_counter, len(idx),
                                             hcfg.fraud_purity_threshold)
        at_counter = Counter(_ANCHOR_NAME[int(anchor_type[i])] for i in idx)
        at_dom_name, at_dom_n = at_counter.most_common(1)[0]
        anchor_purity = at_dom_n / len(idx)
        clusters.append({
            "label": int(lbl), "size": int(len(idx)),
            "n_anomaly": n_anom, "n_normal": int(len(idx) - n_anom),
            "p_fraud": float(p_fraud), "purity": float(max(p_fraud, 1 - p_fraud)),
            "assigned": assigned, "semantic_type": semantic_type, "centroid": centroid,
            "anchor_purity": float(anchor_purity),
            "anchor_dominant": f"{at_dom_name}:{anchor_purity:.0%}",
            "anchor_mix": at_counter.most_common(4),
            "dominant_scenarios": scen_counter.most_common(3),
            "max_align_distance": _align_distance(X, idx, centroid, hcfg.align_distance_percentile),
            "median_align_distance": _align_distance(X, idx, centroid, 50.0),
        })

    noise_idx = np.where(labels == -1)[0]
    noise_counter = Counter(scenarios[i] for i in noise_idx)
    noise_info = {
        "count": n_noise,
        "n_anomaly": int(is_anom[noise_idx].sum()),
        "top_scenarios": noise_counter.most_common(6),
    }
    return clusters, noise_info


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

    benign = [c for c in clusters if c.get("semantic_type", "").startswith("NORMAL_")
              and c["semantic_type"] != "NORMAL_BASELINE"]
    if benign:
        print(f"\n  [LLM hint] Auto-identified benign clusters ({len(benign)}):")
        for c in sorted(benign, key=lambda x: x["semantic_type"]):
            top_sc = c["dominant_scenarios"][0] if c["dominant_scenarios"] else ("?", 0)
            print(f"    label={c['label']:2d}  {c['semantic_type']:<20s}  "
                  f"size={c['size']}  purity={top_sc[1] / max(1, c['size']):.0%}  "
                  f"top={top_sc[0]}:{top_sc[1]}")


def _best_f1(y_true: np.ndarray, y_scores: np.ndarray):
    roc = float(roc_auc_score(y_true, y_scores)) if y_true.sum() and (y_true == 0).sum() else float("nan")
    pr = float(average_precision_score(y_true, y_scores))
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    b = int(np.argmax(f1))
    best_th = float(thr[b]) if b < len(thr) else float(thr[-1] if len(thr) else 0.0)
    y_pred = (y_scores >= best_th).astype(int)
    acc = float((y_pred == y_true).mean())
    return roc, pr, float(f1[b]), best_th, float(prec[b]), float(rec[b]), acc


def evaluate_v4(embed: Dict[str, Any], clusters: List[Dict[str, Any]],
                hcfg: HBSCAN4Config) -> Dict[str, Any]:
    """Evaluate record-to-fraud-centroid alignment and return compact metrics."""
    print("\n--- Alignment scoring (record → transaction MAX) ---")
    pure = [c for c in clusters if c["purity"] >= hcfg.min_centroid_purity]
    f_clusters = [c for c in pure if c["assigned"] == "FRAUD"]
    print(f"purity gate >= {hcfg.min_centroid_purity}: kept {len(pure)}/{len(clusters)}  "
          f"| FRAUD centroids: {len(f_clusters)}")
    if not f_clusters:
        print("no FRAUD centroids — cannot score.")
        return {
            "available": False,
            "reason": "No FRAUD centroids passed the purity gate.",
            "pure_cluster_count": len(pure),
            "fraud_centroid_count": 0,
            "transactions_aligned": 0,
            "fraud_aligned": 0,
            "benign_aligned": 0,
            "per_scenario": [],
        }

    X = embed["X"]
    F_cent = np.stack([c["centroid"] for c in f_clusters], axis=0)
    f_max_align = np.array([c["max_align_distance"] for c in f_clusters], dtype=np.float32)

    D = np.stack([np.linalg.norm(X - fc, axis=1) for fc in F_cent], axis=1)
    nn_idx = np.argmin(D, axis=1)
    d_min = D[np.arange(len(X)), nn_idx]
    aligned = d_min <= f_max_align[nn_idx]

    cid = embed["trx_id"]
    num_c = embed["num_trx"]
    txn_aligned = np.zeros(num_c, dtype=bool)
    for i in range(len(X)):
        if aligned[i]:
            txn_aligned[cid[i]] = True

    y = embed["cid_is_anom"]
    cs = np.array(embed["trx_scenario"])
    aln = int(txn_aligned.sum())
    fraud_aligned = int((txn_aligned & (y == 1)).sum())
    benign_aligned = int((txn_aligned & (y == 0)).sum())
    print(f"\n[Centroid alignment coverage — analysis only]")
    print(f"  Transactions within FRAUD centroid radius: {aln}/{num_c}")
    print(f"    fraud aligned: {fraud_aligned}  |  benign aligned: {benign_aligned}")

    per_scenario = []
    print(f"\n--- Per-scenario alignment rate (* = fraud) ---")
    print(f"  {'scenario':24s} {'n':>4s} {'align%':>7s}   role")
    for sc in sorted(set(cs.tolist()),
                     key=lambda name: -float(txn_aligned[cs == name].mean()) if (cs == name).any() else 0.0):
        m = cs == sc
        if not m.any():
            continue
        is_fraud = bool(seq_gen._is_anomaly(sc))
        role = "detect" if is_fraud else "FP-rate"
        rate = float(txn_aligned[m].mean())
        count = int(m.sum())
        tag = "*" if is_fraud else " "
        print(f" {tag}{sc:23s} {count:4d} {rate:7.1%}   {role}")
        per_scenario.append({
            "scenario": str(sc),
            "n": count,
            "is_fraud": is_fraud,
            "role": role,
            "alignment_rate": rate,
        })

    return {
        "available": True,
        "pure_cluster_count": len(pure),
        "fraud_centroid_count": len(f_clusters),
        "transactions_total": int(num_c),
        "transactions_aligned": aln,
        "fraud_aligned": fraud_aligned,
        "benign_aligned": benign_aligned,
        "per_scenario": per_scenario,
    }

def evaluate_intra_centroid_spread(embed: Dict[str, Any],
                                   clusters: List[Dict[str, Any]],
                                   hcfg: HBSCAN4Config) -> Dict[str, Any]:
    """Compute transaction-level centroid spread and return internal + compact metrics."""
    print("\n--- Intra-transaction centroid spread ---")
    if not clusters:
        raise RuntimeError("HDBSCAN produced no clusters; centroid-spread scoring is unavailable")

    X = embed["X_all"]
    cid = embed["trx_id_all"]
    num_c = embed["num_trx"]
    y = embed["cid_is_anom"]
    cs = np.array(embed["trx_scenario"])

    all_centroids = np.stack([c["centroid"] for c in clusters], axis=0)

    D = np.stack([np.linalg.norm(X - c, axis=1) for c in all_centroids], axis=1)
    nearest = np.argmin(D, axis=1)

    from collections import defaultdict
    txn_to_cents: Dict[int, set] = defaultdict(set)
    for i, c in enumerate(cid):
        txn_to_cents[int(c)].add(int(nearest[i]))

    benign_cidx = {i for i, c in enumerate(clusters)
                   if c.get("semantic_type", "") not in ("FRAUD", "NORMAL_BASELINE", "")}
    benign_types = {clusters[i]["semantic_type"] for i in benign_cidx}
    if benign_types:
        print(f"\n  Benign cluster categories identified (suppression disabled): {sorted(benign_types)}")
        print(f"  Benign centroid indices: {sorted(benign_cidx)}")

    centroid_spread = np.zeros(num_c, dtype=np.float32)
    txn_suppressed = np.zeros(num_c, dtype=bool)

    for c, cent_set in txn_to_cents.items():
        # Benign-cluster suppression is intentionally disabled in the current scorer.
        if len(cent_set) < 2:
            continue
        idx_list = list(cent_set)
        cents = all_centroids[idx_list]
        rows, cols = np.triu_indices(len(idx_list), k=1)
        dist = np.linalg.norm(cents[rows] - cents[cols], axis=1)
        centroid_spread[c] = float(dist.max())

    n_suppressed = int(txn_suppressed.sum())
    if n_suppressed:
        fraud_supp = int((txn_suppressed & y.astype(bool)).sum())
        benign_supp = n_suppressed - fraud_supp
        print(f"  Suppressed transactions: {n_suppressed}  "
              f"(fraud={fraud_supp}, benign={benign_supp})")

    roc, pr, f1, th, prec, rec, acc = _best_f1(y, centroid_spread)
    print(f"\n[Centroid-spread score — transaction level]")
    print(f"  ROC-AUC {roc:.4f} | PR-AUC {pr:.4f} | best-F1 {f1:.4f} @ spread >= {th:.4f}")
    print(f"  precision {prec:.4f} | recall {rec:.4f} | accuracy {acc:.4f}")

    y_pred = (centroid_spread >= th).astype(int)
    tp = int((y_pred & y).sum())
    fp = int((y_pred & (1 - y)).sum())
    fn = int(((1 - y_pred) & y).sum())
    tn = int(((1 - y_pred) & (1 - y)).sum())
    fraud_detection_rate = tp / max(1, tp + fn)
    benign_fp_rate = fp / max(1, fp + tn)
    print(f"\n--- Confusion @ best-F1 (centroid spread) ---")
    print(f"                 pred_fraud   pred_benign")
    print(f"  actual_fraud   {tp:10d}   {fn:11d}")
    print(f"  actual_benign  {fp:10d}   {tn:11d}")
    print(f"  fraud detection rate {fraud_detection_rate:.1%}  |  benign FP rate {benign_fp_rate:.1%}")

    per_scenario = []
    print(f"\n--- Per-pattern centroid-spread (transaction level; * = fraud) ---")
    print(f"  {'scenario':24s} {'n':>4s} {'mean_cs':>7s} {'flag%':>6s} {'supp%':>6s}   role")
    order = sorted(set(cs.tolist()),
                   key=lambda name: -float(centroid_spread[cs == name].mean()) if (cs == name).any() else 0.0)
    for sc in order:
        m = cs == sc
        if not m.any():
            continue
        is_fraud = bool(seq_gen._is_anomaly(sc))
        role = "detect" if is_fraud else "FP-rate"
        supp = float(txn_suppressed[m].mean())
        mean_spread = float(centroid_spread[m].mean())
        flag_rate = float(y_pred[m].mean())
        count = int(m.sum())
        tag = "*" if is_fraud else " "
        print(f" {tag}{sc:23s} {count:4d} {mean_spread:7.4f} "
              f"{flag_rate:6.1%} {supp:6.1%}   {role}")
        per_scenario.append({
            "scenario": str(sc),
            "n": count,
            "is_fraud": is_fraud,
            "role": role,
            "mean_centroid_spread": mean_spread,
            "flag_rate": flag_rate,
            "suppressed_rate": supp,
        })

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "best_f1": f1,
        "threshold": th,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "fraud_detection_rate": fraud_detection_rate,
        "benign_fp_rate": benign_fp_rate,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "transactions_total": int(num_c),
        "fraud_transactions": int(y.sum()),
        "benign_transactions": int((y == 0).sum()),
    }

    # Keep internal arrays for existing show_samples() callers, while also
    # exposing compact JSON-friendly summaries for MCP transport.
    return {
        "y_pred": y_pred,
        "spread": centroid_spread,
        "threshold": th,
        "nearest": nearest,
        "all_centroids": all_centroids,
        "cluster_types": [c.get("semantic_type", "") for c in clusters],
        "metrics": metrics,
        "per_scenario": per_scenario,
    }


_ANCHOR_NAME = {0: "BCA", 1: "EM", 2: "FP", 3: "SA"}
_SIDE_NAME = {0: "main", 1: "contrast"}


def _print_record_steps(embed: Dict[str, Any], i: int, L: int) -> None:
    dt = embed["seq_dt"][i]
    amt = embed["seq_amount"][i]
    sw = {"ip": embed["seq_sw_ip"][i], "em": embed["seq_sw_email"][i],
          "fp": embed["seq_sw_fp"][i], "bill": embed["seq_sw_bill"][i],
          "ship": embed["seq_sw_ship"][i]}
    print(f"        {'step':>4s} {'dt(s)':>11s} {'amt':>9s}  switches")
    for k in range(L):
        flags = ",".join(n for n, v in sw.items() if v[k] > 0.5) or "-"
        print(f"        {k:4d} {dt[k]:11.1f} {amt[k]:9.2f}  {flags}")


def show_samples(embed: Dict[str, Any], num_samples: int, pattern_name: str,
                 spread_info, want_caught: bool = True,
                 show_steps: bool = True) -> None:
    num_c = embed["num_trx"]
    cs = np.array(embed["trx_scenario"])

    fraud_transactions = spread_info["y_pred"]
    nearest = spread_info["nearest"]
    all_cents = spread_info["all_centroids"]
    ctypes = spread_info["cluster_types"]
    spread_val = spread_info["spread"]
    th = spread_info["threshold"]

    caught = np.zeros(num_c, dtype=bool)
    ft = np.asarray(list(fraud_transactions)) if not isinstance(fraud_transactions, np.ndarray) \
        else fraud_transactions
    if ft.dtype == bool and ft.shape[0] == num_c:
        caught = ft
    elif ft.shape[0] == num_c and ft.min() >= 0 and ft.max() <= 1:
        caught = ft.astype(bool)
    else:
        caught[ft.astype(int)] = True

    cand = [c for c in range(num_c)
            if (cs[c] == pattern_name or cs[c].startswith(pattern_name))
            and bool(caught[c]) == want_caught]
    tag = "PREDICTED-FRAUD" if want_caught else "PREDICTED-BENIGN"
    print(f"\n===== showSamples('{pattern_name}', {tag}) "
          f"({len(cand)} available, showing up to {num_samples}) =====")
    if not cand:
        print(f"  (no {tag} transactions for this pattern)")
        return

    trx_id = embed["trx_id_all"]
    for c in cand[:num_samples]:
        ridx = np.where(trx_id == c)[0]
        print(f"\n--- trx #{c}  scenario={cs[c]}  ({len(ridx)} anchor-view records)"
              f"  spread={spread_val[c]:.4f} (>= {th:.4f} => FRAUD) ---")
        view_label = {}
        for i in sorted(ridx, key=lambda j: (int(embed["rec_side"][j]),
                                             int(embed["rec_anchor_type"][j]))):
            L = int(embed["rec_valid_len"][i])
            anc = _ANCHOR_NAME.get(int(embed["rec_anchor_type"][i]), "?")
            side = _SIDE_NAME.get(int(embed["rec_side"][i]), "?")
            fr = "FRAUD" if embed["rec_is_fraud_all"][i] else "benign"
            cidx = int(nearest[i])
            view_label[i] = f"{anc}/{side}"
            print(f"  [{anc:3s} / {side:8s}]  len={L:2d}  record={fr}"
                  f"  -> centroid c{cidx:02d}({ctypes[cidx] or '?'})")
            if show_steps:
                _print_record_steps(embed, i, L)

        cent_of = {i: int(nearest[i]) for i in ridx}
        best = None
        for a in ridx:
            for b in ridx:
                if b <= a or cent_of[a] == cent_of[b]:
                    continue
                d = float(np.linalg.norm(all_cents[cent_of[a]] - all_cents[cent_of[b]]))
                if best is None or d > best[0]:
                    best = (d, a, b)
        if best is None:
            print(f"  >> all views share one centroid — spread driven at txn level "
                  f"(no divergent pair; see suppression/threshold)")
        else:
            d, a, b = best
            print(f"  >> FLAG driver: {view_label[a]} (c{cent_of[a]:02d}) vs "
                  f"{view_label[b]} (c{cent_of[b]:02d})  centroid-dist={d:.4f}")


# ========================================================
# 更新：拆分质心构建与模型评估逻辑
# ========================================================
def build_hdbscan_centroids(model, cfg: CFG2, hcfg: HBSCAN4Config):
    """
    Phase 1: 质心构建。
    模拟无监督的历史数据学习过程，传入 is_training=True，允许异常特征随机出现在序列的任意位置。
    """
    print("\n" + "=" * 60)
    print(" PHASE 1: BUILDING CENTROIDS (is_training=True)")
    print("=" * 60)
    embed_train = collect_record_embeddings(model, cfg, hcfg, is_training=True)
    clusters, noise_info = run_hdbscan_v4(embed_train, hcfg)
    return clusters, noise_info


def evaluate_on_clusters(model, cfg: CFG2, hcfg: HBSCAN4Config, clusters: List[Dict[str, Any]]):
    """
    Phase 2: 评估与扩散评分计算。
    模拟实时在线交易判定，传入 is_training=False，确保当前正在被评估的交易（可能是攻击步）强制对齐在序列的最右侧。
    """
    print("\n" + "=" * 60)
    print(" PHASE 2: EVALUATION ON BUILT CLUSTERS (is_training=False)")
    print("=" * 60)
    embed_eval = collect_record_embeddings(model, cfg, hcfg, is_training=False)
    alignment = evaluate_v4(embed_eval, clusters, hcfg)
    spread_info = evaluate_intra_centroid_spread(embed_eval, clusters, hcfg)
    spread_info["alignment"] = alignment
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
    # Avoid initializing CUDA solely for an evaluation running on CPU.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def run_hdbscan_evaluation(model, cfg: CFG2, hcfg: Optional[HBSCAN4Config] = None) -> Dict[str, Any]:
    """Run the two-phase HDBSCAN evaluation on an in-memory model.

    The function is safe to call during an active training session: it restores
    the model's train/eval mode and Python/NumPy/Torch RNG states before return.
    Returned data is compact and transport-friendly; large embeddings, centroid
    vectors, and per-transaction arrays are intentionally omitted.
    """
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

        # Phase 1: build centroids from training-style anchored views.
        clusters, noise_info = build_hdbscan_centroids(model, cfg, hcfg)
        report_clusters(clusters, noise_info)
        if not clusters:
            raise RuntimeError("HDBSCAN produced no non-noise clusters")

        # Phase 2: score evaluation-style anchored views.
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
    # update to override to use a specified model_path
    cfg.model_path = cfg.model_path
    hcfg = HBSCAN4Config()
    set_seed(hcfg.seed)
    print("===== HDBSCAN v4 — label-free record clustering on Seq-on-Graph =====")
    model = load_trained_model(cfg)

    # 第一阶段：构建聚类质心 (is_training=True)
    clusters, noise_info = build_hdbscan_centroids(model, cfg, hcfg)
    report_clusters(clusters, noise_info)

    # 第二阶段：在已构建的簇上执行实时评估打分 (is_training=False)
    embed_eval, spread = evaluate_on_clusters(model, cfg, hcfg, clusters)


if __name__ == "__main__":
    main()
