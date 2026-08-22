import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# ---------------------------
# Config
# ---------------------------

@dataclass
class CFG:
    """Fixed experiment configuration.

    Parameters in this class are part of the baseline experiment definition and
    are not eligible for diagnostic-agent overrides.
    """
    diagnostic_config_path = (
            PACKAGE_ROOT
            / "config"
            / "model_structure.diagnostic.json"
    )

    model_path: str = "./model/seq_encoder_seq_on_graph.tar"
    seed: int = 2026
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    DAY = 86400.0
    WEEK = DAY * 7

    # Data Dimensions / Training Runtime
    num_seqs: int = 12000
    num_trx: int = 5000

    print_every: int = 200

    max_seq_len: int = 35
    min_seq_len: int = 5

    # Robust Scaling
    epsilon: float = 1e-6

    # Robust Masking baseline behavior
    # Whether masking is enabled and the mask span definition remain fixed.
    use_robust_masking: bool = False
    p_mask_range = (1, 6)
    # Training augmentation
    p_mask: float = 0.15

    # === Scenario Probabilities ===
    p_ato: float = 0.05
    p_stuffing: float = 0.05
    p_stuff_ato: float = 0.03
    p_traveler: float = 0.05
    p_shopper: float = 0.07
    p_upgrader: float = 0.07
    p_chotic_normal: float = 0.05

    anomaly_patterns = [
        "Stuffing_low_and_slow",
        "Stuffing_rotating_proxy",
        "Stuffing_dumb_script",
        "Stuffing_card_tester",
        "ATO_classic",
        "ATO_blitz",
        "ATO_sneaky",
    ]
    normal_patterns = ["normal", "Shopper", "Upgrader", "Traveler", "Chaotic_Normal"]

    # Feature / model shape contract.
    # These values are tied to generated feature shapes or head cardinalities,
    # so they are not exposed as agent overrides.
    # input_dim: int = 21
    num_anchor_types: int = 4
    input_base_dim: int = 16
    sw_classes: int = 32
    is_new_classes: int = 5

    # Feature Engineering Params
    taus: Tuple[float, ...] = (10.0, 600.0, 3600.0, DAY, WEEK)

    # seconds
    dt_range_normal = (300.0, 60 * 60 * 24 * 7)
    dt_short_range_normal_p = 0.1
    dt_short_range_normal = (20, 300)
    shopper_dt_range_fast = (5.0, 60.0 * 6)
    bot_dt_range = (0.01, 10.0)
    bot_dt_range_low_slow = (120.0, 600.0)
    traveler_dt_range = (300, 60 * 60 * 2)
    ato_dt_range_1_2 = (10.0, 120.0)
    ato_dt_range_3 = (1.0, 10.0)

    ato_len_range = (1, 3)
    traveler_len_range = (3, 5)
    shopper_len_range = (3, 6)

    ip_sw_p = 0.1
    stuffing_bill_sw_p = 0.3

    amt_clip_val: float = 4.0
    amt_decrease_weight = 0.2
    amt_mean = 3.0
    amt_dv = 1.2
    amt_high_range_p = 0.03
    amt_high_range = [100, 3000]
    amt_mid_range = [100, 500]
    ato_amt_mean = 4.0
    ato_amt_dv = 1.2
    ato_1time_amt_mean = 6.5
    ato_1time_amt_dv = 1.0
    stuffing_amt_range = (0.0, 5.0)
    bot_camouflage_prob = 0.4

    # 标签稀缺性模拟
    positive_p = 0.5
    negative_p = 1

    # Scoring / evaluation policy
    score_w_emb: float = 0.1
    score_w_len: float = 0.6
    score_w_den: float = 0.3
    score_pair_agg: str = "mean"
    score_wmean_temp: float = 0.25
    score_fp_budget: float = 0.05

    # Multi-task loss definition stays fixed for now.
    lambda_v: float = 1.0
    lambda_amt: float = 1.0
    lambda_sw: float = 1.0
    lambda_sw_bit = 1.0
    lambda_recon: float = 1.0
    lambda_is_new: float = 1.0

    aug_crop_p: float = 0.0

    # VICReg alignment experiment parameters
    use_vicreg: bool = False
    vicreg_expander_dim: int = 256
    vicreg_sim_coef: float = 25.0
    vicreg_std_coef: float = 25.0
    vicreg_cov_coef: float = 1.0
    lambda_align: float = 1.0
    align_warmup_steps: int = 500

    embedding_dim: int = 128
    time_emb_dim: int = 16
    combo_dim: int = 16
    attn_pooling_step_len: int = 3


@dataclass
class CFG2(CFG):
    """Diagnostic-agent overridable experiment parameters only.

    restart_session() may apply controlled overrides only to parameters defined
    directly in this class. CFG remains fixed.

    ``get_tunable_parameters()`` is the single source of truth consumed by the
    diagnostic runtime and MCP server.
    """

    steps: int = 3000
    steps: int = 3000
    txn_batch_size: int = 64

    # A week
    # tau_sw = 86400.0 * 7
    tau_sw = 86400.0

    lr: float = 1e-3
    # clip_val: float = 1.0
    clip_val: float = 5.0

    # Model architecture parameters that can be retrained from scratch
    # hidden_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    @classmethod
    def get_tunable_parameters(cls) -> Dict[str, Any]:
        """Return authoritative agent-overridable CFG2 parameters and defaults.

        A parameter is tunable iff it is a public data attribute declared
        directly on CFG2. Inherited CFG attributes are intentionally excluded.
        This also includes unannotated fields such as ``tau_sw``.
        """
        cfg = cls()
        result: Dict[str, Any] = {}

        for name, declared_value in vars(cls).items():
            if name.startswith("_"):
                continue
            if isinstance(declared_value, (classmethod, staticmethod, property)):
                continue
            if callable(declared_value):
                continue
            result[name] = getattr(cfg, name)

        return dict(sorted(result.items()))


@dataclass
class HBSCAN4Config:
    eval_num_trx: int = 3000
    embed_batch: int = 512
    min_cluster_size: int = 10
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "leaf"
    metric: str = "euclidean"

    # Retained for backward-compatible experiment config serialization.
    # They are intentionally not wired into the current executable DAG yet.
    use_density: bool = False
    density_lambda: float = 0.0

    fraud_purity_threshold: float = 0.5
    min_centroid_purity: float = 0.5
    align_distance_percentile: float = 95.0

    # Retained but intentionally not wired into the current scorer yet.
    gray_zone_alpha: float = 1.5
    hard_distance_cap: float = 1.0

    min_len_gate: int = 1
    seed: int = 42

    # Reporting/debug-only compatibility parameter; not part of the DAG algorithm.
    show_n_samples: int = 3
