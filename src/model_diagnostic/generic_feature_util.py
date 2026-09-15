import random
import numpy as np
import time
import torch
import math


def compute_stable_log(input_tensor: torch.Tensor, config: str = None) -> torch.Tensor:
    """
    Computes numerically stable logarithms directly on the input tensor.
    Supported configs: 'log1p', 'log21p', 'log101p'
    """
    cfg = 'log1p' if config is None else config

    if cfg == "log1p":
        return torch.log1p(input_tensor)

    elif cfg == "log21p":
        # log2(x + 1) = ln(x + 1) / ln(2)
        return torch.log1p(input_tensor) / math.log(2)

    elif cfg == "log101p":
        # log10(x + 1) = ln(x + 1) / ln(10)
        return torch.log1p(input_tensor) / math.log(10)

    else:
        raise ValueError(
            f"Unsupported configuration: '{config}'. "
            f"Choose from 'log1p', 'log21p', or 'log101p'."
        )


def fields_to_history(fields: list[torch.Tensor], is_inverse: list[bool], mask: torch.Tensor) -> list[
    torch.Tensor]:
    """
    Calculates the dynamic change rate (Log1p Ratio) of sequence fields relative to their historical average.

    Args:
        fields (list[torch.Tensor]): Tensors to process, shape [B, L]
        mask (torch.Tensor): Sequence mask, shape [B, L]
        is_inverse (list[bool]): If True, calculates (hist_avg / current) to capture unnatural drops (e.g., dt).
                                 If False/None, calculates (current / hist_avg) to capture unnatural spikes (e.g., amt).
        cfg: Config object

    Returns:
        list[torch.Tensor]: Processed tensors in the exact order of `fields`.
    """
    run_cnt = torch.cumsum(mask, dim=1)
    results = []

    # Default all to forward logic if not specified
    if is_inverse is None:
        is_inverse = [False] * len(fields)

    for i, field in enumerate(fields):
        # 1. Cumulative sum under mask
        run_sum = torch.cumsum(field * mask, dim=1)

        # 2. Moving average (add 1e-6 to prevent division by zero)
        moving_avg = run_sum / (run_cnt + 1e-6)

        # 3. Shift right to get historical average (excluding current step)
        hist_avg = torch.roll(moving_avg, shifts=1, dims=1)

        # 4. First step has no history, use itself as baseline
        hist_avg[:, 0] = field[:, 0]

        # 5. Apply the correct logic based on feature semantics
        if is_inverse[i]:
            # Inverse logic: Captures unnatural drops (e.g., time intervals shrinking due to bots)
            field_to_hist = torch.log1p(hist_avg / (field + 1e-6)) * mask
        else:
            # Forward logic: Captures unnatural spikes (e.g., transaction amounts surging)
            field_to_hist = torch.log1p(field / (hist_avg + 1e-6)) * mask

        results.append(field_to_hist)

    return results


def dt_to_pre(dt):
    dt_prev = torch.zeros_like(dt)
    dt_prev[:, 1:] = dt[:, :-1]
    return dt_prev / (dt + dt_prev + 1e-6)

def amt_to_pre(amt):
    amt_prev = torch.zeros_like(amt)
    amt_prev[:, 1:] = amt[:, :-1]
    return amt / (amt + amt_prev + 1e-6)

def amt_norm(amt, cfg):
    a_t = torch.log1p(amt)
    return torch.clamp((a_t - cfg.amt_mean) / cfg.amt_dv, -cfg.amt_clip_val, cfg.amt_clip_val)

def combine_sw(targets):
    target_class = (
            targets["sw_ip"] * 16 +
            targets["sw_email"] * 8 +
            targets["sw_fp"] * 4 +
            targets["sw_bca"] * 2 +
            targets["sw_ship"] * 1
    ).long()
    return target_class

# ========================================================
# 更新：为 is_new 头提供 Multi-label Binary Target 打包
# ========================================================
def combine_is_new(raw_targets):
    # 将字典中的 5 个 is_new label 堆叠成张量 [B, L-1, 5]
    return torch.stack([
        raw_targets["is_new_ip"][:, 1:],
        raw_targets["is_new_email"][:, 1:],
        raw_targets["is_new_fp"][:, 1:],
        raw_targets["is_new_bca"][:, 1:],
        raw_targets["is_new_ship"][:, 1:]
    ], dim=-1)

def get_decay_factor(cfg, dt):
    tau_sw = cfg.tau_sw
    switch_decay_factor = torch.exp(-dt / tau_sw)
    return switch_decay_factor

def apply_sw_smooth_weight(loss, sw_decay, base_weight=0.2):
    return loss
    '''weight = base_weight + (1.0 - base_weight) * sw_decay
    return loss * weight'''

def set_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000000) % 1000000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
