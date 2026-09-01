import copy
import math
import random
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import time

import torch

from model_diagnostic.cfg_base import CFG2


def gen_one_sequence2(cfg, random_pos_anomaly: bool = False) -> Tuple[np.ndarray, ...]:
    force_right_align = not random_pos_anomaly
    valid_len = random.randint(cfg.min_seq_len, cfg.max_seq_len)
    T = cfg.max_seq_len
    dt = np.zeros(T, dtype=np.float32)
    amt = np.zeros(T, dtype=np.float32)
    sw_ip = np.zeros(T, dtype=np.float32)
    sw_email = np.zeros(T, dtype=np.float32)
    sw_fp = np.zeros(T, dtype=np.float32)
    sw_bill = np.zeros(T, dtype=np.float32)
    sw_ship = np.zeros(T, dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)
    pattern_start = 0
    v_dt = np.zeros(valid_len, dtype=np.float32)
    v_amt = np.zeros(valid_len, dtype=np.float32)
    v_sw_ip = np.zeros(valid_len, dtype=np.float32)
    v_sw_email = np.zeros(valid_len, dtype=np.float32)
    v_sw_fp = np.zeros(valid_len, dtype=np.float32)
    v_sw_bill = np.zeros(valid_len, dtype=np.float32)
    v_sw_ship = np.zeros(valid_len, dtype=np.float32)
    scenario = "normal"
    rand = random.random()
    thresh_ato = cfg.p_ato
    thresh_stuff = thresh_ato + cfg.p_stuffing
    thresh_stuff_ato = thresh_stuff + getattr(cfg, 'p_stuff_ato', 0.0)
    thresh_trav = thresh_stuff_ato + cfg.p_traveler
    thresh_shop = thresh_trav + cfg.p_shopper
    thresh_upg = thresh_shop + cfg.p_upgrader
    thresh_chotic_normal = thresh_upg + cfg.p_chotic_normal

    def get_start_idx(v_len, i_len, is_aligned):
        i_len = min(i_len, v_len)
        max_start = v_len - i_len
        return max_start if is_aligned else random.randint(0, max_start)

    for t in range(valid_len):
        v_dt[t] = random.uniform(*cfg.dt_range_normal)
        if random.random() < cfg.dt_short_range_normal_p:
            v_dt[t] = random.uniform(*cfg.dt_short_range_normal)
        v_amt[t] = np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv)
        if random.random() < cfg.amt_high_range_p:
            v_amt[t] = random.uniform(*cfg.amt_high_range)
        if random.random() < cfg.ip_sw_p:
            v_sw_ip[t] = 1.0
        if random.random() < 0.01:
            v_sw_fp[t] = 1.0
        if random.random() < 0.03:
            v_sw_email[t] = 1.0
            v_sw_ip[t] = 1.0 if random.random() < 0.15 else 0.0
            v_sw_fp[t] = 1.0 if random.random() < 0.08 else 0.0
        if random.random() < 0.04:
            v_sw_bill[t] = 1.0
            v_sw_fp[t] = 1.0 if random.random() < 0.08 else 0.0
        if random.random() < getattr(cfg, 'ship_sw_p', 0.04):
            v_sw_ship[t] = 1.0
        clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship)

    if rand < thresh_ato:
        scenario = "ATO"
        inj_len = min(random.randint(*cfg.ato_len_range), valid_len)
        start = get_start_idx(valid_len, inj_len, force_right_align)
        pattern_start = start
        attack_style = random.choice(["classic", "blitz", "sneaky"])
        scenario = "ATO_" + attack_style
        for i in range(inj_len):
            t = start + i
            if inj_len == 1:
                if random.random() < 0.9:
                    v_amt[t] = random.uniform(*cfg.amt_high_range)
                else:
                    v_amt[t] = np.random.lognormal(mean=cfg.ato_1time_amt_mean, sigma=cfg.ato_1time_amt_dv)
            else:
                if i == 0 and random.random() < 0.7:
                    v_amt[t] = np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv)
                else:
                    v_amt[t] = np.random.lognormal(mean=cfg.ato_amt_mean, sigma=cfg.ato_amt_dv)
            if attack_style == "classic":
                if inj_len == 1:
                    v_sw_ip[t] = 1.0 if random.random() < 0.8 else 0.0
                    v_sw_fp[t] = 1.0 if random.random() < 0.9 else 0.0
                    v_sw_email[t] = 1.0 if random.random() < 0.95 else 0.0
                    v_sw_bill[t] = 0.0
                    v_sw_ship[t] = 1.0 if random.random() < 0.7 else 0.0
                    v_dt[t] = random.uniform(*cfg.dt_range_normal)
                else:
                    if i == 0:
                        v_sw_ip[t] = 1.0
                        v_sw_fp[t] = 1.0
                        v_dt[t] = random.uniform(*cfg.dt_range_normal)
                    elif i == 1:
                        v_sw_email[t] = 1.0
                        v_sw_ship[t] = 1.0
                        v_dt[t] = random.uniform(*cfg.ato_dt_range_1_2)
                    else:
                        v_dt[t] = random.uniform(*cfg.ato_dt_range_3)
            elif attack_style == "blitz":
                if i == 0:
                    v_sw_ip[t] = 1.0
                    v_sw_fp[t] = 1.0
                    v_sw_email[t] = 1.0
                    v_sw_bill[t] = 0
                    v_sw_ship[t] = 1.0
                    v_dt[t] = random.uniform(*cfg.dt_range_normal)
                    if inj_len > 1:
                        v_amt[t] = np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv)
                else:
                    v_dt[t] = random.uniform(*cfg.ato_dt_range_3)
                    v_amt[t] = np.random.lognormal(mean=cfg.ato_amt_mean, sigma=cfg.ato_amt_dv)
            elif attack_style == "sneaky":
                sneaky_dt = getattr(cfg, "ato_sneaky_dt_range", (5.0, 30.0))
                if i == 0:
                    v_sw_ip[t] = 0.0
                    v_sw_fp[t] = 1.0
                    v_sw_email[t] = 1.0
                    v_sw_ship[t] = 1.0
                    v_dt[t] = random.uniform(*sneaky_dt)
                else:
                    v_dt[t] = random.uniform(*sneaky_dt)
                    v_amt[t] = np.random.lognormal(mean=cfg.ato_amt_mean, sigma=cfg.ato_amt_dv)
    elif rand < thresh_stuff:
        scenario = "Stuffing"
        stuff_len = max(1, int(valid_len * random.uniform(0.5, 0.9)))
        start = get_start_idx(valid_len, stuff_len, force_right_align)
        pattern_start = start
        bot_style = random.choice(["rotating_proxy", "dumb_script", "low_and_slow", "card_tester"])
        scenario = "Stuffing_" + bot_style
        is_testing_cards = False
        for i in range(stuff_len):
            t = start + i
            v_dt[t] = random.uniform(*cfg.bot_dt_range)
            bot_camouflage_prob = getattr(cfg, 'bot_camouflage_prob', 0.8)
            if random.random() < bot_camouflage_prob:
                v_amt[t] = np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv)
            else:
                v_amt[t] = random.uniform(*cfg.stuffing_amt_range)
            if bot_style == "rotating_proxy":
                if random.random() < 0.7:
                    v_sw_ip[t] = 1.0
                    v_sw_fp[t] = 1.0
            elif bot_style == "low_and_slow":
                v_dt[t] = random.uniform(*cfg.bot_dt_range_low_slow)
                v_sw_ip[t] = 0.0
                v_sw_fp[t] = 0.0
                if random.random() < 0.8:
                    v_sw_bill[t] = 1.0
            elif bot_style == "dumb_script":
                v_sw_ip[t] = 0.0
                v_sw_fp[t] = 0.0
                if random.random() < 0.7:
                    v_sw_bill[t] = 1.0
            elif bot_style == "card_tester":
                if not is_testing_cards:
                    v_dt[t] = random.uniform(10.0, 60.0)
                    v_sw_ip[t] = 0.0
                    v_sw_fp[t] = 0.0
                    v_sw_bill[t] = 0.0
                    if random.random() < 0.3:
                        is_testing_cards = True
                elif is_testing_cards:
                    v_dt[t] = random.uniform(1.0, 3.0)
                    v_amt[t] = random.uniform(*cfg.stuffing_amt_range)
                    v_sw_bill[t] = 1.0
                    v_sw_fp[t] = 1.0
                    v_sw_ip[t] = 1.0 if random.random() < 0.5 else 0.0
    elif rand < thresh_stuff_ato:
        scenario = "Stuffing_then_ATO"
        combo_len = max(2, int(valid_len * random.uniform(0.5, 0.9)))
        start = get_start_idx(valid_len, combo_len, force_right_align)
        pattern_start = start
        ato_tail = min(random.randint(2, 3), combo_len - 1)
        stuff_phase = combo_len - ato_tail
        for i in range(combo_len):
            t = start + i
            if i < stuff_phase:
                v_dt[t] = random.uniform(*cfg.bot_dt_range)
                v_sw_ip[t] = 0.0
                v_sw_fp[t] = 0.0
                if random.random() < 0.7:
                    v_sw_bill[t] = 1.0
                if random.random() < getattr(cfg, 'bot_camouflage_prob', 0.4):
                    v_amt[t] = np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv)
                else:
                    v_amt[t] = random.uniform(*cfg.stuffing_amt_range)
            else:
                j = i - stuff_phase
                if j == 0:
                    v_sw_email[t] = 1.0
                    v_sw_ship[t] = 1.0
                    v_sw_bill[t] = 0.0
                    v_dt[t] = random.uniform(*cfg.ato_dt_range_1_2)
                else:
                    v_dt[t] = random.uniform(*cfg.ato_dt_range_3)
                if random.random() < 0.3:
                    v_amt[t] = random.uniform(*cfg.amt_mid_range)
                else:
                    v_amt[t] = np.random.lognormal(mean=cfg.ato_amt_mean, sigma=cfg.ato_amt_dv)
    elif rand < thresh_trav:
        scenario = "Traveler"
        inj_len = min(random.randint(*cfg.traveler_len_range), valid_len)
        start = get_start_idx(valid_len, inj_len, force_right_align)
        pattern_start = start
        for i in range(inj_len):
            t = start + i
            v_dt[t] = random.uniform(*cfg.traveler_dt_range)
            v_sw_ip[t] = 1.0
            v_sw_fp[t] = 1.0 if random.random() < 0.1 else 0.0
            v_sw_bill[t] = 1.0 if random.random() < 0.05 else 0.0
            v_sw_email[t] = 0.0
            clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship)
    elif rand < thresh_shop:
        scenario = "Shopper"
        inj_len = min(random.randint(*cfg.shopper_len_range), valid_len)
        start = get_start_idx(valid_len, inj_len, force_right_align)
        pattern_start = start
        for i in range(inj_len):
            t = start + i
            v_sw_email[t] = 0
            rand_dt = random.random()
            if rand_dt < 0.1:
                v_dt[t] = random.uniform(1.0, 5.0)
            elif rand_dt < 0.4:
                v_dt[t] = random.uniform(60.0, 600.0)
            else:
                v_dt[t] = random.uniform(15.0, 90.0)
            if random.random() < 0.1:
                v_amt[t] = random.uniform(*cfg.amt_high_range)
            elif random.random() < 0.2:
                v_amt[t] = random.uniform(*cfg.amt_mid_range)
            else:
                v_amt[t] = np.random.lognormal(mean=cfg.amt_mean + 0.5, sigma=0.5)
            if random.random() < 0.25:
                v_sw_bill[t] = 1.0
            if random.random() < 0.05:
                v_sw_fp[t] = 1.0
                v_sw_ip[t] = 1.0 if random.random() < 0.5 else 0.0
            clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship)
    elif rand < thresh_upg:
        scenario = "Upgrader"
        t = valid_len - 1 if force_right_align else random.randint(0, valid_len - 1)
        pattern_start = t
        upgrade_style = random.choice(["new_device", "new_card"])
        if upgrade_style == "new_device":
            v_sw_fp[t] = 1.0
            v_sw_ip[t] = 1.0
        else:
            v_sw_bill[t] = 1.0
            v_sw_ip[t] = 0.0
        clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship)
    elif rand < thresh_chotic_normal:
        scenario = "Chaotic_Normal"
        t = random.randint(0, valid_len - 1)
        pattern_start = t
        v_dt[t] = random.uniform(1.0, 5.0)
        v_sw_ip[t] = 1.0
        v_sw_fp[t] = 1.0
        v_sw_bill[t] = 1.0
        v_sw_email[t] = 1.0
        v_sw_ship[t] = 1.0
        if random.random() < 0.5:
            v_amt[t] = random.uniform(*cfg.amt_high_range)
        else:
            v_amt[t] = np.random.lognormal(mean=cfg.amt_mean + 1.0, sigma=0.5)
        clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship)

    dt[:valid_len] = v_dt
    amt[:valid_len] = v_amt
    sw_ip[:valid_len] = v_sw_ip
    sw_email[:valid_len] = v_sw_email
    sw_fp[:valid_len] = v_sw_fp
    sw_bill[:valid_len] = v_sw_bill
    sw_ship[:valid_len] = v_sw_ship
    mask[:valid_len] = 1.0
    return dt, amt, sw_ip, sw_email, sw_fp, sw_bill, sw_ship, mask, scenario, pattern_start


def clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill, v_sw_ship):
    if scenario in ["normal", "Shopper", "Traveler", "Upgrader"]:
        if v_sw_email[t] == 1.0 or v_sw_fp[t] == 1.0 or v_sw_bill[t] == 1.0 or v_sw_ship[t] == 1.0:
            v_dt[t] = random.uniform(*cfg.dt_range_normal)


def _clean_label(cfg, scenario, t, v_amt, v_dt, v_sw_email, v_sw_fp, v_sw_bill):
    if scenario in ["normal", "Shopper", "Traveler", "Upgrader"]:
        if v_sw_email[t] == 1.0 and v_sw_fp[t] == 1.0:
            if random.random() < 0.5:
                v_amt[t] = min(v_amt[t], cfg.amt_mean)
            else:
                v_dt[t] = random.uniform(*cfg.dt_range_normal)


ANCHOR_FIELDS: Tuple[str, ...] = ("bca", "em", "fp", "sa")
ANCHOR_TYPE_ID: Dict[str, int] = {"bca": 0, "em": 1, "fp": 2, "sa": 3}
NUM_ANCHOR_TYPES: int = len(ANCHOR_FIELDS)
SIDE_ID: Dict[str, int] = {"main": 0, "contrast": 1}
_SW_TO_ENTITY: Dict[str, str] = {
    "sw_ip": "ip",
    "sw_email": "em",
    "sw_fp": "fp",
    "sw_bill": "bca",
    "sw_ship": "sa",
}
_ENTITY_KEYS: Tuple[str, ...] = ("bca", "em", "fp", "sa", "ip")


def _is_anomaly(scenario: str) -> bool:
    return scenario.startswith("ATO") or scenario.startswith("Stuffing")

class _EntityAllocator:
    def __init__(self, start: int = 1):
        self._next = start

    def new(self) -> int:
        v = self._next
        self._next += 1
        return v


def _forced_cfg(cfg, scenario: Optional[str]):
    if scenario is None:
        return cfg
    c = copy.copy(cfg)
    for name in ("p_ato", "p_stuffing", "p_stuff_ato", "p_traveler",
                 "p_shopper", "p_upgrader", "p_chotic_normal"):
        setattr(c, name, 0.0)
    if scenario == "ATO":
        c.p_ato = 1.0
    elif scenario == "Stuffing":
        c.p_stuffing = 1.0
    elif scenario == "Stuffing_then_ATO":
        c.p_stuff_ato = 1.0
    elif scenario == "Traveler":
        c.p_traveler = 1.0
    elif scenario == "Shopper":
        c.p_shopper = 1.0
    elif scenario == "Upgrader":
        c.p_upgrader = 1.0
    return c


# ========================================================
# 更新：为 _gen_session_txns 显式挂载 is_training 参数
# ========================================================
def _gen_session_txns(cfg, alloc: _EntityAllocator, scenario: Optional[str],
                      start_time: float, identity: Optional[Dict[str, int]] = None,
                      share: Optional[Dict[str, int]] = None,
                      stable_fields: frozenset = frozenset(),
                      random_pos_anomaly: bool = False) -> Tuple[List[dict], str, int]:
    fcfg = _forced_cfg(cfg, scenario)
    d, a, sip, sem, sfp, sbl, ssh, m, sc, pstart = gen_one_sequence2(fcfg, random_pos_anomaly=random_pos_anomaly)
    L = int(m.sum())
    cur = dict(identity) if identity is not None else {f: alloc.new() for f in _ENTITY_KEYS}
    flags = {"bca": sbl, "em": sem, "fp": sfp, "sa": ssh, "ip": sip}
    anomaly = _is_anomaly(sc)
    txns: List[dict] = []
    t = float(start_time)
    for i in range(L):
        if i > 0:
            t += float(d[i])
        for f in _ENTITY_KEYS:
            if flags[f][i] == 1.0 and f not in stable_fields:
                cur[f] = alloc.new()
        is_atk = anomaly and i >= pstart
        if share and is_atk:
            for f, val in share.items():
                cur[f] = val
        txns.append({
            "t": t,
            "amt": float(a[i]),
            "dt_self": float(d[i]),
            "bca": cur["bca"], "em": cur["em"], "fp": cur["fp"],
            "sa": cur["sa"], "ip": cur["ip"],
            "scenario": sc,
            "is_attack": is_atk,
        })
    return txns, sc, pstart


def _scenario_weights(cfg) -> List[Tuple[str, float]]:
    p = [
        ("ATO", getattr(cfg, "p_ato", 0.0)),
        ("Stuffing", getattr(cfg, "p_stuffing", 0.0)),
        ("Stuffing_then_ATO", getattr(cfg, "p_stuff_ato", 0.0)),
        ("Traveler", getattr(cfg, "p_traveler", 0.0)),
        ("Shopper", getattr(cfg, "p_shopper", 0.0)),
        ("Upgrader", getattr(cfg, "p_upgrader", 0.0)),
    ]
    p = [(k, float(w)) for k, w in p if w and w > 0.0]
    normal_w = max(0.0, 1.0 - sum(w for _, w in p))
    return [("normal", normal_w)] + p


def _pick_from_weights(pairs: List[Tuple[str, float]]) -> str:
    r = random.random()
    acc = 0.0
    for k, w in pairs:
        acc += w
        if r <= acc:
            return k
    return pairs[-1][0]


def _blend_attack_steps(a_txns: List[dict], cfg) -> None:
    t = a_txns[0]["t"]
    for i, tx in enumerate(a_txns):
        if i > 0:
            dt = (random.uniform(*cfg.dt_range_normal)
                  if tx["is_attack"] else tx["dt_self"])
            t += dt
            tx["dt_self"] = dt
            tx["t"] = t
        if tx["is_attack"]:
            tx["amt"] = float(np.random.lognormal(mean=cfg.amt_mean, sigma=cfg.amt_dv))


# ========================================================
# 更新：为 _emit_victim_cluster 显式挂载 is_training 参数
# ========================================================
def _emit_victim_cluster(cfg, alloc: "_EntityAllocator", txns: List[dict],
                         currents: List[dict], WINDOW: float, remaining: int,
                         random_pos_anomaly: bool = False) -> int:
    cold_frac = getattr(cfg, "anchor_cold_start_frac", 0.15)
    p_gang = getattr(cfg, "anchor_gang_frac", 0.4)
    gang_range = getattr(cfg, "anchor_gang_size_range", (2, 4))
    normal_sessions_range = getattr(cfg, "anchor_victim_history_range", (1, 3))
    hacker_reuse_frac = getattr(cfg, "hacker_reuse_frac", 0.5)
    hacker_fanout_range = getattr(cfg, "hacker_fanout_range", (2, 5))
    gang = random.random() < p_gang
    g = random.randint(*gang_range) if gang else 1
    g = max(1, min(g, remaining))
    drop_sa = alloc.new()
    drop_fp = alloc.new() if random.random() < 0.5 else None
    base_time = random.uniform(0.0, WINDOW * 0.5)
    regime_b = random.random() < hacker_reuse_frac
    hacker_em = None
    hacker_fp = None
    hacker_aged_until = base_time
    if regime_b:
        hacker_em = alloc.new()
        hacker_fp = alloc.new()
        cursor = base_time
        for _f in range(random.randint(*hacker_fanout_range)):
            fake_id = {"bca": alloc.new(), "em": hacker_em, "fp": hacker_fp,
                       "sa": drop_sa, "ip": alloc.new()}
            s_txns, _sc, _ps = _gen_session_txns(
                cfg, alloc, scenario="normal", start_time=cursor,
                identity=fake_id, stable_fields=frozenset({"em", "fp", "sa"}),
                random_pos_anomaly=random_pos_anomaly)
            txns.extend(s_txns)
            cursor = s_txns[-1]["t"] + random.uniform(60.0, WINDOW * 0.05)
        hacker_aged_until = cursor
    made = 0
    for _ in range(g):
        identity = {"bca": alloc.new(), "em": alloc.new(), "fp": alloc.new(),
                    "sa": alloc.new(), "ip": alloc.new()}
        cold = random.random() < cold_frac
        n_hist = 0 if cold else random.randint(*normal_sessions_range)
        cursor = base_time
        for _h in range(n_hist):
            s_txns, _sc, _ps = _gen_session_txns(
                cfg, alloc, scenario="normal", start_time=cursor, identity=dict(identity),
                random_pos_anomaly=random_pos_anomaly)
            txns.extend(s_txns)
            cursor = s_txns[-1]["t"] + random.uniform(3600.0, WINDOW * 0.1)
        share = {"bca": identity["bca"], "sa": drop_sa}
        if regime_b:
            share["em"] = hacker_em
            share["fp"] = hacker_fp
        elif drop_fp is not None:
            share["fp"] = drop_fp
        a_txns, _sc, _ps = _gen_session_txns(
            cfg, alloc, scenario="ATO", start_time=max(cursor, base_time, hacker_aged_until),
            identity=dict(identity), share=share, random_pos_anomaly=random_pos_anomaly)
        if regime_b:
            _blend_attack_steps(a_txns, cfg)
        txns.extend(a_txns)
        currents.append(a_txns[-1])
        made += 1
    return made


# ========================================================
# 更新：为 build_transaction_pool 显式挂载 is_training 参数
# ========================================================
def build_transaction_pool(cfg, n_currents: int, random_pos_anomaly: bool = False) -> Tuple[List[dict], List[dict]]:
    alloc = _EntityAllocator()
    WINDOW = cfg.WEEK
    weights = _scenario_weights(cfg)
    txns: List[dict] = []
    currents: List[dict] = []
    for _ in range(n_currents):
        pat = _pick_from_weights(weights)
        if pat == "ATO":
            _emit_victim_cluster(cfg, alloc, txns, currents, WINDOW, remaining=1, random_pos_anomaly=random_pos_anomaly)
        else:
            start = random.uniform(0.0, WINDOW * 0.95)
            s_txns, _sc, _ps = _gen_session_txns(cfg, alloc, scenario=pat, start_time=start, random_pos_anomaly=random_pos_anomaly)
            txns.extend(s_txns)
            currents.append(s_txns[-1])
    return txns, currents


def _index_pool(txns: List[dict]) -> Dict[str, Dict[int, List[dict]]]:
    idx: Dict[str, Dict[int, List[dict]]] = {f: {} for f in ANCHOR_FIELDS}
    for x in txns:
        for f in ANCHOR_FIELDS:
            idx[f].setdefault(x[f], []).append(x)
    for f in ANCHOR_FIELDS:
        for group in idx[f].values():
            group.sort(key=lambda z: z["t"])
    return idx


def _seq_for_anchor(idx: Dict[str, Dict[int, List[dict]]], anchor: str,
                    anchor_value: int, ct: float, cfg) -> List[dict]:
    T = cfg.max_seq_len
    window = getattr(cfg, "anchor_window_seconds", None)
    group = idx[anchor].get(anchor_value, [])
    seq = [x for x in group
           if x["t"] <= ct and (window is None or ct - x["t"] <= window)]
    return seq[-T:]


def _view_arrays(seq: List[dict], cfg) -> Tuple:
    T = cfg.max_seq_len
    dt = np.zeros(T, dtype=np.float32)
    amt = np.zeros(T, dtype=np.float32)
    sw = {name: np.zeros(T, dtype=np.float32) for name in _SW_TO_ENTITY}
    mask = np.zeros(T, dtype=np.float32)

    cum_dist = {name: np.zeros(T, dtype=np.float32) for name in _SW_TO_ENTITY}
    is_new = {name: np.zeros(T, dtype=np.float32) for name in _SW_TO_ENTITY}
    seen = {name: set() for name in _SW_TO_ENTITY}

    L = min(len(seq), T)
    pattern_start = 0
    seen_attack = False

    for k in range(L):
        x = seq[k]
        amt[k] = x["amt"]
        dt[k] = x["dt_self"] if k == 0 else max(0.0, seq[k]["t"] - seq[k - 1]["t"])
        mask[k] = 1.0

        for name, ent in _SW_TO_ENTITY.items():
            val = x[ent]
            if val not in seen[name]:
                is_new[name][k] = 1.0
                seen[name].add(val)
            cum_dist[name][k] = len(seen[name])

            if k > 0 and x[ent] != seq[k - 1][ent]:
                sw[name][k] = 1.0

        if x["is_attack"] and not seen_attack:
            pattern_start = k
            seen_attack = True

    return dt, amt, sw, mask, L, pattern_start, cum_dist, is_new


def _density_vec(seq: List[dict], anchor: str) -> List[float]:
    denom = float(max(1, len(seq)))
    return [len({x[f] for x in seq}) / denom
            for f in ANCHOR_FIELDS if f != anchor]


def _latest_different_value(seq: List[dict], field: str, current_val: int) -> Optional[int]:
    for txn in reversed(seq):
        if txn[field] != current_val:
            return txn[field]
    return None


CLEAN_NAMES = {
    "sw_ip": "ip",
    "sw_email": "email",
    "sw_fp": "fp",
    "sw_bill": "bill",
    "sw_ship": "ship"
}


def _emit(seq: List[dict], anchor: str, side: str, pair_id: int,
          scenario: str, cfg) -> dict:
    dt, amt, sw, mask, L, pstart, cum_dist, is_new = _view_arrays(seq, cfg)
    is_fraud = _is_anomaly(scenario) and any(t.get("is_attack", False) for t in seq)

    res = {
        "dt": dt, "amount": amt, "mask": mask,
        "anchor_type": ANCHOR_TYPE_ID[anchor],
        "pattern_start": pstart,
        "valid_len": L,
        "is_anchor_new": 1 if L <= 1 else 0,
        "density": _density_vec(seq, anchor),
        "side": SIDE_ID[side],
        "pair_id": pair_id,
        "scenario": scenario,
        "record_is_fraud": 1 if is_fraud else 0,
    }

    for sw_name, clean_name in CLEAN_NAMES.items():
        res[sw_name] = sw[sw_name]
        res[f"cum_dist_{clean_name}"] = cum_dist[sw_name]
        res[f"is_new_{clean_name}"] = is_new[sw_name]

    return res


def _emit_current(idx: Dict[str, Dict[int, List[dict]]], current: dict, cfg,
                  next_pair_id: int) -> Tuple[List[dict], int]:
    ct = current["t"]
    sc = current["scenario"]
    main_seqs: Dict[str, List[dict]] = {
        f: _seq_for_anchor(idx, f, current[f], ct, cfg) for f in ANCHOR_FIELDS
    }
    records: List[dict] = []
    pid = next_pair_id
    seen_contrast: Dict[str, set] = {f: set() for f in ANCHOR_FIELDS}
    for y_field in ANCHOR_FIELDS:
        y_seq = main_seqs[y_field]
        y_curr_val = current[y_field]
        for x_field in ANCHOR_FIELDS:
            if x_field == y_field:
                continue
            y_other = _latest_different_value(main_seqs[x_field], y_field, y_curr_val)
            if y_other is None:
                continue
            if y_other in seen_contrast[y_field]:
                continue
            contrast_seq = _seq_for_anchor(idx, y_field, y_other, ct, cfg)
            if not contrast_seq:
                continue
            seen_contrast[y_field].add(y_other)
            records.append(_emit(y_seq, y_field, "main", pid, sc, cfg))
            records.append(_emit(contrast_seq, y_field, "contrast", pid, sc, cfg))
            pid += 1
    return records, pid


# ========================================================
# 更新：将 is_training 向下透传至 build_transaction_pool
# ========================================================
def gen_anchored_dataset(cfg, num_samples: Optional[int] = None,
                         anomaly_position_random_occurs: bool = False,
                         num_trx: Optional[int] = None) -> Dict[str, Any]:
    if num_trx is not None:
        n_currents = max(1, int(num_trx))
    elif getattr(cfg, "num_trx", None):
        n_currents = max(1, int(cfg.num_trx))
    else:
        target = num_samples if num_samples is not None else cfg.num_seqs
        n_currents = max(1, target // NUM_ANCHOR_TYPES)

    txns, currents = build_transaction_pool(cfg, n_currents, random_pos_anomaly=anomaly_position_random_occurs)
    idx = _index_pool(txns)
    records: List[dict] = []
    pair_id = 0
    for trx_id, c in enumerate(currents):
        recs, pair_id = _emit_current(idx, c, cfg, pair_id)
        for r in recs:
            r["trx_id"] = trx_id
        records.extend(recs)

    seq_keys = ["dt", "amount", "sw_ip", "sw_email", "sw_fp", "sw_bill", "sw_ship", "mask"]
    for clean_name in CLEAN_NAMES.values():
        seq_keys.append(f"cum_dist_{clean_name}")
        seq_keys.append(f"is_new_{clean_name}")

    out: Dict[str, Any] = {
        k: np.array([r[k] for r in records], dtype=np.float32) for k in seq_keys
    }
    out["anchor_type"] = np.array([r["anchor_type"] for r in records], dtype=np.int64)
    out["pattern_start"] = np.array([r["pattern_start"] for r in records], dtype=np.int64)
    out["valid_len"] = np.array([r["valid_len"] for r in records], dtype=np.int64)
    out["is_anchor_new"] = np.array([r["is_anchor_new"] for r in records], dtype=np.int64)
    out["density"] = np.array([r["density"] for r in records], dtype=np.float32)
    out["record_pair_side"] = np.array([r["side"] for r in records], dtype=np.int64)
    out["record_pair_id"] = np.array([r["pair_id"] for r in records], dtype=np.int64)
    out["trx_id"] = np.array([r["trx_id"] for r in records], dtype=np.int64)
    out["record_scenario"] = [r["scenario"] for r in records]
    out["record_is_fraud"] = np.array([r["record_is_fraud"] for r in records], dtype=np.int64)
    out["num_trx"] = len(currents)
    out["trx_scenario"] = [c["scenario"] for c in currents]
    return out

def make_anchored_dataset(current_cfg: CFG2, num_samples: int = None, anomaly_position_random_occurs: bool = False,
                          num_trx: int = None) -> dict:
    raw = gen_anchored_dataset(
        current_cfg, num_samples=(num_samples if num_samples is not None else current_cfg.num_seqs),
        anomaly_position_random_occurs=anomaly_position_random_occurs, num_trx=num_trx,
    )
    out = {
        "dt": torch.tensor(raw["dt"], dtype=torch.float32),
        "amount": torch.tensor(raw["amount"], dtype=torch.float32),
        "sw_ip": torch.tensor(raw["sw_ip"], dtype=torch.float32),
        "sw_email": torch.tensor(raw["sw_email"], dtype=torch.float32),
        "sw_fp": torch.tensor(raw["sw_fp"], dtype=torch.float32),
        "sw_bill": torch.tensor(raw["sw_bill"], dtype=torch.float32),
        "sw_ship": torch.tensor(raw["sw_ship"], dtype=torch.float32),
        "mask": torch.tensor(raw["mask"], dtype=torch.float32),
        "anchor_type": torch.tensor(raw["anchor_type"], dtype=torch.long),
        "valid_len": torch.tensor(raw["valid_len"], dtype=torch.long),
        "is_anchor_new": torch.tensor(raw["is_anchor_new"], dtype=torch.long),
        "density": torch.tensor(raw["density"], dtype=torch.float32),
        "record_pair_side": torch.tensor(raw["record_pair_side"], dtype=torch.long),
        "record_pair_id": torch.tensor(raw["record_pair_id"], dtype=torch.long),
        "trx_id": torch.tensor(raw["trx_id"], dtype=torch.long),
        "record_scenario": raw["record_scenario"],
        "record_is_fraud": torch.tensor(raw["record_is_fraud"], dtype=torch.long),
        "num_trx": raw["num_trx"],
        "trx_scenario": raw["trx_scenario"],
    }

    for name in ["ip", "email", "fp", "bill", "ship"]:
        out[f"cum_dist_{name}"] = torch.tensor(raw[f"cum_dist_{name}"], dtype=torch.float32)
        out[f"is_new_{name}"] = torch.tensor(raw[f"is_new_{name}"], dtype=torch.float32)

    return out
