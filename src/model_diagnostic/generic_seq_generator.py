"""V3 realistic transaction and Sequence-on-Graph generator.

The generator creates a 30-day warm-up period followed by a 30-day sampling
period.  Every sampled transaction is projected independently through a
rolling 30-day lookup over the configured anchors.  Pattern event counts are
business controls; ``model_sequence_capacity`` is only a tensor/truncation
control and never changes how many transactions are generated.

Public compatibility entry points:

* ``gen_one_sequence2``
* ``build_transaction_pool``
* ``gen_anchored_dataset``
* ``make_anchored_dataset``

V3 uses four graph anchors: BCA, email, device fingerprint, and shipping
address.  IP and billing address remain event features, but are intentionally
not anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # raw NumPy generation remains usable without PyTorch
    torch = None  # type: ignore[assignment]

try:  # normal project import
    from model_diagnostic.cfg_base import CFG2
except ModuleNotFoundError:  # permits isolated review/testing of this file
    CFG2 = Any  # type: ignore[misc,assignment]


DAY = 86_400.0
ANCHOR_FIELDS = ("bca", "em", "fp", "ship")
ANCHOR_TYPE_ID: Dict[str, int] = {name: i for i, name in enumerate(ANCHOR_FIELDS)}
NUM_ANCHOR_TYPES = len(ANCHOR_FIELDS)
SIDE_ID: Dict[str, int] = {"main": 0, "contrast": 1}
ENTITY_FIELDS: Tuple[str, ...] = ("bca", "em", "fp", "bill", "ship", "ip")

SW_TO_ENTITY: Dict[str, str] = {
    "sw_ip": "ip",
    "sw_email": "em",
    "sw_fp": "fp",
    "sw_bill": "bill",
    "sw_ship": "ship",
}
CLEAN_NAMES: Dict[str, str] = {
    "sw_ip": "ip",
    "sw_email": "email",
    "sw_fp": "fp",
    "sw_bill": "bill",
    "sw_ship": "ship",
}

# V3 Auxiliary BCA Keys added to provide explicit visibility into card switching
# behaviors within device, email, and shipping anchored sequences, without breaking
# the legacy 16-dimensional tensor contract.
AUX_SW_TO_ENTITY: Dict[str, str] = {"sw_bca": "bca"}
AUX_CLEAN_NAMES: Dict[str, str] = {"sw_bca": "bca"}

ALL_SW_TO_ENTITY: Dict[str, str] = {**SW_TO_ENTITY, **AUX_SW_TO_ENTITY}
ALL_CLEAN_NAMES: Dict[str, str] = {**CLEAN_NAMES, **AUX_CLEAN_NAMES}

DEFAULT_PATTERN_TXN_COUNT_RANGES: Dict[str, Tuple[int, int]] = {
    "normal": (2, 3),
    "Shopper": (5, 15),
    "Upgrader": (4, 8),
    "Traveler": (3, 7),
    "Shared_Household": (2, 5),
    "Stuffing_low_and_slow": (2, 5),
    "Stuffing_rotating_proxy": (1, 3),
    "Stuffing_dumb_script": (3, 8),
    "Stuffing_card_tester": (1, 3),
    "ATO_classic": (4, 8),
    "ATO_blitz": (4, 9),
    "ATO_sneaky": (4, 7),
    "Stuffing_then_ATO": (5, 8),
}

DEFAULT_PATTERN_DT_RANGES: Dict[str, Tuple[float, float]] = {
    "normal": (DAY, 7 * DAY),
    "Shopper": (3_600.0, DAY),
    "Upgrader": (DAY, 3 * DAY),
    "Traveler": (3_600.0, 43_200.0),
    "Shared_Household": (300.0, 7_200.0),
    "Stuffing_low_and_slow": (2 * DAY, 7 * DAY),
    "Stuffing_rotating_proxy": (1.0, 15.0),
    "Stuffing_dumb_script": (1.0, 5.0),
    "Stuffing_card_tester": (10.0, 60.0),
    "ATO_classic": (30.0, 300.0),
    "ATO_blitz": (5.0, 30.0),
    "ATO_sneaky": (DAY, 3 * DAY),
    "Stuffing_then_ATO": (10.0, 300.0),
}

MALICIOUS_PATTERNS: FrozenSet[str] = frozenset({
    "Stuffing_low_and_slow",
    "Stuffing_rotating_proxy",
    "Stuffing_dumb_script",
    "Stuffing_card_tester",
    "ATO_classic",
    "ATO_blitz",
    "ATO_sneaky",
    "Stuffing_then_ATO",
})


def is_anomaly(scenario: str) -> bool:
    return scenario.startswith("ATO") or scenario.startswith("Stuffing")


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _range_cfg(cfg: Any, name: str, defaults: Mapping[str, Tuple[Any, Any]], pattern: str) -> Tuple[Any, Any]:
    configured = _cfg(cfg, name, defaults)
    value = configured.get(pattern, defaults[pattern])
    if len(value) != 2 or value[0] > value[1]:
        raise ValueError(f"{name}[{pattern!r}] must be an ordered pair")
    return value[0], value[1]


def _probability(value: Any) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass
class _Rng:
    py: random.Random
    np: np.random.Generator

    def chance(self, probability: float) -> bool:
        return self.py.random() < _probability(probability)

    def uniform(self, bounds: Sequence[float]) -> float:
        return self.py.uniform(float(bounds[0]), float(bounds[1]))

    def choice(self, values: Sequence[Any]) -> Any:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self.py.randrange(len(values))]

    def weighted_choice(self, weights: Mapping[str, float]) -> str:
        positive = [(name, max(0.0, float(weight))) for name, weight in weights.items()]
        total = sum(weight for _, weight in positive)
        if total <= 0.0:
            raise ValueError("weighted choice requires at least one positive weight")
        draw = self.py.random() * total
        cumulative = 0.0
        for name, weight in positive:
            cumulative += weight
            if draw <= cumulative:
                return name
        return positive[-1][0]

    def different_choice(self, values: Sequence[Any], current: Any) -> Any:
        candidates = [value for value in values if value != current]
        return self.choice(candidates) if candidates else current


def _make_rng(cfg: Any) -> _Rng:
    seed = _cfg(cfg, "generator_seed", _cfg(cfg, "seed", 2026))
    return _Rng(random.Random(int(seed)), np.random.default_rng(int(seed)))


class _EntityAllocator:
    def __init__(self, start: int = 1):
        self._next = int(start)

    def new(self) -> int:
        value = self._next
        self._next += 1
        return value


@dataclass
class _Actor:
    actor_id: int
    entities: Dict[str, int]
    amount_mean: float
    amount_std: float
    entity_pools: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    household_id: Optional[int] = None


def _new_actor(
    allocator: _EntityAllocator,
    actor_id: int,
    rng: _Rng,
    cfg: Any,
    shared: Optional[Mapping[str, int]] = None,
) -> _Actor:
    entities = {entity: allocator.new() for entity in ENTITY_FIELDS}
    if shared:
        unknown = set(shared) - set(ENTITY_FIELDS)
        if unknown:
            raise ValueError(f"unknown shared entities: {sorted(unknown)}")
        entities.update({name: int(value) for name, value in shared.items()})
    median = float(_cfg(cfg, "bca_amount_median", 100.0))
    log_sigma = float(_cfg(cfg, "bca_amount_log_sigma", 0.55))
    mean_bounds = _cfg(cfg, "bca_amount_mean_range", (20.0, 500.0))
    std_ratio_bounds = _cfg(cfg, "bca_amount_std_ratio_range", (0.20, 0.45))
    amount_mean = float(np.clip(rng.np.lognormal(np.log(median), log_sigma), *mean_bounds))
    amount_std = amount_mean * rng.uniform(std_ratio_bounds)
    return _Actor(
        actor_id=actor_id,
        entities=entities,
        amount_mean=amount_mean,
        amount_std=amount_std,
    )


def _share_entity(
    entity: str,
    value: int,
    *,
    actors: Sequence[_Actor] = (),
    transactions: Sequence[dict] = (),
) -> None:
    """Assign one explicit shared entity to actors and/or transactions."""
    if entity not in ENTITY_FIELDS:
        raise ValueError(f"unknown shared entity: {entity}")
    for actor in actors:
        actor.entities[entity] = int(value)
    for txn in transactions:
        txn[entity] = int(value)
        _refresh_alias(txn)


def _new_entity(actor: _Actor, allocator: _EntityAllocator, entity: str) -> int:
    value = allocator.new()
    actor.entities[entity] = value
    return value


def _positive_normal(rng: _Rng, mean: float, sigma: float, floor: float = 1.0) -> float:
    return round(max(float(floor), float(rng.np.normal(mean, sigma))), 2)


def _lognormal_amount(cfg: Any, rng: _Rng) -> float:
    amount = float(rng.np.lognormal(float(_cfg(cfg, "amt_mean", 3.0)), float(_cfg(cfg, "amt_dv", 1.2))))
    if rng.chance(float(_cfg(cfg, "amt_high_range_p", 0.03))):
        amount = rng.uniform(_cfg(cfg, "amt_high_range", (100.0, 3000.0)))
    return round(max(1.0, amount), 2)


def _benign_amount(pattern: str, actor: _Actor, cfg: Any, rng: _Rng) -> float:
    if rng.chance(float(_cfg(cfg, "benign_small_purchase_probability", 0.05))):
        return round(rng.uniform(_cfg(cfg, "benign_small_purchase_range", (1.0, 8.0))), 2)
    mean_multipliers = {
        "normal": 1.0,
        "Shopper": 1.50,
        "Upgrader": 1.25,
        "Traveler": 1.10,
        "Shared_Household": 0.80,
    }
    multiplier = mean_multipliers.get(pattern, 1.0)
    amount = _positive_normal(rng, actor.amount_mean * multiplier, actor.amount_std, 1.0)
    high_probabilities = _cfg(cfg, "benign_high_ticket_probabilities", {})
    if rng.chance(float(high_probabilities.get(pattern, 0.0))):
        amount *= rng.uniform(_cfg(cfg, "benign_high_ticket_multiplier_range", (2.0, 6.0)))
    return round(max(1.0, amount), 2)


def _amount_sampler(pattern: str, actor: _Actor, cfg: Any, rng: _Rng) -> Callable[[int, int], float]:
    """Return the one amount policy for a pattern; every policy is strictly positive."""
    if pattern in {"normal", "Shopper", "Upgrader", "Traveler", "Shared_Household"}:
        return lambda _i, _n: _benign_amount(pattern, actor, cfg, rng)
    if pattern == "Stuffing_card_tester":
        return lambda _i, _n: round(rng.uniform(_cfg(cfg, "card_testing_amount_range", (1.0, 5.0))), 2)
    if pattern.startswith("Stuffing"):
        micro_bounds = _cfg(cfg, "stuffing_amount_range", (1.0, 5.0))
        probabilities = _cfg(cfg, "stuffing_normal_amount_probabilities", {})
        normal_p = float(probabilities.get(pattern, _cfg(cfg, "stuffing_normal_amount_probability", 0.45)))
        return lambda _i, _n: (
            _positive_normal(rng, actor.amount_mean, actor.amount_std, 1.0)
            if rng.chance(normal_p)
            else round(rng.uniform(micro_bounds), 2)
        )
    if pattern.startswith("ATO_"):
        # Phase-specific attack amounts are applied together with attack labels
        # after the takeover point has been planned.
        return lambda _i, _n: _positive_normal(rng, actor.amount_mean, actor.amount_std, 1.0)
    return lambda _i, _n: _lognormal_amount(cfg, rng)


def _sample_count(pattern: str, cfg: Any, rng: _Rng) -> int:
    lo, hi = _range_cfg(cfg, "pattern_txn_count_ranges", DEFAULT_PATTERN_TXN_COUNT_RANGES, pattern)
    return rng.py.randint(int(lo), int(hi))


def _sample_timestamps(
    count: int,
    start: float,
    end: float,
    gap_bounds: Sequence[float],
    rng: _Rng,
) -> List[float]:
    """Schedule ordered absolute times while honoring the requested gap bounds."""
    if count <= 0:
        return []
    if end <= start:
        raise ValueError("timestamp interval must be positive")
    if count == 1:
        return [rng.py.uniform(start, np.nextafter(end, start))]

    lo, hi = float(gap_bounds[0]), float(gap_bounds[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError("gap bounds must be positive and ordered")
    usable = end - start
    if (count - 1) * lo >= usable:
        raise ValueError(f"{count} events with minimum gap {lo} do not fit in interval {usable}")

    max_total = usable * 0.96
    gaps: List[float] = []
    for _ in range(128):
        gaps = [rng.py.uniform(lo, hi) for _ in range(count - 1)]
        if sum(gaps) <= max_total:
            break
    else:
        gaps = [lo] * (count - 1)

    total = sum(gaps)
    first = rng.py.uniform(start, np.nextafter(end - total, start))
    timestamps = [first]
    for gap in gaps:
        timestamps.append(timestamps[-1] + gap)
    return timestamps


def _sample_timestamps_from_gap_plan(
    start: float, end: float, gap_plan: Sequence[Sequence[float]], rng: _Rng,
) -> List[float]:
    if not gap_plan:
        return [rng.py.uniform(start, np.nextafter(end, start))]
    usable = end - start
    minimum = sum(float(bounds[0]) for bounds in gap_plan)
    if minimum >= usable:
        raise ValueError("planned event gaps do not fit in the simulation interval")
    gaps: List[float] = []
    for _ in range(128):
        gaps = [rng.uniform(bounds) for bounds in gap_plan]
        if sum(gaps) <= usable * 0.96:
            break
    else:
        gaps = [float(bounds[0]) for bounds in gap_plan]
    total = sum(gaps)
    first = rng.py.uniform(start, np.nextafter(end - total, start))
    timestamps = [first]
    for gap in gaps:
        timestamps.append(timestamps[-1] + gap)
    return timestamps


def _pattern_timestamps(
    pattern: str, count: int, start: float, end: float, cfg: Any, rng: _Rng,
    takeover_index: Optional[int] = None,
) -> List[float]:
    pattern_bounds = _range_cfg(cfg, "pattern_dt_ranges", DEFAULT_PATTERN_DT_RANGES, pattern)
    if not pattern.startswith("ATO_") or count <= 1 or takeover_index is None:
        return _sample_timestamps(count, start, end, pattern_bounds, rng)

    normal_bounds = _range_cfg(cfg, "pattern_dt_ranges", DEFAULT_PATTERN_DT_RANGES, "normal")
    gap_plan: List[Sequence[float]] = []
    for target_index in range(1, count):
        if pattern == "ATO_sneaky":
            # The takeover itself follows natural pacing; dormancy begins after
            # the recovery/profile change.
            gap_plan.append(normal_bounds if target_index <= takeover_index else pattern_bounds)
        else:
            gap_plan.append(normal_bounds if target_index < takeover_index else pattern_bounds)
    return _sample_timestamps_from_gap_plan(start, end, gap_plan, rng)


def _make_transactions(
    actor: _Actor,
    pattern: str,
    timestamps: Sequence[float],
    cfg: Any,
    rng: _Rng,
) -> List[dict]:
    sample_amount = _amount_sampler(pattern, actor, cfg, rng)
    transactions: List[dict] = []
    for i, timestamp in enumerate(timestamps):
        txn = {
            "t": float(timestamp),
            "amt": float(sample_amount(i, len(timestamps))),
            "bca": actor.entities["bca"],
            "em": actor.entities["em"],
            "fp": actor.entities["fp"],
            "bill": actor.entities["bill"],
            "ship": actor.entities["ship"],
            "ip": actor.entities["ip"],
            "scenario": pattern,
            "phase": "baseline",
            "is_attack": False,
            "actor_id": actor.actor_id,
        }
        txn["sa"] = txn["ship"]  # legacy raw-transaction alias only
        transactions.append(txn)
    return transactions


def _refresh_alias(txn: dict) -> None:
    txn["sa"] = txn["ship"]


def _replace_on_txn(txn: dict, actor: _Actor, allocator: _EntityAllocator, entity: str) -> None:
    txn[entity] = _new_entity(actor, allocator, entity)
    _refresh_alias(txn)


def _carry_forward(transactions: List[dict], start_index: int, entity: str, value: int) -> None:
    for txn in transactions[start_index:]:
        txn[entity] = int(value)
        _refresh_alias(txn)


def _mark_attack(transactions: List[dict], start_index: int, phase: str = "attack") -> None:
    for txn in transactions[start_index:]:
        txn["is_attack"] = True
        txn["phase"] = phase


@dataclass
class _Campaign:
    scenario: str
    campaign_id: int
    victim_capacity: int
    ips: Tuple[int, ...]
    fps: Tuple[int, ...]
    bills: Tuple[int, ...]
    ships: Tuple[int, ...]
    emails: Tuple[int, ...]
    victims: int = 0


@dataclass
class _CampaignManager:
    allocator: _EntityAllocator
    rng: _Rng
    cfg: Any
    active: Dict[str, _Campaign] = field(default_factory=dict)
    next_id: int = 1

    def acquire(self, scenario: str) -> _Campaign:
        campaign = self.active.get(scenario)
        if campaign is None or campaign.victims >= campaign.victim_capacity:
            campaign = self._create(scenario)
            self.active[scenario] = campaign
        campaign.victims += 1
        return campaign

    def _values(self, count: int) -> Tuple[int, ...]:
        return tuple(self.allocator.new() for _ in range(max(0, count)))

    def _create(self, scenario: str) -> _Campaign:
        victim_ranges = _cfg(self.cfg, "campaign_victim_count_ranges", {
            "Stuffing_low_and_slow": (3, 10),
            "Stuffing_rotating_proxy": (5, 20),
            "Stuffing_dumb_script": (2, 6),
            "Stuffing_card_tester": (3, 12),
            "ATO_classic": (2, 6),
            "ATO_blitz": (2, 5),
            "ATO_sneaky": (1, 4),
            "Stuffing_then_ATO": (2, 6),
        })
        lo, hi = victim_ranges.get(scenario, (1, 1))
        capacity = self.rng.py.randint(int(lo), int(hi))
        if scenario == "Stuffing_low_and_slow":
            ip_count = self.rng.py.randint(2, 4)
            fp_count = self.rng.py.randint(1, 3)
        elif scenario in {"Stuffing_dumb_script", "Stuffing_card_tester"}:
            ip_count = 1
            fp_count = 1
        elif scenario.startswith("ATO") or scenario == "Stuffing_then_ATO":
            ip_count = 1
            fp_count = 1
        else:  # rotating proxy allocates a fresh IP and FP for every event
            ip_count = 0
            fp_count = 0
        bill_count = 1 if scenario in {"Stuffing_card_tester", "Stuffing_then_ATO"} or scenario.startswith("ATO") else 0
        ship_count = 1 if scenario.startswith("ATO") or scenario in {"Stuffing_card_tester", "Stuffing_then_ATO"} else 0
        email_count = 1 if scenario in {"Stuffing_card_tester", "Stuffing_then_ATO"} or scenario.startswith("ATO") else 0
        campaign = _Campaign(
            scenario=scenario,
            campaign_id=self.next_id,
            victim_capacity=capacity,
            ips=self._values(ip_count),
            fps=self._values(fp_count),
            bills=self._values(bill_count),
            ships=self._values(ship_count),
            emails=self._values(email_count),
        )
        self.next_id += 1
        return campaign


def _apply_normal(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    _campaign: Optional[_Campaign], cfg: Any,
) -> None:
    for i in range(1, len(txns)):
        if rng.chance(float(_cfg(cfg, "normal_ip_change_probability", 0.05))):
            value = _new_entity(actor, allocator, "ip")
            _carry_forward(txns, i, "ip", value)


def _apply_shopper(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    _campaign: Optional[_Campaign], cfg: Any,
) -> None:
    for i in range(1, len(txns)):
        if rng.chance(float(_cfg(cfg, "shopper_ip_change_probability", 0.15))):
            value = _new_entity(actor, allocator, "ip")
            _carry_forward(txns, i, "ip", value)


def _apply_upgrader(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    _campaign: Optional[_Campaign], cfg: Any,
) -> None:
    index = rng.py.randint(1, len(txns) - 1)
    subtype = rng.weighted_choice(_cfg(cfg, "upgrader_subtype_probabilities", {
        "device_upgrade": 0.50,
        "shipping_change": 0.30,
        "home_move": 0.20,
    }))
    entities = {
        "device_upgrade": ("fp",),
        "shipping_change": ("ship",),
        "home_move": ("bill", "ship"),
    }[subtype]
    for entity in entities:
        value = _new_entity(actor, allocator, entity)
        _carry_forward(txns, index, entity, value)
    txns[index]["phase"] = subtype


def _apply_traveler(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    _campaign: Optional[_Campaign], cfg: Any,
) -> None:
    count_bounds = _cfg(cfg, "traveler_ip_count_range", (2, 4))
    distinct_count = rng.py.randint(int(count_bounds[0]), min(int(count_bounds[1]), len(txns)))
    change_points = sorted(rng.py.sample(range(1, len(txns)), distinct_count - 1))
    for index in change_points:
        value = _new_entity(actor, allocator, "ip")
        _carry_forward(txns, index, "ip", value)
        txns[index]["phase"] = "travel"


def _apply_shared_household(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    _campaign: Optional[_Campaign], cfg: Any,
) -> None:
    ip_pool = actor.entity_pools.get("ip", (actor.entities["ip"],))
    current_ip = txns[0]["ip"]
    for i in range(1, len(txns)):
        if rng.chance(float(_cfg(cfg, "household_ip_change_probability", 0.35))):
            current_ip = rng.different_choice(ip_pool, current_ip)
            _carry_forward(txns, i, "ip", current_ip)
    for txn in txns:
        txn["phase"] = "shared_household"


def _apply_stuffing(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    campaign: Optional[_Campaign], cfg: Any,
) -> None:
    if campaign is None:
        raise ValueError("stuffing pattern requires a campaign")
    _mark_attack(txns, 0, "credential_attack")
    if campaign.scenario == "Stuffing_dumb_script":
        _share_entity("ip", campaign.ips[0], transactions=txns)
        _share_entity("fp", campaign.fps[0], transactions=txns)
    elif campaign.scenario == "Stuffing_card_tester":
        _share_entity("ip", campaign.ips[0], transactions=txns)
        _share_entity("fp", campaign.fps[0], transactions=txns)
        _share_entity("em", campaign.emails[0], transactions=txns)
        _share_entity("ship", campaign.ships[0], transactions=txns)
        if not rng.chance(float(_cfg(cfg, "card_tester_victim_bill_probability", 0.85))):
            _share_entity("bill", campaign.bills[0], transactions=txns)
    elif campaign.scenario == "Stuffing_low_and_slow":
        current_ip = rng.choice(campaign.ips)
        current_fp = rng.choice(campaign.fps)
        for i, txn in enumerate(txns):
            if i > 0 and rng.chance(float(_cfg(cfg, "low_slow_ip_rotation_probability", 0.65))):
                current_ip = rng.different_choice(campaign.ips, current_ip)
            if i > 0 and rng.chance(float(_cfg(cfg, "low_slow_fp_rotation_probability", 0.35))):
                current_fp = rng.different_choice(campaign.fps, current_fp)
            txn["ip"] = current_ip
            txn["fp"] = current_fp
    else:  # rotating proxy: every event receives fresh infrastructure
        for txn in txns:
            txn["ip"] = allocator.new()
            txn["fp"] = allocator.new()
    for txn in txns:
        txn["campaign_id"] = campaign.campaign_id
        _refresh_alias(txn)


def _takeover_index(pattern: str, length: int, random_position: bool, rng: _Rng) -> int:
    if pattern == "ATO_sneaky":
        if length < 4:
            raise ValueError("ATO_sneaky requires at least four events")
        return length - 2  # benign prefix, takeover, one dormant gap, payout
    if pattern == "ATO_blitz":
        feasible = [i for i in range(2, length) if i <= 4 and 2 <= length - i <= 5]
        preferred = 2
    else:
        feasible = [i for i in range(2, length) if i <= 5 and 1 <= length - i <= 3]
        preferred = max(2, length // 2)
    if not feasible:
        raise ValueError(f"{pattern} length {length} cannot satisfy phase-count constraints")
    if random_position:
        return int(rng.choice(feasible))
    return min(feasible, key=lambda index: abs(index - preferred))


def _relative_amount(actor: _Actor, bounds: Sequence[float], rng: _Rng) -> float:
    return round(max(1.0, actor.amount_mean * rng.uniform(bounds)), 2)


def _apply_ato(
    txns: List[dict],
    actor: _Actor,
    allocator: _EntityAllocator,
    rng: _Rng,
    campaign: Optional[_Campaign],
    takeover_index: int,
    cfg: Any,
) -> None:
    if campaign is None:
        raise ValueError("ATO pattern requires a campaign")
    start = int(takeover_index)
    _mark_attack(txns, start, "takeover")
    probabilities = _cfg(cfg, "ato_entity_change_probabilities", {}).get(campaign.scenario, {})
    campaign_values = {
        "fp": campaign.fps[0],
        "ip": campaign.ips[0],
        "em": campaign.emails[0],
        "bill": campaign.bills[0],
        "ship": campaign.ships[0],
    }
    for entity in ("fp", "ip", "em", "bill"):
        if rng.chance(float(probabilities.get(entity, 0.0))):
            _share_entity(entity, campaign_values[entity], transactions=txns[start:])
    if campaign.scenario != "ATO_sneaky" and rng.chance(float(probabilities.get("ship", 0.0))):
        _share_entity("ship", campaign_values["ship"], transactions=txns[start:])
    for i in range(start, len(txns)):
        txns[i]["campaign_id"] = campaign.campaign_id
        _refresh_alias(txns[i])
    if campaign.scenario == "ATO_sneaky":
        txns[start]["phase"] = "takeover_change"
        txns[-1]["phase"] = "monetization"
        txns[start]["amt"] = _relative_amount(
            actor, _cfg(cfg, "ato_sneaky_takeover_multiplier_range", (0.7, 1.5)), rng,
        )
        txns[-1]["amt"] = _relative_amount(
            actor, _cfg(cfg, "ato_sneaky_final_multiplier_range", (2.5, 7.0)), rng,
        )
        if rng.chance(float(probabilities.get("ship", 0.0))):
            _share_entity("ship", campaign_values["ship"], transactions=txns[-1:])
    elif campaign.scenario == "ATO_classic":
        for txn in txns[start:]:
            txn["phase"] = "monetization" if txn is not txns[start] else "takeover"
            draw = rng.py.random()
            near_p = float(_cfg(cfg, "ato_classic_near_normal_probability", 0.20))
            high_p = float(_cfg(cfg, "ato_classic_very_high_probability", 0.20))
            if draw < near_p:
                bounds = _cfg(cfg, "ato_classic_near_normal_multiplier_range", (0.8, 1.5))
            elif draw < near_p + high_p:
                bounds = _cfg(cfg, "ato_classic_very_high_multiplier_range", (5.0, 10.0))
            else:
                bounds = _cfg(cfg, "ato_classic_elevated_multiplier_range", (2.0, 5.0))
            txn["amt"] = _relative_amount(actor, bounds, rng)
    elif campaign.scenario == "ATO_blitz":
        for attack_offset, txn in enumerate(txns[start:]):
            txn["phase"] = "fraud_sweep"
            bounds = (
                _cfg(cfg, "ato_blitz_first_multiplier_range", (0.8, 2.0))
                if attack_offset == 0
                else _cfg(cfg, "ato_blitz_later_multiplier_range", (3.0, 10.0))
            )
            txn["amt"] = _relative_amount(actor, bounds, rng)


def _apply_stuffing_then_ato(
    txns: List[dict], actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    campaign: Optional[_Campaign], random_position: bool,
) -> None:
    if campaign is None:
        raise ValueError("combined pattern requires a campaign")
    pivot = max(1, len(txns) - 2)
    _mark_attack(txns, 0, "credential_attack")
    for i, txn in enumerate(txns):
        txn["fp"] = campaign.fps[0]
        txn["ip"] = allocator.new() if i < pivot else txn["ip"]
        txn["campaign_id"] = campaign.campaign_id
        if i >= pivot:
            txn["phase"] = "takeover"
            txn["amt"] = round(rng.uniform((300.0, 1500.0)), 2)
            if campaign.ships:
                txn["ship"] = campaign.ships[0]
            if campaign.emails:
                txn["em"] = campaign.emails[0]
        _refresh_alias(txn)


def _apply_pattern(
    txns: List[dict], pattern: str, actor: _Actor, allocator: _EntityAllocator,
    rng: _Rng, campaign: Optional[_Campaign], random_position: bool, cfg: Any,
    takeover_index: Optional[int] = None,
) -> None:
    benign_handlers: Dict[str, Callable[..., None]] = {
        "normal": _apply_normal,
        "Shopper": _apply_shopper,
        "Upgrader": _apply_upgrader,
        "Traveler": _apply_traveler,
        "Shared_Household": _apply_shared_household,
    }
    if pattern in benign_handlers:
        benign_handlers[pattern](txns, actor, allocator, rng, campaign, cfg)
    elif pattern.startswith("Stuffing_") and pattern != "Stuffing_then_ATO":
        _apply_stuffing(txns, actor, allocator, rng, campaign, cfg)
    elif pattern.startswith("ATO_"):
        if takeover_index is None:
            raise ValueError("ATO pattern requires a planned takeover index")
        _apply_ato(txns, actor, allocator, rng, campaign, takeover_index, cfg)
    elif pattern == "Stuffing_then_ATO":
        _apply_stuffing_then_ato(txns, actor, allocator, rng, campaign, random_position)
    else:
        raise ValueError(f"unsupported pattern: {pattern}")


def _generate_period(
    cfg: Any,
    actor: _Actor,
    allocator: _EntityAllocator,
    rng: _Rng,
    pattern: str,
    start: float,
    end: float,
    campaign: Optional[_Campaign] = None,
    count: Optional[int] = None,
    random_position: bool = False,
) -> List[dict]:
    event_count = _sample_count(pattern, cfg, rng) if count is None else int(count)
    takeover_index = (
        _takeover_index(pattern, event_count, random_position, rng)
        if pattern.startswith("ATO_") else None
    )
    timestamps = _pattern_timestamps(pattern, event_count, start, end, cfg, rng, takeover_index)
    txns = _make_transactions(actor, pattern, timestamps, cfg, rng)
    _apply_pattern(txns, pattern, actor, allocator, rng, campaign, random_position, cfg, takeover_index)
    return txns


def _scenario_weights(cfg: Any) -> List[Tuple[str, float]]:
    explicit = _cfg(cfg, "pattern_weights", None)
    if explicit:
        pairs = [(str(name), max(0.0, float(weight))) for name, weight in explicit.items()]
    else:
        p_ato = max(0.0, float(_cfg(cfg, "p_ato", 0.05))) / 3.0
        p_stuffing = max(0.0, float(_cfg(cfg, "p_stuffing", 0.05))) / 4.0
        pairs = [
            ("ATO_classic", p_ato), ("ATO_blitz", p_ato), ("ATO_sneaky", p_ato),
            ("Stuffing_low_and_slow", p_stuffing),
            ("Stuffing_rotating_proxy", p_stuffing),
            ("Stuffing_dumb_script", p_stuffing),
            ("Stuffing_card_tester", p_stuffing),
            ("Stuffing_then_ATO", max(0.0, float(_cfg(cfg, "p_stuff_ato", 0.03)))),
            ("Traveler", max(0.0, float(_cfg(cfg, "p_traveler", 0.05)))),
            ("Shopper", max(0.0, float(_cfg(cfg, "p_shopper", 0.07)))),
            ("Upgrader", max(0.0, float(_cfg(cfg, "p_upgrader", 0.07)))),
            ("Shared_Household", max(0.0, float(_cfg(cfg, "p_shared_household", 0.05)))),
        ]
    total = sum(weight for _, weight in pairs)
    intended_actor_weights = (
        [(name, weight / total) for name, weight in pairs if weight > 0.0]
        if total > 1.0
        else [("normal", 1.0 - total)] + [(name, weight) for name, weight in pairs if weight > 0.0]
    )

    # One Shared_Household draw creates several BCA actors. Convert intended
    # BCA-level prevalence into generation-unit weights so the cluster does not
    # silently multiply its population share.
    household_bounds = _cfg(cfg, "household_bca_count_range", (2, 4))
    expected_household_size = (float(household_bounds[0]) + float(household_bounds[1])) / 2.0
    unit_weights = [
        (name, weight / expected_household_size if name == "Shared_Household" else weight)
        for name, weight in intended_actor_weights
    ]
    unit_total = sum(weight for _, weight in unit_weights)
    return [(name, weight / unit_total) for name, weight in unit_weights if weight > 0.0]


def _pick_scenario(weights: Sequence[Tuple[str, float]], rng: _Rng) -> str:
    value = rng.py.random()
    cumulative = 0.0
    for name, weight in weights:
        cumulative += weight
        if value <= cumulative:
            return name
    return weights[-1][0]


@dataclass
class _HouseholdFactory:
    allocator: _EntityAllocator
    rng: _Rng
    cfg: Any
    next_id: int = 1

    def create(self, next_actor_id: int) -> List[_Actor]:
        bounds = _cfg(self.cfg, "household_bca_count_range", (2, 4))
        count = self.rng.py.randint(int(bounds[0]), int(bounds[1]))
        actors = [
            _new_actor(self.allocator, next_actor_id + i, self.rng, self.cfg)
            for i in range(count)
        ]
        probabilities = _cfg(self.cfg, "household_shared_probabilities", {
            "fp": 0.70, "ship": 0.90, "bill": 0.60, "em": 0.10,
        })
        for entity in ("fp", "ship", "bill", "em"):
            if self.rng.chance(float(probabilities.get(entity, 0.0))):
                _share_entity(entity, self.allocator.new(), actors=actors)
        pool_bounds = _cfg(self.cfg, "household_ip_pool_size_range", (1, 3))
        ip_pool = tuple(
            self.allocator.new()
            for _ in range(self.rng.py.randint(int(pool_bounds[0]), int(pool_bounds[1])))
        )
        for actor in actors:
            actor.household_id = self.next_id
            actor.entity_pools["ip"] = ip_pool
            _share_entity("ip", self.rng.choice(ip_pool), actors=(actor,))
        self.next_id += 1
        return actors


def _warmup_count(cfg: Any, rng: _Rng) -> int:
    lo, hi = _cfg(cfg, "warmup_txn_count_range", (2, 5))
    return rng.py.randint(int(lo), int(hi))


def _simulation_bounds(cfg: Any) -> Tuple[float, float, float]:
    warmup = float(_cfg(cfg, "simulation_warmup_days", 30)) * DAY
    sampling = float(_cfg(cfg, "current_sampling_days", 30)) * DAY
    if warmup <= 0 or sampling <= 0:
        raise ValueError("warm-up and sampling durations must be positive")
    window = float(_cfg(cfg, "anchor_window_seconds", 30 * DAY))
    if warmup < window:
        raise ValueError("simulation warm-up must be at least anchor_window_seconds")
    return 0.0, warmup, warmup + sampling


def _generate_actor_timeline(
    cfg: Any, actor: _Actor, allocator: _EntityAllocator, rng: _Rng,
    pattern: str, bounds: Tuple[float, float, float], campaigns: _CampaignManager,
    random_position: bool,
) -> Tuple[List[dict], List[dict]]:
    origin, training_start, end = bounds
    warmup = _generate_period(
        cfg, actor, allocator, rng, "normal", origin, training_start,
        count=_warmup_count(cfg, rng),
    )
    campaign = campaigns.acquire(pattern) if pattern in MALICIOUS_PATTERNS else None
    sampling = _generate_period(
        cfg, actor, allocator, rng, pattern, training_start, end,
        campaign=campaign, random_position=random_position,
    )
    return warmup + sampling, sampling


def _assign_transaction_ids(transactions: List[dict]) -> None:
    transactions.sort(key=lambda txn: (txn["t"], txn["actor_id"]))
    for transaction_id, txn in enumerate(transactions):
        txn["trx_uid"] = transaction_id


def build_transaction_pool(
    cfg: Any,
    n_currents: int,
    random_pos_anomaly: bool = False,
    rng: Optional[_Rng] = None,
) -> Tuple[List[dict], List[dict]]:
    """Build raw events and choose current transactions after a full warm-up."""
    if int(n_currents) <= 0:
        return [], []
    rng = rng or _make_rng(cfg)
    allocator = _EntityAllocator()
    campaigns = _CampaignManager(allocator, rng, cfg)
    households = _HouseholdFactory(allocator, rng, cfg)
    weights = _scenario_weights(cfg)
    bounds = _simulation_bounds(cfg)
    transactions: List[dict] = []
    candidates: List[dict] = []
    next_actor_id = 1

    while len(candidates) < int(n_currents):
        pattern = _pick_scenario(weights, rng)
        actors = (
            households.create(next_actor_id)
            if pattern == "Shared_Household"
            else [_new_actor(allocator, next_actor_id, rng, cfg)]
        )
        next_actor_id += len(actors)
        for actor in actors:
            actor_txns, actor_candidates = _generate_actor_timeline(
                cfg, actor, allocator, rng, pattern, bounds, campaigns, random_pos_anomaly,
            )
            household_id = actor.household_id
            if household_id is not None:
                for txn in actor_txns:
                    txn["household_id"] = household_id
            transactions.extend(actor_txns)
            candidates.extend(actor_candidates)

    _assign_transaction_ids(transactions)
    earliest = min(txn["t"] for txn in transactions)
    window = float(_cfg(cfg, "anchor_window_seconds", 30 * DAY))
    eligible = [txn for txn in candidates if txn["t"] >= earliest + window]
    extension_count = 0
    extension_limit = max(1000, int(n_currents) * 10)
    while len(eligible) < int(n_currents):
        # This is only reachable for unusual config combinations; extend safely
        # instead of returning currents without a complete history window.
        actor = _new_actor(allocator, next_actor_id, rng, cfg)
        next_actor_id += 1
        actor_txns, actor_candidates = _generate_actor_timeline(
            cfg, actor, allocator, rng, "normal", bounds, campaigns, random_pos_anomaly,
        )
        transactions.extend(actor_txns)
        candidates.extend(actor_candidates)
        _assign_transaction_ids(transactions)
        earliest = min(txn["t"] for txn in transactions)
        eligible = [txn for txn in candidates if txn["t"] >= earliest + window]
        extension_count += 1
        if extension_count >= extension_limit:
            raise RuntimeError("unable to create enough post-warmup currents; check simulation durations")

    rng.py.shuffle(eligible)
    currents = eligible[: int(n_currents)]
    currents.sort(key=lambda txn: txn["t"])
    return transactions, currents


def _index_pool(transactions: Iterable[dict]) -> Dict[str, Dict[int, List[dict]]]:
    index: Dict[str, Dict[int, List[dict]]] = {field: {} for field in ANCHOR_FIELDS}
    for txn in transactions:
        for field_name in ANCHOR_FIELDS:
            index[field_name].setdefault(int(txn[field_name]), []).append(txn)
    for groups in index.values():
        for group in groups.values():
            group.sort(key=lambda txn: (txn["t"], txn["trx_uid"]))
    return index


@dataclass(frozen=True)
class _AnchorView:
    window_transactions: Tuple[dict, ...]
    visible_transactions: Tuple[dict, ...]


def _anchor_view(
    index: Dict[str, Dict[int, List[dict]]], anchor: str, anchor_value: int,
    current_time: float, cfg: Any, current_uid: Optional[int] = None,
) -> _AnchorView:
    window = float(_cfg(cfg, "anchor_window_seconds", 30 * DAY))
    capacity = int(_cfg(cfg, "model_sequence_capacity", _cfg(cfg, "max_seq_len", 35)))
    if capacity <= 0:
        raise ValueError("model_sequence_capacity must be positive")
    lower = current_time - window
    def is_visible(txn: dict) -> bool:
        timestamp = float(txn["t"])
        if not lower <= timestamp <= current_time:
            return False
        return timestamp < current_time or current_uid is None or int(txn["trx_uid"]) <= int(current_uid)

    full = tuple(txn for txn in index[anchor].get(int(anchor_value), []) if is_visible(txn))
    return _AnchorView(full, full[-capacity:])


def _feature_rows(view: _AnchorView) -> List[dict]:
    """Compute features on the full rolling window, then let callers truncate."""
    seen = {entity: set() for entity in ALL_SW_TO_ENTITY.values()}
    rows: List[dict] = []
    previous: Optional[dict] = None
    for txn in view.window_transactions:
        row: Dict[str, Any] = {"txn": txn, "dt": 0.0 if previous is None else max(0.0, txn["t"] - previous["t"])}
        for switch_name, entity in ALL_SW_TO_ENTITY.items():
            value = txn[entity]
            row[switch_name] = float(previous is not None and value != previous[entity])
            clean_name = ALL_CLEAN_NAMES[switch_name]
            row[f"is_new_{clean_name}"] = float(value not in seen[entity])
            seen[entity].add(value)
            row[f"cum_dist_{clean_name}"] = float(len(seen[entity]))
        rows.append(row)
        previous = txn
    visible_count = len(view.visible_transactions)
    visible = rows[-visible_count:] if visible_count else []
    if visible:
        visible[0]["dt"] = 0.0
    return visible


def _view_arrays(view: _AnchorView, cfg: Any) -> Tuple[Any, ...]:
    capacity = int(_cfg(cfg, "model_sequence_capacity", _cfg(cfg, "max_seq_len", 35)))
    rows = _feature_rows(view)
    dt = np.zeros(capacity, dtype=np.float32)
    amount = np.zeros(capacity, dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.float32)
    step_label = np.zeros(capacity, dtype=np.float32)
    switches = {name: np.zeros(capacity, dtype=np.float32) for name in ALL_SW_TO_ENTITY}
    cumulative = {name: np.zeros(capacity, dtype=np.float32) for name in ALL_CLEAN_NAMES.values()}
    is_new = {name: np.zeros(capacity, dtype=np.float32) for name in ALL_CLEAN_NAMES.values()}
    pattern_start = 0
    found_attack = False
    for i, row in enumerate(rows):
        txn = row["txn"]
        dt[i] = float(row["dt"])
        amount[i] = float(txn["amt"])
        mask[i] = 1.0
        step_label[i] = float(bool(txn["is_attack"]))
        for switch_name, clean_name in ALL_CLEAN_NAMES.items():
            switches[switch_name][i] = row[switch_name]
            cumulative[clean_name][i] = row[f"cum_dist_{clean_name}"]
            is_new[clean_name][i] = row[f"is_new_{clean_name}"]
        if txn["is_attack"] and not found_attack:
            pattern_start = i
            found_attack = True
    return dt, amount, switches, mask, step_label, len(rows), pattern_start, cumulative, is_new


def _density_vector(view: _AnchorView, anchor: str) -> List[float]:
    denominator = float(max(1, len(view.window_transactions)))
    return [
        len({txn[field_name] for txn in view.window_transactions}) / denominator
        for field_name in ANCHOR_FIELDS if field_name != anchor
    ]


def _derive_record_scenario(view: _AnchorView, current: dict) -> str:
    """Return the latest anomaly scenario actually visible in this record."""
    for txn in reversed(view.visible_transactions):
        if bool(txn["is_attack"]):
            return str(txn["scenario"])
    current_scenario = str(current["scenario"])
    return "normal" if is_anomaly(current_scenario) else current_scenario


def _emit_record(
    view: _AnchorView, anchor: str, side: str, pair_id: int,
    current: dict, cfg: Any,
) -> dict:
    dt, amount, switches, mask, step_label, length, pattern_start, cumulative, is_new = _view_arrays(view, cfg)
    record_has_anomaly = any(bool(txn["is_attack"]) for txn in view.visible_transactions)
    result = {
        "dt": dt,
        "amount": amount,
        "mask": mask,
        "step_label": step_label,
        "anchor_type": ANCHOR_TYPE_ID[anchor],
        "pattern_start": pattern_start,
        "valid_len": length,
        "is_anchor_new": int(len(view.window_transactions) == 1),
        "density": _density_vector(view, anchor),
        "side": SIDE_ID[side],
        "pair_id": pair_id,
        "scenario": _derive_record_scenario(view, current),
        "current_label": int(bool(current["is_attack"])),
        "history_has_fraud": int(any(txn["is_attack"] for txn in view.window_transactions)),
        "record_is_fraud": int(record_has_anomaly),
    }
    for switch_name, clean_name in ALL_CLEAN_NAMES.items():
        result[switch_name] = switches[switch_name]
        result[f"cum_dist_{clean_name}"] = cumulative[clean_name]
        result[f"is_new_{clean_name}"] = is_new[clean_name]
    return result


def _emit_current(
    index: Dict[str, Dict[int, List[dict]]], current: dict, cfg: Any, next_pair_id: int,
) -> Tuple[List[dict], int]:
    current_time = float(current["t"])
    current_uid = int(current["trx_uid"])
    main_views = {
        field_name: _anchor_view(index, field_name, current[field_name], current_time, cfg, current_uid)
        for field_name in ANCHOR_FIELDS
    }
    records: List[dict] = []
    for anchor in ANCHOR_FIELDS:
        records.append(_emit_record(main_views[anchor], anchor, "main", -1, current, cfg))
    return records, int(next_pair_id)


def _sequence_array(records: Sequence[dict], key: str, capacity: int) -> np.ndarray:
    if not records:
        return np.empty((0, capacity), dtype=np.float32)
    return np.asarray([record[key] for record in records], dtype=np.float32)

def gen_anchored_dataset(
    cfg: Any,
    num_samples: Optional[int] = None,
    anomaly_position_random_occurs: bool = False,
    num_trx: Optional[int] = None,
) -> Dict[str, Any]:

    n_currents = int(num_trx) if num_trx is not None else 5000
    rng = _make_rng(cfg)
    transactions, currents = build_transaction_pool(cfg, n_currents, anomaly_position_random_occurs, rng)
    index = _index_pool(transactions)
    records: List[dict] = []
    transaction_scenarios: List[str] = []
    pair_id = 0
    for current_id, current in enumerate(currents):
        emitted, pair_id = _emit_current(index, current, cfg, pair_id)
        for record in emitted:
            record["trx_id"] = current_id
        records.extend(emitted)
        anomaly_scenarios = [record["scenario"] for record in emitted if record["record_is_fraud"]]
        if anomaly_scenarios:
            transaction_scenarios.append(str(anomaly_scenarios[-1]))
        else:
            current_scenario = str(current["scenario"])
            transaction_scenarios.append("normal" if is_anomaly(current_scenario) else current_scenario)

    capacity = int(_cfg(cfg, "model_sequence_capacity", _cfg(cfg, "max_seq_len", 35)))
    sequence_keys = ["dt", "amount", "sw_ip", "sw_email", "sw_fp", "sw_bill", "sw_ship", "mask", "step_label", "sw_bca"]
    for clean_name in ALL_CLEAN_NAMES.values():
        sequence_keys.extend((f"cum_dist_{clean_name}", f"is_new_{clean_name}"))
    output: Dict[str, Any] = {key: _sequence_array(records, key, capacity) for key in sequence_keys}
    output.update({
        "anchor_type": np.asarray([r["anchor_type"] for r in records], dtype=np.int64),
        "pattern_start": np.asarray([r["pattern_start"] for r in records], dtype=np.int64),
        "valid_len": np.asarray([r["valid_len"] for r in records], dtype=np.int64),
        "is_anchor_new": np.asarray([r["is_anchor_new"] for r in records], dtype=np.int64),
        "density": np.asarray([r["density"] for r in records], dtype=np.float32) if records else np.empty((0, NUM_ANCHOR_TYPES - 1), dtype=np.float32),
        "record_pair_side": np.asarray([r["side"] for r in records], dtype=np.int64),
        "record_pair_id": np.asarray([r["pair_id"] for r in records], dtype=np.int64),
        "trx_id": np.asarray([r["trx_id"] for r in records], dtype=np.int64),
        "record_scenario": [r["scenario"] for r in records],
        "record_is_fraud": np.asarray([r["record_is_fraud"] for r in records], dtype=np.int64),
        "current_label": np.asarray([r["current_label"] for r in records], dtype=np.int64),
        "history_has_fraud": np.asarray([r["history_has_fraud"] for r in records], dtype=np.int64),
        "num_trx": len(currents),
        "trx_scenario": transaction_scenarios,
        #"trx_label": np.asarray([int(current["is_attack"]) for current in currents], dtype=np.int64),
    })
    return output


def make_anchored_dataset(
    current_cfg: CFG2,
    num_samples: Optional[int] = None,
    anomaly_position_random_occurs: bool = False,
    num_trx: Optional[int] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("make_anchored_dataset requires PyTorch; use gen_anchored_dataset for NumPy output")
    raw = gen_anchored_dataset(current_cfg, num_samples, anomaly_position_random_occurs, num_trx)
    float_keys = [
        "dt", "amount", "sw_ip", "sw_email", "sw_fp", "sw_bill", "sw_ship", "mask", "step_label", "density", "sw_bca",
        *[f"{prefix}_{name}" for name in ALL_CLEAN_NAMES.values() for prefix in ("cum_dist", "is_new")],
    ]
    long_keys = [
        "anchor_type", "valid_len", "is_anchor_new", "record_pair_side", "record_pair_id", "trx_id",
        "record_is_fraud", "current_label", "history_has_fraud", #"trx_label",
    ]
    output = {key: torch.tensor(raw[key], dtype=torch.float32) for key in float_keys}
    output.update({key: torch.tensor(raw[key], dtype=torch.long) for key in long_keys})
    output.update({
        "record_scenario": raw["record_scenario"],
        "num_trx": raw["num_trx"],
        "trx_scenario": raw["trx_scenario"],
    })
    return output


def gen_one_sequence2(cfg: Any, random_pos_anomaly: bool = False) -> Tuple[np.ndarray, ...]:
    """Backward-compatible single-sequence view using V3 pattern controls."""
    rng = _make_rng(cfg)
    allocator = _EntityAllocator()
    actor = _new_actor(allocator, 1, rng, cfg)
    pattern = _pick_scenario(_scenario_weights(cfg), rng)
    campaign = _CampaignManager(allocator, rng, cfg).acquire(pattern) if pattern in MALICIOUS_PATTERNS else None
    count = _sample_count(pattern, cfg, rng)
    txns = _generate_period(cfg, actor, allocator, rng, pattern, 0.0, 30 * DAY, campaign, count, random_pos_anomaly)
    for i, txn in enumerate(txns):
        txn["trx_uid"] = i
    view = _AnchorView(tuple(txns), tuple(txns[-int(_cfg(cfg, "model_sequence_capacity", _cfg(cfg, "max_seq_len", 35))):]))
    dt, amount, switches, mask, _labels, _length, pattern_start, _cumulative, _is_new = _view_arrays(view, cfg)
    return (
        dt, amount, switches["sw_ip"], switches["sw_email"], switches["sw_fp"],
        switches["sw_bill"], switches["sw_ship"], mask, pattern, pattern_start,
    )


def generate_pattern_transactions(pattern: str, cfg: Any, seed: int = 2026) -> List[dict]:
    """Generate one sampling-period actor for focused tests and inspection."""
    class _SeedCfg:
        def __getattr__(self, name: str) -> Any:
            if name == "generator_seed":
                return seed
            return getattr(cfg, name)

    proxy = _SeedCfg()
    rng = _make_rng(proxy)
    allocator = _EntityAllocator()
    actor = _new_actor(allocator, 1, rng, proxy)
    campaign = _CampaignManager(allocator, rng, proxy).acquire(pattern) if pattern in MALICIOUS_PATTERNS else None
    return _generate_period(proxy, actor, allocator, rng, pattern, 30 * DAY, 60 * DAY, campaign)


def validate_transaction_pool(transactions: Sequence[dict], currents: Sequence[dict], cfg: Any) -> Dict[str, Any]:
    """Return structural validation results and shortcut-oriented statistics."""
    errors: List[str] = []
    ids = [txn.get("trx_uid") for txn in transactions]
    if len(ids) != len(set(ids)):
        errors.append("transaction IDs are not unique")
    if any(float(txn["amt"]) <= 0.0 for txn in transactions):
        errors.append("non-positive transaction amount found")
    required = {"t", "amt", "bca", "em", "fp", "bill", "ship", "ip", "scenario", "is_attack", "actor_id"}
    if any(not required.issubset(txn) for txn in transactions):
        errors.append("transaction schema is incomplete")
    earliest = min((float(txn["t"]) for txn in transactions), default=0.0)
    window = float(_cfg(cfg, "anchor_window_seconds", 30 * DAY))
    if any(float(txn["t"]) < earliest + window for txn in currents):
        errors.append("current transaction exists before full warm-up")
    zero_count = sum(float(txn["amt"]) == 0.0 for txn in transactions)
    labels = [int(bool(txn["is_attack"])) for txn in transactions]
    _, training_start, simulation_end = _simulation_bounds(cfg)
    if any(float(txn["t"]) < 0.0 or float(txn["t"]) >= simulation_end for txn in transactions):
        errors.append("transaction exists outside simulation bounds")

    sampling_by_actor: Dict[int, List[dict]] = {}
    for txn in transactions:
        if float(txn["t"]) >= training_start:
            sampling_by_actor.setdefault(int(txn["actor_id"]), []).append(txn)
    scenario_actor_counts: Dict[str, int] = {}
    count_violations = 0
    for actor_txns in sampling_by_actor.values():
        scenarios = {str(txn["scenario"]) for txn in actor_txns}
        if len(scenarios) != 1:
            errors.append("actor has multiple sampling-period scenarios")
            continue
        scenario = next(iter(scenarios))
        scenario_actor_counts[scenario] = scenario_actor_counts.get(scenario, 0) + 1
        lo, hi = _range_cfg(cfg, "pattern_txn_count_ranges", DEFAULT_PATTERN_TXN_COUNT_RANGES, scenario)
        if not int(lo) <= len(actor_txns) <= int(hi):
            count_violations += 1
        has_attack = any(bool(txn["is_attack"]) for txn in actor_txns)
        if (scenario in MALICIOUS_PATTERNS) != has_attack:
            errors.append(f"attack label does not match scenario {scenario}")
    if count_violations:
        errors.append(f"{count_violations} actors violate pattern transaction-count ranges")
    return {
        "ok": not errors,
        "errors": errors,
        "transaction_count": len(transactions),
        "current_count": len(currents),
        "earliest_timestamp": earliest,
        "earliest_current_timestamp": min((float(txn["t"]) for txn in currents), default=None),
        "zero_amount_count": zero_count,
        "attack_rate": float(sum(labels) / len(labels)) if labels else 0.0,
        "scenario_actor_counts": dict(sorted(scenario_actor_counts.items())),
        "count_range_violations": count_violations,
    }


__all__ = [
    "ANCHOR_FIELDS", "ANCHOR_TYPE_ID", "NUM_ANCHOR_TYPES", "SIDE_ID",
    "DEFAULT_PATTERN_TXN_COUNT_RANGES", "DEFAULT_PATTERN_DT_RANGES",
    "is_anomaly",
    "build_transaction_pool", "generate_pattern_transactions", "validate_transaction_pool",
    "gen_one_sequence2", "gen_anchored_dataset", "make_anchored_dataset",
]
