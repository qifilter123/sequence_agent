"""Build causal, model-independent features from the Alpaca Parquet store.

``collect_np(symbol, start_date, end_date, steps=LOOKBACK_BARS)`` returns sliding samples
whose lookback contains exactly ``steps`` observed regular-session 15-minute
bars. The returned arrays share a sample axis and can be sliced into batches:

* ``delta_features``: float32 [samples, steps, len(DELTA_FEATURE_NAMES)]
* ``volume_bar``: float64 [samples, steps], current volume at each input bar
* ``vwap_bar``: float64 [samples, steps], VWAP at each input bar
* ``history_context``: float32 [samples, len(CONTEXT_NAMES)]
* ``sequence_position``: int64 [samples, steps]
* ``timestamp_utc``, ``session_date``, ``slot_ordinal``: input-bar metadata

Input history may span sessions. Every observed bar can end a sample, including
the final bar of a session. No future VWAP or target is returned. Missing
historical bars are not padded. ``sequence_position`` starts at zero and advances by
the actual UTC time difference in 15-minute units, including nights, weekends,
holidays, missing bars, and daylight-saving changes. The model can convert these raw
positions to standard Transformer sinusoidal encodings at its own embedding
dimension. Calendar and market context can be added separately.

The 15-minute baseline uses the preceding scheduled slot in the same session.
The one-hour baseline uses four earlier slots in the current session. The
one-day baseline uses the 26 immediately preceding scheduled trading slots,
even across a session boundary. Yesterday's hour uses up to four slots on
the preceding trading session, ending at the current slot's clock position.
Context windows use 5, 10, and 21 preceding complete trading sessions.
Historical VWAP is weighted by each bar's volume; historical volume is the
mean volume per observed bar. Absent history produces zero.

Both raw and all-adjusted bars must be present and complete at a timestamp.
The values keep all-adjusted VWAP and all-adjusted volume separate;
the collector does not combine price and volume. No API requests,
normalization fitting, or future bars are included in the features.

Dependencies: numpy, pandas, pyarrow, exchange-calendars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "stock"
LOOKBACK_DAYS = 3
FULL_SESSION_BARS = 26
LOOKBACK_BARS = LOOKBACK_DAYS * FULL_SESSION_BARS
_NY = "America/New_York"
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")
_CONTEXT_SESSIONS = (("5d", 5), ("10d", 10), ("1month", 21))
DELTA_FEATURE_NAMES = (
    "delta_p_15min", "delta_v_15min",
    "delta_p_1hr", "delta_v_1hr",
    "delta_p_1day", "delta_v_1day",
    "delta_p_1hr_yesterday", "delta_v_1hr_yesterday",
)
CONTEXT_NAMES = tuple(
    name
    for label, _ in _CONTEXT_SESSIONS
    for name in (f"vwap_prev_{label}", f"volume_prev_{label}")
)


class _Bar(NamedTuple):
    vwap_all: float
    volume_all: float


class _HistoryStats(TypedDict):
    vwap_volume_sum: float
    total_volume: float
    valid_bars: int


@dataclass(frozen=True)
class _HistoryView:
    day: date
    slot: int
    current_slots: dict[int, _Bar]
    previous_days: dict[date, dict[int, _Bar]]
    trading_days: list[date]
    day_to_index: dict[date, int]
    scheduled_count: dict[date, int]


def _calendar():
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ImportError(
            "Install exchange-calendars for the XNYS trading calendar"
        ) from exc
    return xcals.get_calendar("XNYS")


def _file_path(root: Path, symbol: str, adjustment: str, year: int) -> Path:
    return (
        root
        / "bars"
        / "timeframe=15Min"
        / "feed=sip"
        / f"adjustment={adjustment}"
        / f"symbol={symbol}"
        / f"year={year}"
        / "bars.parquet"
    )


def _read_adjustment_file(path: Path, symbol: str, adjustment: str) -> pd.DataFrame:
    columns = [
        "symbol",
        "timestamp_utc",
        "timeframe",
        "feed",
        "adjustment",
        "is_complete",
        "vwap",
        "volume",
    ]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except (ImportError, OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot read required columns from {path}: {exc}") from exc
    if (
        not frame["symbol"].eq(symbol).all()
        or not frame["timeframe"].eq("15Min").all()
        or not frame["feed"].eq("sip").all()
        or not frame["adjustment"].eq(adjustment).all()
    ):
        raise ValueError(f"Partition metadata disagrees with its path: {path}")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if frame["timestamp_utc"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {symbol}/{adjustment} partitions")
    return frame


def _read_data_from_file(raw_path: Path, all_path: Path, symbol: str) -> pd.DataFrame:
    """Load and align one year's raw and adjusted Parquet partitions."""
    raw = _read_adjustment_file(raw_path, symbol, "raw")
    adjusted = _read_adjustment_file(all_path, symbol, "all")
    bars = raw[["timestamp_utc", "is_complete"]].merge(
        adjusted[["timestamp_utc", "is_complete", "vwap", "volume"]],
        on="timestamp_utc",
        how="inner",
        suffixes=("_raw", "_all"),
        validate="one_to_one",
    )
    bars = bars.rename(columns={"vwap": "vwap_all", "volume": "volume_all"})
    for column in ("vwap_all", "volume_all"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    mask = (
        bars["is_complete_raw"].eq(True)
        & bars["is_complete_all"].eq(True)
        & np.isfinite(bars["vwap_all"])
        & np.isfinite(bars["volume_all"])
        & bars["vwap_all"].gt(0)
        & bars["volume_all"].gt(0)
    )
    bars = bars.loc[mask, ["timestamp_utc", "vwap_all", "volume_all"]].copy()
    return bars.sort_values("timestamp_utc").reset_index(drop=True)


def _volume_mean(total: float, valid_bars: int) -> float:
    """Mean volume per observed bar, or zero for an empty window."""
    return total / valid_bars if valid_bars else 0.0


def _history_vwap(stats: _HistoryStats) -> float:
    """VWAP across bars, weighting each bar VWAP by its adjusted volume."""
    volume = stats["total_volume"]
    return stats["vwap_volume_sum"] / volume if volume else 0.0


def _stats(values) -> _HistoryStats:
    vwap_volume_sum = total_volume = 0.0
    valid_bars = 0
    for bar in values:
        vwap_volume_sum += bar.vwap_all * bar.volume_all
        total_volume += bar.volume_all
        valid_bars += 1
    return {
        "vwap_volume_sum": vwap_volume_sum,
        "total_volume": total_volume,
        "valid_bars": valid_bars,
    }


def _field_with_history(hours: float, view: _HistoryView) -> _HistoryStats:
    """Collect the same-session hour or preceding complete sessions."""
    if hours == 1.0:
        slots = range(max(0, view.slot - 4), view.slot)
        return _stats(view.current_slots[s] for s in slots if s in view.current_slots)
    sessions = {6.5 * n: n for _, n in _CONTEXT_SESSIONS}
    if hours not in sessions:
        raise ValueError(f"Unsupported trading-hour window: {hours}")
    index = view.day_to_index[view.day]
    days = view.trading_days[max(0, index - sessions[hours]):index]
    return _stats(
        bar for day in days for bar in view.previous_days.get(day, {}).values()
    )


def _prior_slots(view: _HistoryView, count: int) -> _HistoryStats:
    """Previous scheduled slots, crossing session boundaries without padding."""
    index, slot = view.day_to_index[view.day], view.slot
    selected: list[_Bar] = []
    expected = 0
    while expected < count and index >= 0:
        day = view.trading_days[index]
        bars = view.current_slots if day == view.day else view.previous_days.get(day, {})
        take = min(slot, count - expected)
        selected.extend(bars[s] for s in range(slot - take, slot) if s in bars)
        expected += take
        index -= 1
        if index >= 0:
            slot = view.scheduled_count[view.trading_days[index]]
    return _stats(selected)


def _yesterday_hour(view: _HistoryView) -> _HistoryStats:
    index = view.day_to_index[view.day] - 1
    if index < 0:
        return _stats(())
    day = view.trading_days[index]
    slots = range(max(0, view.slot - 3), min(view.slot + 1, view.scheduled_count[day]))
    bars = view.previous_days.get(day, {})
    return _stats(bars[s] for s in slots if s in bars)


def _delta(current: float, historical: float) -> float:
    return current / (current + historical) - 0.5 if historical > 0 else 0.0


def _empty_data() -> dict[str, np.ndarray]:
    return {
        "delta_features": np.empty((0, len(DELTA_FEATURE_NAMES)), dtype=np.float32),
        "volume_bar": np.empty(0, dtype=np.float64),
        "vwap_bar": np.empty(0, dtype=np.float64),
        "history_context": np.empty((0, len(CONTEXT_NAMES)), dtype=np.float32),
        "timestamp_utc": np.empty(0, dtype="datetime64[ns]"),
        "session_date": np.empty(0, dtype="datetime64[D]"),
        "slot_ordinal": np.empty(0, dtype=np.int64),
    }


def _logical_positions(timestamps_utc: np.ndarray) -> np.ndarray:
    """Elapsed wall-clock 15-minute units since each sample's first bar."""
    if timestamps_utc.ndim != 2:
        raise ValueError("timestamps_utc must be a 2-D array")
    if timestamps_utc.shape[1] == 0:
        return np.empty(timestamps_utc.shape, dtype=np.int64)
    elapsed = timestamps_utc - timestamps_utc[:, :1]
    units = elapsed / np.timedelta64(15, "m")
    if (units < 0).any() or not np.isfinite(units).all() or not np.equal(units, np.floor(units)).all():
        raise ValueError("Input timestamps must be ordered on 15-minute boundaries")
    positions = units.astype(np.int64)
    if (np.diff(positions, axis=1) <= 0).any():
        raise ValueError("Input timestamps must be strictly increasing")
    return positions


def _collect_np(
    bars: pd.DataFrame,
    year: int,
    start_date: date,
    end_date: date,
    *,
    trading_days: list[date],
    day_to_index: dict[date, int],
    schedule: dict[date, tuple[pd.Timestamp, int]],
    slot_starts: np.ndarray,
    previous_days: dict[date, dict[int, _Bar]],
) -> dict[str, np.ndarray]:
    """Collect one yearly file pair, retaining its days for following files."""
    current_year: dict[date, dict[int, _Bar]] = {}
    timestamps: dict[tuple[date, int], pd.Timestamp] = {}
    for item in bars.itertuples(index=False):
        timestamp = item.timestamp_utc
        day = timestamp.tz_convert(_NY).date()
        if day.year != year:
            raise ValueError(f"Bar {timestamp} stored under wrong year={year}")
        if day not in schedule:
            continue
        opening, count = schedule[day]
        delta = timestamp - opening
        slot = int(delta / pd.Timedelta(minutes=15))
        if slot < 0 or slot >= count or delta != pd.Timedelta(minutes=15 * slot):
            continue
        current_year.setdefault(day, {})[slot] = _Bar(
            float(item.vwap_all), float(item.volume_all)
        )
        timestamps[(day, slot)] = timestamp

    delta_features: list[tuple[float, ...]] = []
    contexts: list[tuple[float, ...]] = []
    output_volume: list[float] = []
    output_vwap: list[float] = []
    output_stamps: list[np.datetime64] = []
    output_days: list[date] = []
    output_slots: list[int] = []
    scheduled_count = {day: count for day, (_, count) in schedule.items()}
    for day in sorted(current_year):
        day_slots = current_year[day]
        if start_date <= day <= end_date:
            base = _HistoryView(
                day,
                0,
                day_slots,
                previous_days,
                trading_days,
                day_to_index,
                scheduled_count,
            )
            daily = [_field_with_history(6.5 * n, base) for _, n in _CONTEXT_SESSIONS]
            context = tuple(
                value
                for s in daily
                for value in (
                    _history_vwap(s),
                    _volume_mean(s["total_volume"], s["valid_bars"]),
                )
            )
            for slot in sorted(day_slots):
                bar = day_slots[slot]
                view = _HistoryView(
                    day,
                    slot,
                    day_slots,
                    previous_days,
                    trading_days,
                    day_to_index,
                    scheduled_count,
                )
                hourly = _field_with_history(1.0, view)
                prior_slots = range(max(0, slot - 1), slot)
                prior = _stats(day_slots[s] for s in prior_slots if s in day_slots)
                day_history = _prior_slots(view, 26)
                yesterday = _yesterday_hour(view)
                stats = (prior, hourly, day_history, yesterday)
                means = [
                    (_history_vwap(s),
                     _volume_mean(s["total_volume"], s["valid_bars"]))
                    for s in stats
                ]
                delta_features.append(
                    (
                        *(
                            value
                            for price, volume in means
                            for value in (_delta(bar.vwap_all, price),
                                          _delta(bar.volume_all, volume))
                        ),
                    )
                )
                contexts.append(context)
                output_volume.append(bar.volume_all)
                output_vwap.append(bar.vwap_all)
                output_stamps.append(
                    timestamps[(day, slot)].tz_localize(None).to_datetime64()
                )
                output_days.append(day)
                output_slots.append(int(slot_starts[day_to_index[day]]) + slot)
        previous_days[day] = day_slots

    if not delta_features:
        return _empty_data()
    return {
        "delta_features": np.asarray(delta_features, dtype=np.float32),
        "volume_bar": np.asarray(output_volume, dtype=np.float64),
        "vwap_bar": np.asarray(output_vwap, dtype=np.float64),
        "history_context": np.asarray(contexts, dtype=np.float32),
        "timestamp_utc": np.asarray(output_stamps, dtype="datetime64[ns]"),
        "session_date": np.asarray(output_days, dtype="datetime64[D]"),
        "slot_ordinal": np.asarray(output_slots, dtype=np.int64),
    }


def _collect_chunks(
    symbol: str,
    start_date: date,
    end_date: date,
    steps: int,
    *,
    data_root: Path = DATA_ROOT,
) -> list[dict[str, np.ndarray]]:
    """Read yearly file pairs; process each with retained historical state."""
    symbol = symbol.strip().upper()
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError(f"Invalid stock symbol: {symbol!r}")
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("start_date and end_date must be datetime.date values")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    calendar = _calendar()
    if (
        start_date < calendar.first_session.date()
        or end_date > calendar.last_session.date()
    ):
        raise ValueError("Dates outside the installed XNYS calendar range")
    session_start = pd.Timestamp(start_date)
    if calendar.sessions.tz is not None:
        session_start = session_start.tz_localize(calendar.sessions.tz)
    first_position = int(calendar.sessions.searchsorted(session_start))
    # Read enough earlier sessions to build a full input window at the first
    # requested bar, then another 21 sessions for that bar's causal features.
    feature_position = first_position
    missing_slots = steps - 1
    while missing_slots > 0 and feature_position > 0:
        feature_position -= 1
        session = calendar.sessions[feature_position]
        duration = calendar.session_close(session) - calendar.session_open(session)
        missing_slots -= int(duration / pd.Timedelta(minutes=15))
    feature_start = calendar.sessions[feature_position].date()
    earliest = calendar.sessions[max(0, feature_position - 21)].date()
    sessions = calendar.sessions_in_range(earliest, end_date)
    if len(sessions) == 0 or sessions[-1].date() < start_date:
        return []

    schedule: dict[date, tuple[pd.Timestamp, int]] = {}
    for session in sessions:
        opening, closing = (
            calendar.session_open(session),
            calendar.session_close(session),
        )
        count = int((closing - opening) / pd.Timedelta(minutes=15))
        if (
            count < 1
            or count > 26
            or closing - opening != pd.Timedelta(minutes=15 * count)
        ):
            raise ValueError(f"Unexpected XNYS session duration on {session.date()}")
        schedule[session.date()] = (opening, count)

    trading_days = [session.date() for session in sessions]
    day_to_index = {day: i for i, day in enumerate(trading_days)}
    slot_starts = np.r_[0, np.cumsum([schedule[day][1] for day in trading_days])]
    previous_days: dict[date, dict[int, _Bar]] = {}
    chunks: list[dict[str, np.ndarray]] = []
    found_file_pair = False
    for year in range(earliest.year, end_date.year + 1):
        raw_path = _file_path(Path(data_root), symbol, "raw", year)
        all_path = _file_path(Path(data_root), symbol, "all", year)
        if not raw_path.exists() and not all_path.exists():
            continue
        if not raw_path.is_file() or not all_path.is_file():
            raise FileNotFoundError(
                f"Both raw and all-adjusted files are required for {symbol}/year={year}"
            )
        found_file_pair = True
        chunk = _collect_np(
            _read_data_from_file(raw_path, all_path, symbol),
            year,
            feature_start,
            end_date,
            trading_days=trading_days,
            day_to_index=day_to_index,
            schedule=schedule,
            slot_starts=slot_starts,
            previous_days=previous_days,
        )
        if len(chunk["delta_features"]):
            chunks.append(chunk)
    if not found_file_pair:
        raise FileNotFoundError(f"No 15Min/sip raw and all Parquet files for {symbol}")
    return chunks


def collect_np(
    symbol: str, start_date: date, end_date: date, steps: int = LOOKBACK_BARS
) -> dict[str, np.ndarray]:
    """Build samples with N=steps observed past bars (default: three full days).

    Historical inputs may begin before start_date. A sample ends at every
    observed bar in the requested date range. Comparison order follows
    DELTA_FEATURE_NAMES. Position starts at 0 and reflects actual elapsed
    UTC time / 15 minutes. Its maximum varies with the sample's dates and
    missing bars, even when steps is fixed. The returned positions are ready
    for the model's positional encoding.

    Returns arrays with a shared first dimension M (number of samples):
        delta_features: float32 [M, steps, 8], eight historical comparisons.
        volume_bar: float64 [M, steps], untransformed all-adjusted volume.
        vwap_bar: float64 [M, steps], all-adjusted VWAP at each input bar.
        history_context: float32 [M, 6], context at the input endpoint:
            (VWAP, mean volume) for the preceding 5, 10, 21 complete sessions.
        sequence_position: int64 [M, steps], elapsed 15-minute position.
        timestamp_utc: datetime64[ns] [M, steps], input bar start times.
        session_date: datetime64[D] [M, steps], New York session dates.
        slot_ordinal: int64 [M, steps], scheduled slot indices; gaps remain.
    """
    chunks = _collect_chunks(symbol, start_date, end_date, steps)
    if not chunks:
        return _empty_samples(steps)
    data = {
        key: np.concatenate([chunk[key] for chunk in chunks]) for key in _empty_data()
    }
    count = len(data["vwap_bar"])
    if count < steps:
        return _empty_samples(steps)

    endpoints = np.arange(steps - 1, count)
    endpoint_dates = data["session_date"][endpoints]
    endpoints = endpoints[
        (endpoint_dates >= np.datetime64(start_date))
        & (endpoint_dates <= np.datetime64(end_date))
    ]
    if not len(endpoints):
        return _empty_samples(steps)

    indices = endpoints[:, None] - steps + 1 + np.arange(steps)[None, :]
    timestamps = data["timestamp_utc"][indices]
    sessions = data["session_date"][indices]
    ordinals = data["slot_ordinal"][indices]
    return {
        "delta_features": data["delta_features"][indices],
        "volume_bar": data["volume_bar"][indices],
        "vwap_bar": data["vwap_bar"][indices],
        "history_context": data["history_context"][endpoints],
        "sequence_position": _logical_positions(timestamps),
        "timestamp_utc": timestamps,
        "session_date": sessions,
        "slot_ordinal": ordinals,
    }

def load_input_data(symbol: str, start: date, end: date, lookup_bars: int = LOOKBACK_BARS) -> dict[str, np.ndarray]:
    input_data = collect_np(symbol, start, end, lookup_bars)

    if len(input_data["delta_features"]) == 0:
        raise ValueError("data_collector returned no training samples")
    return input_data


def _empty_samples(steps: int) -> dict[str, np.ndarray]:
    return {
        "delta_features": np.empty((0, steps, len(DELTA_FEATURE_NAMES)), dtype=np.float32),
        "volume_bar": np.empty((0, steps), dtype=np.float64),
        "vwap_bar": np.empty((0, steps), dtype=np.float64),
        "history_context": np.empty((0, len(CONTEXT_NAMES)), dtype=np.float32),
        "sequence_position": np.empty((0, steps), dtype=np.int64),
        "timestamp_utc": np.empty((0, steps), dtype="datetime64[ns]"),
        "session_date": np.empty((0, steps), dtype="datetime64[D]"),
        "slot_ordinal": np.empty((0, steps), dtype=np.int64),
    }
