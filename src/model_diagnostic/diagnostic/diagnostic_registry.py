from __future__ import annotations

from dataclasses import dataclass, field, asdict
import copy
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Deque
import threading
import time

import numpy as np
import torch


@dataclass(frozen=True)
class MetricKey:
    stage: str
    metric_name: str


@dataclass
class MetricSpec:
    stage: str
    metric_name: str
    description: str = ""
    category: Optional[str] = None
    unit: Optional[str] = None
    value_type: str = "scalar"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    value: Any
    timestamp: float
    step: Optional[int] = None
    epoch: Optional[int] = None
    phase: str = "train"
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "epoch": self.epoch,
            "phase": self.phase,
            "tags": self.tags,
        }


class DiagnosticRegistry:

    def __init__(
        self,
        max_records_per_metric: int = 10_000,
    ):
        if max_records_per_metric <= 0:
            raise ValueError("max_records_per_metric must be > 0")

        self.max_records_per_metric = max_records_per_metric

        self._repo: Dict[
            MetricKey,
            Deque[MetricRecord]
        ] = defaultdict(
            lambda: deque(maxlen=max_records_per_metric)
        )

        self._specs: Dict[MetricKey, MetricSpec] = {}

        self._lock = threading.RLock()

    # ---------------------------------------------------------
    # Registration / discovery
    # ---------------------------------------------------------

    def register_metric(
        self,
        stage: str,
        metric_name: str,
        *,
        description: str = "",
        category: Optional[str] = None,
        unit: Optional[str] = None,
        value_type: str = "scalar",
    ) -> None:

        key = MetricKey(stage, metric_name)

        spec = MetricSpec(
            stage=stage,
            metric_name=metric_name,
            description=description,
            category=category,
            unit=unit,
            value_type=value_type,
        )

        with self._lock:
            self._specs[key] = spec

    def _ensure_registered(
        self,
        stage: str,
        metric_name: str,
        value: Any,
    ) -> None:

        key = MetricKey(stage, metric_name)

        if key in self._specs:
            return

        self._specs[key] = MetricSpec(
            stage=stage,
            metric_name=metric_name,
            value_type=self._infer_value_type(value),
        )

    def list_metrics(
        self,
        stage: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        with self._lock:
            specs = list(self._specs.values())

        if stage is not None:
            specs = [
                spec for spec in specs
                if spec.stage == stage
            ]

        return [spec.to_dict() for spec in specs]

    def list_stages(self) -> List[str]:

        with self._lock:
            return sorted({
                key.stage
                for key in self._specs
            })

    # ---------------------------------------------------------
    # Store
    # ---------------------------------------------------------

    def store(
        self,
        stage: str,
        metric_name: str,
        value: Any,
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train",
        timestamp: Optional[float] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:

        ts = time.time() if timestamp is None else timestamp
        value = self._normalize_value(value)

        key = MetricKey(stage, metric_name)

        record = MetricRecord(
            value=value,
            timestamp=ts,
            step=step,
            epoch=epoch,
            phase=phase,
            tags=tags or {},
        )

        with self._lock:
            self._ensure_registered(
                stage,
                metric_name,
                value,
            )

            self._repo[key].append(record)

    def store_many(
        self,
        stage: str,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train",
        timestamp: Optional[float] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:

        ts = time.time() if timestamp is None else timestamp
        shared_tags = dict(tags or {})

        records = []

        for metric_name, raw_value in metrics.items():

            value = self._normalize_value(raw_value)

            key = MetricKey(stage, metric_name)

            record = MetricRecord(
                value=value,
                timestamp=ts,
                step=step,
                epoch=epoch,
                phase=phase,
                tags=shared_tags,
            )

            records.append(
                (key, metric_name, value, record)
            )

        # one lock for the entire batch
        with self._lock:

            for key, metric_name, value, record in records:

                self._ensure_registered(
                    stage,
                    metric_name,
                    value,
                )

                self._repo[key].append(record)

    # ---------------------------------------------------------
    # Current
    # ---------------------------------------------------------

    def get_current(
        self,
        stage: str,
        metric_name: str,
        *,
        phase: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:

        key = MetricKey(stage, metric_name)

        with self._lock:

            records = self._repo.get(key)

            if not records:
                return None

            if phase is None:
                return records[-1].to_dict()

            for record in reversed(records):
                if record.phase == phase:
                    return record.to_dict()

        return None

    # ---------------------------------------------------------
    # History / window
    # ---------------------------------------------------------

    def get_history(
        self,
        stage: str,
        metric_name: str,
        *,
        phase: Optional[str] = None,
        last_n: Optional[int] = None,
        since_step: Optional[int] = None,
        until_step: Optional[int] = None,
        since_timestamp: Optional[float] = None,
        until_timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:

        key = MetricKey(stage, metric_name)

        result = []

        with self._lock:

            records = self._repo.get(key)

            if not records:
                return []

            # Iterate newest -> oldest.
            # Allows early exit for last_n / lower bounds.
            for record in reversed(records):

                if (
                    until_step is not None
                    and record.step is not None
                    and record.step > until_step
                ):
                    continue

                if (
                    until_timestamp is not None
                    and record.timestamp > until_timestamp
                ):
                    continue

                # Since records are append-ordered,
                # lower bound means we can stop.
                if (
                    since_step is not None
                    and record.step is not None
                    and record.step < since_step
                ):
                    break

                if (
                    since_timestamp is not None
                    and record.timestamp < since_timestamp
                ):
                    break

                if phase is not None and record.phase != phase:
                    continue

                result.append(record)

                if (
                    last_n is not None
                    and len(result) >= last_n
                ):
                    break

        # return chronological order
        result.reverse()

        return [record.to_dict() for record in result]

    # ---------------------------------------------------------
    # Unified Agent API
    # ---------------------------------------------------------

    def get(
        self,
        stage: str,
        metric_name: str,
        *,
        history: bool = False,
        **kwargs,
    ) -> Any:

        if history:
            return self.get_history(
                stage,
                metric_name,
                **kwargs,
            )

        return self.get_current(
            stage,
            metric_name,
            phase=kwargs.get("phase"),
        )

    # ---------------------------------------------------------
    # Snapshot / maintenance
    # ---------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a detached JSON-safe snapshot of all registered metrics/history."""
        with self._lock:
            metric_entries = []
            total_records = 0

            keys = sorted(
                set(self._specs) | set(self._repo),
                key=lambda key: (key.stage, key.metric_name),
            )
            for key in keys:
                spec = self._specs.get(key)
                records = self._repo.get(key, ())
                record_dicts = [copy.deepcopy(record.to_dict()) for record in records]
                total_records += len(record_dicts)
                metric_entries.append(
                    {
                        "stage": key.stage,
                        "metric_name": key.metric_name,
                        "spec": copy.deepcopy(spec.to_dict()) if spec is not None else None,
                        "records": record_dicts,
                    }
                )

            return {
                "metric_count": len(metric_entries),
                "record_count": total_records,
                "stages": sorted({entry["stage"] for entry in metric_entries}),
                "metrics": metric_entries,
            }

    def clear(self, *, reset_specs: bool = False) -> None:
        """Clear collected history; optionally clear discovered metric specs too."""
        with self._lock:
            self._repo.clear()
            if reset_specs:
                self._specs.clear()

    # ---------------------------------------------------------
    # Value helpers
    # ---------------------------------------------------------

    @staticmethod
    def _infer_value_type(value: Any) -> str:

        if isinstance(value, bool):
            return "boolean"

        if isinstance(value, (int, float)):
            return "scalar"

        if isinstance(value, dict):
            return "structured"

        if isinstance(value, (list, tuple)):
            return "vector"

        return "object"

    @staticmethod
    def _normalize_value(value: Any) -> Any:

        if isinstance(value, torch.Tensor):

            value = value.detach()

            if value.numel() == 1:
                return value.item()

            # Convenience only.
            # Large tensors should normally be summarized
            # by the metric producer before storing.
            return value.cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {
                str(k): DiagnosticRegistry._normalize_value(v)
                for k, v in value.items()
            }

        if isinstance(value, tuple):
            return [
                DiagnosticRegistry._normalize_value(v)
                for v in value
            ]

        if isinstance(value, list):
            return [
                DiagnosticRegistry._normalize_value(v)
                for v in value
            ]

        return value


# Process-global repository
DIAGNOSTICS = DiagnosticRegistry()