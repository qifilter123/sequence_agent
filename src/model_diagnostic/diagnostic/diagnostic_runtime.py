from __future__ import annotations

import copy
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from model_diagnostic import batch_util
from model_diagnostic import model_trainer as train_driver
from model_diagnostic import inference_hbscan as hdbscan_eval
from model_diagnostic import generic_seq_generator as seq_gen
from model_diagnostic.cfg_base import CFG2
from model_diagnostic.diagnostic.diagnostic_registry import DiagnosticRegistry
from model_diagnostic.diagnostic.diagnostic_model_structure import render_model_structure
from model_diagnostic.diagnostic.diagnostic_probe_manager import DiagnosticProbeManager

class DiagnosticRuntimeError(RuntimeError):
    """Raised when a diagnostic runtime operation is not valid for its state."""


class DiagnosticRuntime:
    """
    Stateful runtime backing the first diagnostic MCP surface.

    The runtime deliberately owns the mutable objects that must survive across
    multiple agent tool calls in one MCP server process:

      * model
      * optimizer
      * generated training dataset / transaction index
      * DiagnosticProbeManager
      * DiagnosticRegistry history
      * monotonic diagnostic step counter

    Probe policy is loaded from CFG.diagnostic_config_path. The agent may only
    vary CFG2-declared experiment parameters; source code and fixed CFG values
    remain outside the MCP write surface.
    """

    MAX_STEPS_PER_SEGMENT = 6000
    MAX_METRIC_QUERIES = 16
    MAX_HISTORY_RECORDS_PER_QUERY = 200
    DEFAULT_HISTORY_RECORDS = 50

    _SUPPORTED_ENV_OVERRIDES = {
        "DIAGNOSTIC_DEVICE": ("device", str),
        "DIAGNOSTIC_NUM_TRX": ("num_trx", int),
        "DIAGNOSTIC_SEED": ("seed", int),
    }

    def __init__(
        self,
        *,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._lock = threading.RLock()

        # Diagnostic policy location is a fixed CFG contract. It is resolved
        # from the concrete session cfg, never from MCP arguments or env vars.
        self.diagnostic_config_path: Optional[Path] = None

        self._config_overrides = dict(config_overrides or {})
        self._validate_config_overrides(self._config_overrides)

        self.registry: DiagnosticRegistry = DiagnosticRegistry()

        self.initialized = False
        self.session_id: Optional[str] = None
        self.started_at: Optional[float] = None
        self.session_step = 0
        self.model_step = 0
        self.segment_count = 0

        self.cfg: Optional[CFG2] = None
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.probe = None
        self.train_data: Optional[Dict[str, Any]] = None
        self.txn_index: Optional[Dict[int, List[int]]] = None
        self.structure: Optional[Dict[str, Any]] = None

        self._training_record_count = 0
        self._transaction_count = 0
        self._last_segment: Optional[Dict[str, Any]] = None

        # Snapshots from the immediately preceding successful session. They are
        # replaced only after a new restart_session() has initialized successfully.
        self.previous_cfg2: Optional[Dict[str, Any]] = None
        self.previous_metrics: Optional[Dict[str, Any]] = None
        self.previous_session: Optional[Dict[str, Any]] = None
        self.previous_structure: Optional[Dict[str, Any]] = None

        # Downstream evaluation belongs to the experiment snapshot just like
        # CFG2/metrics/structure. Only the most recent evaluation per session is kept.
        self.current_evaluation: Optional[Dict[str, Any]] = None
        self.previous_evaluation: Optional[Dict[str, Any]] = None

    @classmethod
    def from_environment(cls) -> "DiagnosticRuntime":
        """Build runtime host configuration from environment variables."""
        overrides: Dict[str, Any] = {}

        for env_name, (cfg_name, parser) in cls._SUPPORTED_ENV_OVERRIDES.items():
            raw = os.getenv(env_name)
            if raw is None or raw == "":
                continue
            try:
                overrides[cfg_name] = parser(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {env_name}={raw!r}: expected {parser.__name__}"
                ) from exc

        return cls(config_overrides=overrides)

    # ------------------------------------------------------------------
    # Public MCP-facing operations
    # ------------------------------------------------------------------

    def initialize_session(self) -> Dict[str, Any]:
        """Initialize the default CFG2 baseline session.

        initialize_session never creates PREVIOUS_* evidence. Repeated calls while
        active are idempotent. Starting a fresh baseline clears stale comparison
        state because no restart replacement produced that baseline.
        """
        with self._lock:
            if self.initialized:
                result = self._session_summary()
                result["already_initialized"] = True
                return result

            cfg = self._build_effective_cfg(restart_overrides=None)
            candidate = self._build_session_candidate(cfg)
            self._commit_session_candidate(candidate)

            self.previous_cfg2 = None
            self.previous_metrics = None
            self.previous_session = None
            self.previous_structure = None
            self.previous_evaluation = None

            result = self._session_summary()
            result["already_initialized"] = False
            result["comparison_ready"] = False
            return result

    def get_tunable_parameters(self) -> Dict[str, Any]:
        """Return the exact CFG2 fields the agent may override.

        The allowlist comes directly from CFG2.get_tunable_parameters(),
        which is the single source of truth for agent-overridable parameters.
        Inherited fixed CFG fields are excluded there.
        """
        with self._lock:
            defaults = CFG2()
            current = self.cfg if self.initialized and self.cfg is not None else None
            parameters = []
            for name in self._tunable_field_names():
                default = getattr(defaults, name)
                current_value = getattr(current, name) if current is not None else default
                default_safe, _ = self._to_json_safe(default)
                current_safe, _ = self._to_json_safe(current_value)
                parameters.append({
                    "name": name,
                    "type": self._config_value_type_name(default),
                    "default": default_safe,
                    "current": current_safe,
                })

            return {
                "source_of_truth": "CFG2.get_tunable_parameters()",
                "count": len(parameters),
                "parameters": parameters,
                "fixed_cfg_overrides_allowed": False,
                "source_code_modification_allowed": False,
            }

    def restart_session(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Replace the active session with a fresh controlled CFG2 experiment.

        PREVIOUS_* is published only after a new session is created successfully.
        If there is no active predecessor, the call starts a fresh overridden
        session but comparison remains unavailable.
        """
        with self._lock:
            normalized_overrides = dict(overrides or {})
            self._validate_restart_overrides(normalized_overrides)

            candidate_cfg = self._build_effective_cfg(
                restart_overrides=normalized_overrides,
            )

            had_active = self.initialized
            prior_cfg2 = (
                self._snapshot_tunable_config(self.cfg)
                if had_active and self.cfg is not None
                else None
            )
            prior_metrics = self.registry.snapshot() if had_active else None
            prior_session = self._session_summary() if had_active else None
            prior_structure = copy.deepcopy(self.structure) if had_active else None
            prior_evaluation = (
                copy.deepcopy(self.current_evaluation)
                if had_active
                and isinstance(self.current_evaluation, Mapping)
                and self.current_evaluation.get("model_step") == self.model_step
                else None
            )

            if had_active:
                self._terminate_current_session()

            candidate = self._build_session_candidate(candidate_cfg)
            self._commit_session_candidate(candidate)

            if had_active and prior_session is not None:
                self.previous_cfg2 = copy.deepcopy(prior_cfg2)
                self.previous_metrics = copy.deepcopy(prior_metrics)
                self.previous_session = copy.deepcopy(prior_session)
                self.previous_structure = copy.deepcopy(prior_structure)
                self.previous_evaluation = copy.deepcopy(prior_evaluation)
            else:
                self.previous_cfg2 = None
                self.previous_metrics = None
                self.previous_session = None
                self.previous_structure = None
                self.previous_evaluation = None

            result = self._session_summary()
            result["restart_overrides"] = copy.deepcopy(normalized_overrides)
            result["previous_session_available"] = self.previous_session is not None
            result["comparison_ready"] = self.previous_session is not None
            result["changed_cfg2"] = self._diff_cfg2(
                self.previous_cfg2,
                result.get("cfg2"),
            )
            return result

    def get_previous_cfg2(self) -> Dict[str, Any]:
        """Return CFG2 values from the prior successfully replaced session."""
        with self._lock:
            available = self.previous_cfg2 is not None
            return {
                "available": available,
                "reason": None if available else (
                    "No previous experiment is available. initialize_session() never "
                    "creates previous evidence; PREVIOUS_* is published only after "
                    "restart_session() successfully replaces an active session."
                ),
                "session": copy.deepcopy(self.previous_session),
                "cfg2": copy.deepcopy(self.previous_cfg2),
            }

    def get_previous_metrics(self) -> Dict[str, Any]:
        """Return the metric snapshot from the prior successfully replaced session."""
        with self._lock:
            available = self.previous_metrics is not None
            return {
                "available": available,
                "reason": None if available else (
                    "No previous experiment is available until restart_session() "
                    "successfully replaces an active session."
                ),
                "session": copy.deepcopy(self.previous_session),
                "metrics": copy.deepcopy(self.previous_metrics),
            }

    def get_previous_structure(self) -> Dict[str, Any]:
        """Return structure from the prior successfully replaced session."""
        with self._lock:
            available = self.previous_structure is not None
            return {
                "available": available,
                "reason": None if available else (
                    "No previous experiment is available until restart_session() "
                    "successfully replaces an active session."
                ),
                "session": copy.deepcopy(self.previous_session),
                "model_structure": copy.deepcopy(self.previous_structure),
            }

    def get_previous_evaluation(self) -> Dict[str, Any]:
        """Return the HDBSCAN evaluation from the prior replaced session, if run."""
        with self._lock:
            available = self.previous_evaluation is not None
            return {
                "available": available,
                "reason": None if available else (
                    "No previous HDBSCAN evaluation is available. Evaluate a session "
                    "before restart_session() replaces it."
                ),
                "session": copy.deepcopy(self.previous_session),
                "evaluation": copy.deepcopy(self.previous_evaluation),
            }

    def evaluate_hdbscan(self) -> Dict[str, Any]:
        """Run downstream HDBSCAN evaluation on the active in-memory model.

        No checkpoint round-trip is used. Human-readable evaluator output is
        redirected to stderr so stdio MCP protocol output remains clean. Probe
        hooks are temporarily disabled so evaluation forwards cannot pollute the
        training diagnostic registry. The evaluator itself preserves model mode
        and RNG state.
        """
        with self._lock:
            self._require_initialized()
            assert self.cfg is not None
            assert self.model is not None

            started = time.perf_counter()
            evaluated_model_step = self.model_step
            probe_was_started = bool(
                self.probe is not None and getattr(self.probe, "_started", False)
            )

            result: Optional[Dict[str, Any]] = None
            error: Optional[Exception] = None
            if probe_was_started:
                self.probe.stop()
            try:
                with redirect_stdout(sys.stderr):
                    result = hdbscan_eval.run_hdbscan_evaluation(self.model, self.cfg)
            except Exception as exc:
                error = exc
            finally:
                if probe_was_started and self.probe is not None:
                    try:
                        self.probe.start()
                    except Exception as probe_exc:
                        if error is None:
                            error = probe_exc

            elapsed = time.perf_counter() - started
            if error is not None:
                # A failed evaluation is not valid experiment evidence and must not
                # replace a prior successful result for this model. Return a compact
                # structured failure so the agent can reason about it without losing
                # the MCP session.
                return {
                    "status": "failed",
                    "session_id": self.session_id,
                    "session_step": self.session_step,
                    "model_step": evaluated_model_step,
                    "evaluated_at": time.time(),
                    "elapsed_seconds": elapsed,
                    "cfg2": self._snapshot_tunable_config(self.cfg),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }

            if isinstance(result, Mapping) and result.get("status") != "completed":
                return {
                    "status": "failed",
                    "session_id": self.session_id,
                    "session_step": self.session_step,
                    "model_step": evaluated_model_step,
                    "evaluated_at": time.time(),
                    "elapsed_seconds": elapsed,
                    "cfg2": self._snapshot_tunable_config(self.cfg),
                    "error": {
                        "type": "DiagnosticRuntimeError",
                        "message": (
                            "HDBSCAN evaluator did not complete successfully: "
                            + str(result.get("status"))
                        ),
                    },
                }

            if not isinstance(result, Mapping):
                return {
                    "status": "failed",
                    "session_id": self.session_id,
                    "session_step": self.session_step,
                    "model_step": evaluated_model_step,
                    "evaluated_at": time.time(),
                    "elapsed_seconds": elapsed,
                    "cfg2": self._snapshot_tunable_config(self.cfg),
                    "error": {
                        "type": "DiagnosticRuntimeError",
                        "message": "HDBSCAN evaluator returned a non-mapping result.",
                    },
                }

            enriched = copy.deepcopy(dict(result))
            enriched.update({
                "session_id": self.session_id,
                "session_step": self.session_step,
                "model_step": evaluated_model_step,
                "evaluated_at": time.time(),
                "elapsed_seconds": elapsed,
                "cfg2": self._snapshot_tunable_config(self.cfg),
            })
            self.current_evaluation = copy.deepcopy(enriched)
            return enriched

    def get_context(self) -> Dict[str, Any]:
        """Return the current experiment, model, diagnostics, and evidence context."""
        with self._lock:
            self._require_initialized()
            assert self.cfg is not None
            assert self.structure is not None
            assert self.probe is not None

            enabled_modules = []
            for module in self.structure.get("modules", []):
                diagnostics = module.get("diagnostics", {})
                if not diagnostics.get("enabled", False):
                    continue
                enabled_modules.append(
                    {
                        "module_path": module.get("module_path"),
                        "class_name": module.get("class_name"),
                        "attributes": copy.deepcopy(module.get("attributes", {})),
                        "parameters": copy.deepcopy(module.get("parameters", {})),
                        "diagnostics": copy.deepcopy(diagnostics),
                    }
                )

            stage_configs = self.probe.list_stages()
            categories = (
                "activation",
                "gradient",
                "parameter",
                "update",
                "backward_flow",
            )
            active_categories = []
            inactive_categories = []
            for category in categories:
                active = any(
                    bool(stage.get("config", {}).get("enabled", False))
                    and bool(stage.get("config", {}).get(category, False))
                    for stage in stage_configs.values()
                )
                (active_categories if active else inactive_categories).append(category)

            specs = self.registry.list_metrics()

            return {
                "session": self._session_summary(),
                "model_structure": copy.deepcopy(self.structure),
                "training_config": self._snapshot_config(self.cfg),
                "experiment_control": {
                    **self.get_tunable_parameters(),
                    "host_fixed_overrides": copy.deepcopy(self._config_overrides),
                    "source_code_modification_allowed": False,
                    "fixed_cfg_overrides_allowed": False,
                    "evaluation_tool_available": True,
                    "evaluation_tool": "evaluate_hdbscan",
                },
                "training_data": {
                    "records": self._training_record_count,
                    "transactions": self._transaction_count,
                },
                "diagnostics": {
                    "config_path": (
                        str(self.diagnostic_config_path)
                        if self.diagnostic_config_path is not None
                        else None
                    ),
                    "config_path_source": "CFG.diagnostic_config_path",
                    "structure_id": self.structure.get("structure_id"),
                    "structure_id_validation": False,
                    "probe_policy": copy.deepcopy(
                        self.structure.get("diagnostic_policy_report", {})
                    ),
                    "enabled_modules": enabled_modules,
                    "enabled_stage_count": len(stage_configs),
                    "effective_stage_configs": copy.deepcopy(stage_configs),
                    "active_module_metric_categories": active_categories,
                    "inactive_module_metric_categories": inactive_categories,
                },
                "registry": {
                    "stages_with_recorded_metrics": self.registry.list_stages(),
                    "recorded_metric_spec_count": len(specs),
                },
                "comparison": {
                    "comparison_ready": self.previous_session is not None,
                    "previous_session_available": self.previous_session is not None,
                    "previous_metrics_available": self.previous_metrics is not None,
                    "previous_structure_available": self.previous_structure is not None,
                    "current_evaluation_available": self.current_evaluation is not None,
                    "previous_evaluation_available": self.previous_evaluation is not None,
                },
                "evaluation": {
                    "current_available": self.current_evaluation is not None,
                    "current_is_latest_model": self.current_evaluation is not None,
                    "current_summary": (
                        {
                            "model_step": self.current_evaluation.get("model_step"),
                            "evaluated_at": self.current_evaluation.get("evaluated_at"),
                            "centroid_spread": copy.deepcopy(
                                self.current_evaluation.get("centroid_spread", {})
                            ),
                            "pattern_count": len(self.current_evaluation.get("per_pattern", [])),
                        }
                        if isinstance(self.current_evaluation, Mapping)
                        else None
                    ),
                    "previous_available": self.previous_evaluation is not None,
                },
            }

    def train_segment(self, steps: int) -> Dict[str, Any]:
        """Continue the current model/optimizer for a bounded number of steps.

        Segment ids are reserved before execution so a failed/partial segment is
        never confused with a later retry. Failed segments remain visible through
        get_context().session.last_segment.
        """
        with self._lock:
            self._require_initialized()
            self._validate_segment_steps(steps)

            assert self.cfg is not None
            assert self.model is not None
            assert self.optimizer is not None
            assert self.probe is not None
            assert self.train_data is not None
            assert self.txn_index is not None

            segment_id = self.segment_count + 1
            self.segment_count = segment_id  # reserve even when this segment fails
            start_step = self.session_step
            started = time.perf_counter()

            loss_values: List[float] = []
            recon_values: List[float] = []
            sw_ce_values: List[float] = []
            component_values: Dict[str, List[float]] = {
                "v": [], "sw": [], "amt": [], "is_new": []
            }
            grad_norm_values: List[float] = []
            clip_count = 0

            self.model.train()

            try:
                for _ in range(steps):
                    diagnostic_step = self.session_step
                    optimizer_stepped = False

                    try:
                        with self.probe.step(
                            diagnostic_step,
                            phase="train",
                            tags={"segment_id": segment_id},
                        ):
                            self.optimizer.zero_grad(set_to_none=True)

                            with redirect_stdout(sys.stderr):
                                batch, _, _ = batch_util.sample_txn_batch(
                                    self.train_data,
                                    self.txn_index,
                                    self.cfg.txn_batch_size,
                                )
                                x_full, raw = train_driver.build_full_features(
                                    batch,
                                    self.cfg,
                                    is_training=True,
                                )

                                mask = raw["mask"]
                                fwd_out = self.model.encode(x_full, mask)
                                predicts = self.model.predict_nextstep(fwd_out)
                                l_recon, loss_detail = train_driver.compute_nextstep_recon_loss(
                                    predicts,
                                    raw,
                                    self.cfg,
                                )

                            loss = self.cfg.lambda_recon * l_recon
                            loss_value = float(loss.item())
                            recon_value = float(l_recon.item())

                            # Backward-compatible with the original (total, sw_ce_float)
                            # return and the newer (total, loss_components_dict) contract.
                            component_metrics: Dict[str, float] = {}
                            if isinstance(loss_detail, Mapping):
                                if "sw_ce" not in loss_detail:
                                    raise DiagnosticRuntimeError(
                                        "compute_nextstep_recon_loss returned a mapping "
                                        "without required 'sw_ce'"
                                    )
                                sw_ce_value = float(loss_detail["sw_ce"])
                                for name in ("v", "sw", "amt", "is_new"):
                                    value = loss_detail.get(name)
                                    if value is not None:
                                        component_metrics[f"loss.{name}"] = float(value)
                            else:
                                sw_ce_value = float(loss_detail)

                            training_metrics: Dict[str, Any] = {
                                "loss.total": loss_value,
                                "loss.recon": recon_value,
                                "loss.sw_ce": sw_ce_value,
                                "loss.nonfinite": not np.isfinite(loss_value),
                                "optimizer.lr": self.optimizer.param_groups[0]["lr"],
                            }
                            training_metrics.update(component_metrics)
                            self.probe.store_many("training", training_metrics)

                            loss.backward()
                            self.probe.after_backward()

                            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.cfg.clip_val,
                            )
                            grad_norm_value = float(grad_norm_pre_clip)
                            grad_nonfinite = not np.isfinite(grad_norm_value)
                            clip_triggered = (
                                np.isfinite(grad_norm_value)
                                and grad_norm_value > self.cfg.clip_val
                            )

                            self.probe.store_many(
                                "training",
                                {
                                    "grad.global.l2_pre_clip": grad_norm_value,
                                    "grad.global.nonfinite": grad_nonfinite,
                                    "grad.clip.max_norm": self.cfg.clip_val,
                                    "grad.clip.triggered": clip_triggered,
                                },
                            )

                            self.probe.before_optimizer_step()
                            self.optimizer.step()
                            optimizer_stepped = True
                            # Any successful parameter update makes a prior downstream
                            # evaluation stale. Do not expose it as evidence for the
                            # newly updated model.
                            self.current_evaluation = None
                            self.probe.store(
                                "training",
                                "optimizer.step_applied",
                                True,
                            )

                            self.session_step += 1
                            self.model_step += 1
                            self.probe.after_optimizer_step()

                            loss_values.append(loss_value)
                            recon_values.append(recon_value)
                            sw_ce_values.append(sw_ce_value)
                            for name in ("v", "sw", "amt", "is_new"):
                                metric_name = f"loss.{name}"
                                if metric_name in component_metrics:
                                    component_values[name].append(component_metrics[metric_name])
                            grad_norm_values.append(grad_norm_value)
                            clip_count += int(clip_triggered)

                    except Exception:
                        if not optimizer_stepped:
                            self.optimizer.zero_grad(set_to_none=True)
                            # Partial metrics may already exist at this step.
                            # Mark explicitly that no parameter update happened.
                            try:
                                self.registry.store(
                                    "training",
                                    "optimizer.step_applied",
                                    False,
                                    step=diagnostic_step,
                                    phase="train",
                                    tags={"segment_id": segment_id},
                                )
                            except Exception:
                                pass
                        raise

            except Exception as exc:
                elapsed = time.perf_counter() - started
                self._last_segment = {
                    "status": "failed",
                    "segment_id": segment_id,
                    "start_step": start_step,
                    "end_step_exclusive": self.session_step,
                    "steps_requested": steps,
                    "steps_completed": self.session_step - start_step,
                    "elapsed_seconds": elapsed,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "loss": self._summary_stats(loss_values),
                    "loss_recon": self._summary_stats(recon_values),
                    "loss_sw_ce": self._summary_stats(sw_ce_values),
                    "loss_components": {
                        name: self._summary_stats(values)
                        for name, values in component_values.items()
                    },
                    "grad_global_l2_pre_clip": self._summary_stats(grad_norm_values),
                    "clip": {
                        "triggered_steps": clip_count,
                        "fraction": (
                            clip_count / len(grad_norm_values)
                            if grad_norm_values
                            else 0.0
                        ),
                        "max_norm": self.cfg.clip_val,
                    },
                }
                raise

            elapsed = time.perf_counter() - started
            summary = {
                "status": "completed",
                "segment_id": segment_id,
                "start_step": start_step,
                "end_step_exclusive": self.session_step,
                "steps_requested": steps,
                "steps_completed": self.session_step - start_step,
                "elapsed_seconds": elapsed,
                "loss": self._summary_stats(loss_values),
                "loss_recon": self._summary_stats(recon_values),
                "loss_sw_ce": self._summary_stats(sw_ce_values),
                "loss_components": {
                    name: self._summary_stats(values)
                    for name, values in component_values.items()
                },
                "grad_global_l2_pre_clip": self._summary_stats(grad_norm_values),
                "clip": {
                    "triggered_steps": clip_count,
                    "fraction": clip_count / len(grad_norm_values) if grad_norm_values else 0.0,
                    "max_norm": self.cfg.clip_val,
                },
            }
            self._last_segment = copy.deepcopy(summary)
            return summary

    def list_metrics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """List metrics that have actually been registered in this session."""
        with self._lock:
            self._require_initialized()
            specs = self.registry.list_metrics(stage=stage)
            return {
                "stage_filter": stage,
                "count": len(specs),
                "metrics": specs,
            }

    def query_metrics(self, queries: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """Execute bounded current/history metric queries in one tool call."""
        with self._lock:
            self._require_initialized()
            self._validate_metric_query_batch(queries)
            results = [
                self._execute_metric_query(index, raw_query)
                for index, raw_query in enumerate(queries)
            ]
            return {"query_count": len(results), "results": results}

    def list_previous_metrics(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """List metric specs available in the previous-session snapshot."""
        with self._lock:
            if self.previous_metrics is None:
                return {"available": False, "stage_filter": stage, "count": 0, "metrics": []}

            specs = []
            for entry in self.previous_metrics.get("metrics", []):
                if stage is not None and entry.get("stage") != stage:
                    continue
                spec = entry.get("spec")
                if isinstance(spec, Mapping):
                    specs.append(copy.deepcopy(dict(spec)))
                else:
                    specs.append({
                        "stage": entry.get("stage"),
                        "metric_name": entry.get("metric_name"),
                    })
            return {
                "available": True,
                "stage_filter": stage,
                "count": len(specs),
                "metrics": specs,
            }

    def query_previous_metrics(self, queries: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """Query the previous metric snapshot with the same bounded API as current metrics."""
        with self._lock:
            self._validate_metric_query_batch(queries)
            if self.previous_metrics is None:
                return {
                    "available": False,
                    "query_count": len(queries),
                    "results": [],
                }
            results = [
                self._execute_snapshot_metric_query(index, raw_query)
                for index, raw_query in enumerate(queries)
            ]
            return {
                "available": True,
                "session": copy.deepcopy(self.previous_session),
                "query_count": len(results),
                "results": results,
            }

    def get_experiment_comparison(self) -> Dict[str, Any]:
        """Return a compact A/B comparison envelope for current vs previous run."""
        with self._lock:
            self._require_initialized()
            current_cfg2 = self._snapshot_tunable_config(self.cfg)
            if self.previous_session is None or self.previous_cfg2 is None:
                return {
                    "comparison_ready": False,
                    "reason": (
                        "No previous experiment is available. A comparison becomes ready "
                        "only after restart_session() successfully replaces an active session."
                    ),
                    "current_session": self._session_summary(),
                    "current_cfg2": current_cfg2,
                }

            previous_steps = int(self.previous_session.get("session_step") or 0)
            current_steps = int(self.session_step)
            common_steps = min(previous_steps, current_steps)
            warnings = []
            if previous_steps != current_steps:
                warnings.append(
                    "Runs have different completed-step counts; compare metrics only "
                    "inside matched_metric_window."
                )

            changed_cfg2 = self._diff_cfg2(self.previous_cfg2, current_cfg2)
            if len(changed_cfg2) > 1:
                warnings.append(
                    "More than one tunable changed; causal attribution to any single "
                    "parameter is weaker."
                )
            if common_steps == 0:
                warnings.append("No matched completed training steps are available yet.")

            return {
                "comparison_ready": True,
                "previous_session": copy.deepcopy(self.previous_session),
                "current_session": self._session_summary(),
                "previous_cfg2": copy.deepcopy(self.previous_cfg2),
                "current_cfg2": current_cfg2,
                "changed_cfg2": changed_cfg2,
                "matched_metric_window": {
                    "common_completed_steps": common_steps,
                    "since_step": 0 if common_steps > 0 else None,
                    "until_step": common_steps - 1 if common_steps > 0 else None,
                },
                "structure": {
                    "previous_structure_id": (
                        self.previous_structure.get("structure_id")
                        if isinstance(self.previous_structure, Mapping)
                        else None
                    ),
                    "current_structure_id": (
                        self.structure.get("structure_id")
                        if isinstance(self.structure, Mapping)
                        else None
                    ),
                    "previous_module_count": self._structure_module_count(self.previous_structure),
                    "current_module_count": self._structure_module_count(self.structure),
                    "structure_id_is_informational_only": True,
                },
                "evaluation": self._evaluation_comparison(),
                "warnings": warnings,
            }

    def close(self) -> None:
        """Release hooks and discard all in-process diagnostic state."""
        with self._lock:
            self._terminate_current_session()
            self.previous_cfg2 = None
            self.previous_metrics = None
            self.previous_session = None
            self.previous_structure = None
            self.previous_evaluation = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_effective_cfg(
        self,
        *,
        restart_overrides: Optional[Mapping[str, Any]],
    ) -> CFG2:
        cfg = CFG2()
        self._apply_config_overrides(cfg)
        if restart_overrides:
            self._apply_restart_overrides(cfg, restart_overrides)
        self._validate_runtime_config(cfg)
        return cfg

    def _build_session_candidate(self, cfg: CFG2) -> Dict[str, Any]:
        """Construct a complete session without mutating the active runtime."""
        config_path = Path(cfg.diagnostic_config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Diagnostic config not found: {config_path}")

        registry = DiagnosticRegistry()
        probe = None
        try:
            train_driver.set_seed(cfg.seed)
            model = train_driver.init_model(cfg)

            with redirect_stdout(sys.stderr):
                probe = self._build_probe_for_model(
                    model,
                    registry=registry,
                    config_path=config_path,
                )

            train_driver.set_seed(cfg.seed)
            with redirect_stdout(sys.stderr):
                train_data = seq_gen.make_anchored_dataset(
                    current_cfg=cfg,
                    num_trx=cfg.num_trx
                )

            txn_index = train_driver.build_txn_index(train_data)
            if not txn_index:
                raise DiagnosticRuntimeError(
                    "Generated training dataset contains no transactions"
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            model.train()

            structure = copy.deepcopy(probe.model_structure)
            if not isinstance(structure, dict):
                raise DiagnosticRuntimeError(
                    "Diagnostic probe did not retain the loaded model structure"
                )

            return {
                "cfg": cfg,
                "diagnostic_config_path": config_path,
                "registry": registry,
                "model": model,
                "optimizer": optimizer,
                "probe": probe,
                "train_data": train_data,
                "txn_index": txn_index,
                "structure": structure,
                "training_record_count": self._infer_record_count(train_data),
                "transaction_count": len(txn_index),
            }
        except Exception:
            if probe is not None:
                try:
                    probe.stop()
                except Exception:
                    pass
            registry.clear(reset_specs=True)
            raise

    def _commit_session_candidate(self, candidate: Mapping[str, Any]) -> None:
        self.cfg = candidate["cfg"]
        self.diagnostic_config_path = candidate["diagnostic_config_path"]
        self.registry = candidate["registry"]
        self.model = candidate["model"]
        self.optimizer = candidate["optimizer"]
        self.probe = candidate["probe"]
        self.train_data = candidate["train_data"]
        self.txn_index = candidate["txn_index"]
        self.structure = candidate["structure"]

        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.session_step = 0
        self.model_step = 0
        self.segment_count = 0
        self._last_segment = None
        self.current_evaluation = None
        self._training_record_count = candidate["training_record_count"]
        self._transaction_count = candidate["transaction_count"]
        self.initialized = True

    def _build_probe_for_model(
        self,
        model: torch.nn.Module,
        *,
        registry: DiagnosticRegistry,
        config_path: Path,
    ) -> DiagnosticProbeManager:
        """Apply diagnostic JSON as policy to the freshly rendered model.

        structure_id is informational only. Exact module paths inherit policy.
        New indexed siblings such as fwd_stack.2 inherit a common sibling policy
        when existing indexed sibling policies agree. Stale paths are reported but
        do not block a structure experiment.
        """
        with config_path.open("r", encoding="utf-8") as f:
            policy = json.load(f)
        if not isinstance(policy, Mapping):
            raise DiagnosticRuntimeError("Diagnostic config must contain a JSON object")
        policy_modules = policy.get("modules")
        if not isinstance(policy_modules, list):
            raise DiagnosticRuntimeError("Diagnostic config must contain a 'modules' list")

        exact_policy: Dict[str, Dict[str, Any]] = {}
        indexed_policy: Dict[str, List[Dict[str, Any]]] = {}

        for module in policy_modules:
            if not isinstance(module, Mapping):
                continue
            path = module.get("module_path")
            diagnostics = module.get("diagnostics")
            if not isinstance(path, str) or not isinstance(diagnostics, Mapping):
                continue
            diag = copy.deepcopy(dict(diagnostics))
            exact_policy[path] = diag

            match = re.match(r"^(.*)\.(\d+)$", path)
            if match:
                indexed_policy.setdefault(match.group(1), []).append(diag)

        structure = render_model_structure(model)
        current_paths = {
            module.get("module_path")
            for module in structure.get("modules", [])
            if isinstance(module, Mapping) and isinstance(module.get("module_path"), str)
        }

        exact_matches: List[str] = []
        propagated_matches: List[str] = []
        ambiguous_indexed_prefixes: set[str] = set()

        for module in structure.get("modules", []):
            if not isinstance(module, Mapping):
                continue
            path = module.get("module_path")
            if not isinstance(path, str):
                continue

            if path in exact_policy:
                module["diagnostics"] = copy.deepcopy(exact_policy[path])
                exact_matches.append(path)
                continue

            match = re.match(r"^(.*)\.(\d+)$", path)
            if not match:
                continue
            prefix = match.group(1)
            templates = indexed_policy.get(prefix, [])
            if not templates:
                continue
            first = templates[0]
            if all(template == first for template in templates[1:]):
                module["diagnostics"] = copy.deepcopy(first)
                propagated_matches.append(path)
            else:
                ambiguous_indexed_prefixes.add(prefix)

        stale_enabled_paths = sorted(
            path
            for path, diag in exact_policy.items()
            if bool(diag.get("enabled", False)) and path not in current_paths
        )

        structure["diagnostic_policy_report"] = {
            "source_path": str(config_path),
            "structure_id_validation": False,
            "exact_match_count": len(exact_matches),
            "propagated_paths": sorted(propagated_matches),
            "stale_enabled_paths": stale_enabled_paths,
            "ambiguous_indexed_prefixes": sorted(ambiguous_indexed_prefixes),
        }

        probe = DiagnosticProbeManager(model, registry)
        probe.load_structure_config(structure, validate_structure=False)
        probe.start()
        return probe

    def _terminate_current_session(self) -> None:
        if self.probe is not None:
            try:
                self.probe.stop()
            except Exception:
                pass
        if self.registry is not None:
            self.registry.clear(reset_specs=True)
        self._reset_state()
        self.registry = DiagnosticRegistry()

    @staticmethod
    def _tunable_field_names() -> List[str]:
        """Return the authoritative agent-overridable CFG2 field names."""
        return list(CFG2.get_tunable_parameters().keys())

    @staticmethod
    def _config_value_type_name(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        return type(value).__name__

    @staticmethod
    def _validate_restart_overrides(overrides: Mapping[str, Any]) -> None:
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be an object/map")

        allowed = set(DiagnosticRuntime._tunable_field_names())
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                "Unsupported restart_session override(s): "
                + ", ".join(sorted(unknown))
                + ". Allowed CFG2 fields: "
                + ", ".join(sorted(allowed))
            )

        defaults = CFG2()
        for name, value in overrides.items():
            expected = getattr(defaults, name)
            if isinstance(expected, bool):
                valid = isinstance(value, bool)
                expected_name = "boolean"
            elif isinstance(expected, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
                expected_name = "integer"
            elif isinstance(expected, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                expected_name = "number"
            elif isinstance(expected, str):
                valid = isinstance(value, str)
                expected_name = "string"
            else:
                valid = isinstance(value, type(expected))
                expected_name = type(expected).__name__

            if not valid:
                raise TypeError(
                    f"Override {name!r} must be {expected_name}; "
                    f"got {type(value).__name__}"
                )

    @staticmethod
    def _apply_restart_overrides(cfg: CFG2, overrides: Mapping[str, Any]) -> None:
        for name, value in overrides.items():
            # Preserve float defaults as floats even if an MCP client sends 1.
            default = getattr(CFG2(), name)
            if isinstance(default, float) and isinstance(value, int):
                value = float(value)
            setattr(cfg, name, value)

    @classmethod
    def _snapshot_tunable_config(cls, cfg: CFG2) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in cls._tunable_field_names():
            value = getattr(cfg, name)
            safe, supported = cls._to_json_safe(value)
            if supported:
                result[name] = safe
        return result

    @staticmethod
    def _diff_cfg2(
        previous: Optional[Mapping[str, Any]],
        current: Optional[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(current, Mapping):
            return {}
        previous = previous if isinstance(previous, Mapping) else {}
        changed: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(previous) | set(current)):
            before = previous.get(name)
            after = current.get(name)
            if before != after:
                changed[name] = {"previous": before, "current": after}
        return changed

    @staticmethod
    def _structure_module_count(structure: Optional[Mapping[str, Any]]) -> Optional[int]:
        if not isinstance(structure, Mapping):
            return None
        model = structure.get("model")
        if isinstance(model, Mapping) and isinstance(model.get("module_count"), int):
            return model.get("module_count")
        modules = structure.get("modules")
        return len(modules) if isinstance(modules, list) else None

    @classmethod
    def _validate_metric_query_batch(cls, queries: List[Mapping[str, Any]]) -> None:
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list")
        if len(queries) > cls.MAX_METRIC_QUERIES:
            raise ValueError(
                f"At most {cls.MAX_METRIC_QUERIES} metric queries are allowed per call"
            )
        for index, raw_query in enumerate(queries):
            if not isinstance(raw_query, Mapping):
                raise TypeError(f"queries[{index}] must be an object")

    def _snapshot_metric_entry(
        self,
        stage: str,
        metric_name: str,
    ) -> Optional[Mapping[str, Any]]:
        if self.previous_metrics is None:
            return None
        for entry in self.previous_metrics.get("metrics", []):
            if entry.get("stage") == stage and entry.get("metric_name") == metric_name:
                return entry
        return None

    def _execute_snapshot_metric_query(
        self,
        index: int,
        raw_query: Mapping[str, Any],
    ) -> Dict[str, Any]:
        stage = raw_query.get("stage")
        metric_name = raw_query.get("metric_name")
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"queries[{index}].stage must be a non-empty string")
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(f"queries[{index}].metric_name must be a non-empty string")

        history = raw_query.get("history", False)
        if not isinstance(history, bool):
            raise TypeError(f"queries[{index}].history must be boolean")
        phase = raw_query.get("phase")
        if phase is not None and not isinstance(phase, str):
            raise TypeError(f"queries[{index}].phase must be string or null")

        entry = self._snapshot_metric_entry(stage, metric_name)
        query_echo: Dict[str, Any] = {
            "stage": stage,
            "metric_name": metric_name,
            "history": history,
        }
        if phase is not None:
            query_echo["phase"] = phase
        if entry is None:
            return {
                "query": query_echo,
                "found": False,
                "records": [] if history else None,
            }

        records = [copy.deepcopy(record) for record in entry.get("records", [])]
        if phase is not None:
            records = [record for record in records if record.get("phase") == phase]

        if not history:
            record = records[-1] if records else None
            return {"query": query_echo, "found": record is not None, "record": record}

        last_n = raw_query.get("last_n")
        if last_n is None:
            last_n = self.DEFAULT_HISTORY_RECORDS
        if not isinstance(last_n, int) or isinstance(last_n, bool) or last_n <= 0:
            raise ValueError(f"queries[{index}].last_n must be a positive integer")
        if last_n > self.MAX_HISTORY_RECORDS_PER_QUERY:
            raise ValueError(
                f"queries[{index}].last_n cannot exceed {self.MAX_HISTORY_RECORDS_PER_QUERY}"
            )
        query_echo["last_n"] = last_n

        for name in ("since_step", "until_step", "since_timestamp", "until_timestamp"):
            value = raw_query.get(name)
            if value is None:
                continue
            if name.endswith("_step"):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"queries[{index}].{name} must be an integer")
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"queries[{index}].{name} must be numeric")
            query_echo[name] = value

        def include(record: Mapping[str, Any]) -> bool:
            step = record.get("step")
            ts = record.get("timestamp")
            since_step = raw_query.get("since_step")
            until_step = raw_query.get("until_step")
            since_ts = raw_query.get("since_timestamp")
            until_ts = raw_query.get("until_timestamp")
            if since_step is not None and step is not None and step < since_step:
                return False
            if until_step is not None and step is not None and step > until_step:
                return False
            if since_ts is not None and ts is not None and ts < since_ts:
                return False
            if until_ts is not None and ts is not None and ts > until_ts:
                return False
            return True

        records = [record for record in records if include(record)]
        if len(records) > last_n:
            records = records[-last_n:]

        return {
            "query": query_echo,
            "found": bool(records),
            "record_count": len(records),
            "records": records,
        }



    def _execute_metric_query(
        self,
        index: int,
        raw_query: Mapping[str, Any],
    ) -> Dict[str, Any]:
        stage = raw_query.get("stage")
        metric_name = raw_query.get("metric_name")

        if not isinstance(stage, str) or not stage:
            raise ValueError(f"queries[{index}].stage must be a non-empty string")
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError(
                f"queries[{index}].metric_name must be a non-empty string"
            )

        history = raw_query.get("history", False)
        if not isinstance(history, bool):
            raise TypeError(f"queries[{index}].history must be boolean")

        phase = raw_query.get("phase")
        if phase is not None and not isinstance(phase, str):
            raise TypeError(f"queries[{index}].phase must be string or null")

        query_echo: Dict[str, Any] = {
            "stage": stage,
            "metric_name": metric_name,
            "history": history,
        }
        if phase is not None:
            query_echo["phase"] = phase

        if not history:
            record = self.registry.get_current(
                stage,
                metric_name,
                phase=phase,
            )
            return {
                "query": query_echo,
                "found": record is not None,
                "record": record,
            }

        kwargs: Dict[str, Any] = {"phase": phase}

        last_n = raw_query.get("last_n")
        if last_n is None:
            last_n = self.DEFAULT_HISTORY_RECORDS
        if not isinstance(last_n, int) or isinstance(last_n, bool) or last_n <= 0:
            raise ValueError(
                f"queries[{index}].last_n must be a positive integer"
            )
        if last_n > self.MAX_HISTORY_RECORDS_PER_QUERY:
            raise ValueError(
                f"queries[{index}].last_n cannot exceed "
                f"{self.MAX_HISTORY_RECORDS_PER_QUERY}"
            )
        kwargs["last_n"] = last_n
        query_echo["last_n"] = last_n

        for name in (
            "since_step",
            "until_step",
            "since_timestamp",
            "until_timestamp",
        ):
            value = raw_query.get(name)
            if value is None:
                continue
            if name.endswith("_step"):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"queries[{index}].{name} must be an integer")
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"queries[{index}].{name} must be numeric")
            kwargs[name] = value
            query_echo[name] = value

        records = self.registry.get_history(
            stage,
            metric_name,
            **kwargs,
        )
        return {
            "query": query_echo,
            "found": bool(records),
            "record_count": len(records),
            "records": records,
        }

    def _evaluation_comparison(self) -> Dict[str, Any]:
        current = self.current_evaluation
        previous = self.previous_evaluation
        if not isinstance(current, Mapping) or not isinstance(previous, Mapping):
            return {
                "comparison_ready": False,
                "current_available": isinstance(current, Mapping),
                "previous_available": isinstance(previous, Mapping),
                "reason": "Both current and previous sessions must be evaluated with evaluate_hdbscan.",
            }

        current_step = current.get("model_step")
        previous_step = previous.get("model_step")
        previous_final_step = (
            self.previous_session.get("model_step")
            if isinstance(self.previous_session, Mapping)
            else None
        )
        if current_step != self.model_step or previous_step != previous_final_step:
            return {
                "comparison_ready": False,
                "current_available": True,
                "previous_available": True,
                "reason": (
                    "HDBSCAN evaluation is stale for one of the compared models. "
                    "Evaluate the active model after its final training segment and "
                    "evaluate the prior model before restart_session()."
                ),
                "current_evaluated_model_step": current_step,
                "current_model_step": self.model_step,
                "previous_evaluated_model_step": previous_step,
                "previous_final_model_step": previous_final_step,
            }

        current_metrics = current.get("centroid_spread", {})
        previous_metrics = previous.get("centroid_spread", {})
        metric_names = (
            "roc_auc", "pr_auc", "best_f1", "precision", "recall",
            "accuracy", "fraud_detection_rate", "benign_fp_rate",
        )
        deltas: Dict[str, Any] = {}
        for name in metric_names:
            before = previous_metrics.get(name) if isinstance(previous_metrics, Mapping) else None
            after = current_metrics.get(name) if isinstance(current_metrics, Mapping) else None
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                deltas[name] = {
                    "previous": float(before),
                    "current": float(after),
                    "delta": float(after) - float(before),
                }

        previous_patterns = {
            item.get("scenario"): item
            for item in previous.get("per_pattern", [])
            if isinstance(item, Mapping) and isinstance(item.get("scenario"), str)
        }
        current_patterns = {
            item.get("scenario"): item
            for item in current.get("per_pattern", [])
            if isinstance(item, Mapping) and isinstance(item.get("scenario"), str)
        }
        pattern_deltas: List[Dict[str, Any]] = []
        for scenario in sorted(set(previous_patterns) | set(current_patterns)):
            before = previous_patterns.get(scenario, {})
            after = current_patterns.get(scenario, {})
            item: Dict[str, Any] = {
                "scenario": scenario,
                "is_fraud": after.get("is_fraud", before.get("is_fraud")),
                "previous_n": before.get("n"),
                "current_n": after.get("n"),
            }
            for metric_name in ("mean_centroid_spread", "flag_rate"):
                before_value = before.get(metric_name)
                after_value = after.get(metric_name)
                if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                    item[metric_name] = {
                        "previous": float(before_value),
                        "current": float(after_value),
                        "delta": float(after_value) - float(before_value),
                    }
            pattern_deltas.append(item)

        return {
            "comparison_ready": True,
            "previous_model_step": previous.get("model_step"),
            "current_model_step": current.get("model_step"),
            "metric_deltas": deltas,
            "per_pattern_deltas": pattern_deltas,
        }

    def _session_summary(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "session_step": self.session_step,
            "model_step": self.model_step,
            "segment_count": self.segment_count,
            "device": getattr(self.cfg, "device", None) if self.cfg is not None else None,
            "optimizer": type(self.optimizer).__name__ if self.optimizer is not None else None,
            "learning_rate": (
                self.optimizer.param_groups[0]["lr"]
                if self.optimizer is not None
                else None
            ),
            "training_records": self._training_record_count,
            "training_transactions": self._transaction_count,
            "last_segment": copy.deepcopy(self._last_segment),
            "hdbscan_evaluation_available": self.current_evaluation is not None,
            "hdbscan_evaluated_model_step": (
                self.current_evaluation.get("model_step")
                if isinstance(self.current_evaluation, Mapping)
                else None
            ),
            "cfg2": (
                self._snapshot_tunable_config(self.cfg)
                if self.cfg is not None
                else None
            ),
        }

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise DiagnosticRuntimeError(
                "Diagnostic session is not initialized. Call initialize_session first."
            )

    @classmethod
    def _validate_segment_steps(cls, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise TypeError("steps must be an integer")
        if steps < 1 or steps > cls.MAX_STEPS_PER_SEGMENT:
            raise ValueError(
                f"steps must be between 1 and {cls.MAX_STEPS_PER_SEGMENT}"
            )

    @staticmethod
    def _validate_config_overrides(overrides: Mapping[str, Any]) -> None:
        allowed = {"device", "num_trx", "seed"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                "Unsupported DiagnosticRuntime config override(s): "
                + ", ".join(sorted(unknown))
            )

    def _apply_config_overrides(self, cfg: CFG2) -> None:
        for name, value in self._config_overrides.items():
            setattr(cfg, name, value)

    @staticmethod
    def _validate_runtime_config(cfg: CFG2) -> None:
        if not isinstance(cfg.num_trx, int) or cfg.num_trx <= 0:
            raise ValueError("cfg.num_trx must be a positive integer")
        if not isinstance(cfg.txn_batch_size, int) or isinstance(cfg.txn_batch_size, bool) or cfg.txn_batch_size <= 0:
            raise ValueError("cfg.txn_batch_size must be a positive integer")
        if not isinstance(cfg.num_layers, int) or isinstance(cfg.num_layers, bool) or cfg.num_layers <= 0:
            raise ValueError("cfg.num_layers must be a positive integer")
        if not isinstance(cfg.hidden_dim, int) or isinstance(cfg.hidden_dim, bool) or cfg.hidden_dim <= 0:
            raise ValueError("cfg.hidden_dim must be a positive integer")
        for name in ("lr", "clip_val", "tau_sw"):
            value = getattr(cfg, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"cfg.{name} must be a positive finite number")
        if not isinstance(cfg.dropout, (int, float)) or isinstance(cfg.dropout, bool) or not np.isfinite(cfg.dropout) or not (0.0 <= cfg.dropout < 1.0):
            raise ValueError("cfg.dropout must be a finite number in [0, 1)")
        if not isinstance(cfg.seed, int):
            raise TypeError("cfg.seed must be an integer")
        if not isinstance(cfg.device, str) or not cfg.device:
            raise ValueError("cfg.device must be a non-empty string")
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                f"cfg.device={cfg.device!r} requested CUDA, but CUDA is unavailable"
            )

    @staticmethod
    def _infer_record_count(train_data: Mapping[str, Any]) -> int:
        dt = train_data.get("dt")
        return len(dt) if dt is not None else 0

    @staticmethod
    def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "first": None,
                "last": None,
                "min": None,
                "max": None,
                "mean": None,
            }
        return {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    @classmethod
    def _snapshot_config(cls, cfg: CFG2) -> Dict[str, Any]:
        names = set(vars(cfg))
        for base in type(cfg).mro():
            names.update(
                name
                for name in vars(base)
                if not name.startswith("_")
            )

        result: Dict[str, Any] = {}
        for name in sorted(names):
            if name.startswith("_"):
                continue
            try:
                value = getattr(cfg, name)
            except Exception:
                continue
            if callable(value):
                continue
            safe, supported = cls._to_json_safe(value)
            if supported:
                result[name] = safe
        return result

    @classmethod
    def _to_json_safe(cls, value: Any) -> tuple[Any, bool]:
        if value is None or isinstance(value, (str, bool, int)):
            return value, True
        if isinstance(value, float):
            if np.isfinite(value):
                return value, True
            if np.isnan(value):
                return "NaN", True
            return ("Infinity" if value > 0 else "-Infinity"), True
        if isinstance(value, np.generic):
            return cls._to_json_safe(value.item())
        if isinstance(value, torch.device):
            return str(value), True
        if isinstance(value, (list, tuple)):
            converted = []
            for item in value:
                safe, supported = cls._to_json_safe(item)
                if not supported:
                    return None, False
                converted.append(safe)
            return converted, True
        if isinstance(value, Mapping):
            converted_dict: Dict[str, Any] = {}
            for key, item in value.items():
                safe, supported = cls._to_json_safe(item)
                if not supported:
                    return None, False
                converted_dict[str(key)] = safe
            return converted_dict, True
        return None, False

    def _reset_state(self) -> None:
        self.initialized = False
        self.session_id = None
        self.started_at = None
        self.session_step = 0
        self.model_step = 0
        self.segment_count = 0
        self.diagnostic_config_path = None
        self.cfg = None
        self.model = None
        self.optimizer = None
        self.probe = None
        self.train_data = None
        self.txn_index = None
        self.structure = None
        self._training_record_count = 0
        self._transaction_count = 0
        self._last_segment = None
        self.current_evaluation = None
