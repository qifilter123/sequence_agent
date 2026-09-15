from __future__ import annotations

import atexit
import copy
import math
from typing import Any, Dict, List, Optional

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover - depends on deployment environment
    raise RuntimeError(
        "The official MCP Python SDK v2 is required. "
        "Install it with: pip install 'mcp>=2,<3'"
    ) from exc

from model_diagnostic.diagnostic.diagnostic_runtime import DiagnosticRuntime


runtime = DiagnosticRuntime.from_environment()

# MCP-layer mirrors requested for agent comparison. Runtime owns the canonical
# snapshots; these mirrors are refreshed only after a successful restart.
PREVIOUS_CFG2: Optional[Dict[str, Any]] = None
PREVIOUS_METRICS: Optional[Dict[str, Any]] = None
PREVIOUS_STRUCTURE: Optional[Dict[str, Any]] = None
PREVIOUS_EVALUATION: Optional[Dict[str, Any]] = None


def _transport_safe(value: Any) -> Any:
    """Convert non-finite floats to strict JSON-safe diagnostic markers."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(k): _transport_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_transport_safe(v) for v in value]
    return value


mcp = MCPServer(
    "diagnostics",
    instructions=(
        "Stateful PyTorch training diagnostics with controlled experiments. "
        "CFG2.get_tunable_parameters() and the DAG path fields returned by "
        "get_tunable_parameters are the authoritative experiment boundary; call that "
        "tool before planning experiments. initialize_session starts "
        "the default CFG2 baseline and NEVER creates previous experiment evidence. "
        "Only restart_session may publish PREVIOUS_CFG2/PREVIOUS_METRICS/"
        "PREVIOUS_STRUCTURE/PREVIOUS_EVALUATION, and only after it successfully replaces an active "
        "session. Before that first successful replacement, there is no previous "
        "experiment and no valid A/B comparison. restart_session may change CFG2 "
        "tunables and/or select candidate YAML files for any executable DAG; it creates "
        "a fresh model, optimizer, dataset, probe set, and metric registry. For a "
        "structure experiment, first learn the baseline from the source-graph MCP, form "
        "one evidence-based hypothesis, copy the affected baseline DAG YAML to a candidate "
        "file, edit only that candidate YAML, and pass its path to restart_session. Do not "
        "query the source-graph MCP for candidate topology because it indexes the baseline; "
        "use the candidate YAML plus runtime build, probe, training, and evaluation evidence. "
        "Never modify fixed non-DAG CFG fields or Python source code. Prefer one changed DAG "
        "per experiment, preserve the baseline YAML, and do not accept a candidate that fails "
        "DAG validation or runtime construction. Use bounded "
        "current/previous metric queries and get_experiment_comparison for A/B evidence. "
        "Use evaluate_hdbscan for downstream embedding/anomaly evaluation of the active "
        "in-memory model; evaluate before restart if the result should become previous "
        "experiment evidence. Prefer changing the minimum number of tunables needed for "
        "a hypothesis. HDBSCAN evaluation is downstream evidence for this synthetic/project "
        "evaluation only; do not overclaim production generalization. structure_id is "
        "informational only."
    ),
)


@mcp.tool()
def get_tunable_parameters() -> Dict[str, Any]:
    """Return authoritative CFG2 tunables and executable DAG path fields."""
    return _transport_safe(runtime.get_tunable_parameters())


@mcp.tool()
def initialize_session() -> Dict[str, Any]:
    """Initialize the default baseline; never create previous experiment evidence."""
    global PREVIOUS_CFG2, PREVIOUS_METRICS, PREVIOUS_STRUCTURE, PREVIOUS_EVALUATION

    result = runtime.initialize_session()
    if not result.get("already_initialized", False):
        # A fresh initialize establishes a new baseline lineage. Clearing these
        # mirrors is not a comparison assignment; only successful restart writes
        # replaced-session evidence into them.
        PREVIOUS_CFG2 = None
        PREVIOUS_METRICS = None
        PREVIOUS_STRUCTURE = None
        PREVIOUS_EVALUATION = None
    return _transport_safe(result)


@mcp.tool()
def restart_session(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Start a fresh experiment from defaults plus CFG2/DAG-path overrides.

    PREVIOUS_* mirrors are refreshed only after runtime.restart_session returns
    successfully. A failed restart therefore cannot publish the terminated
    session as previous evidence.
    """
    global PREVIOUS_CFG2, PREVIOUS_METRICS, PREVIOUS_STRUCTURE, PREVIOUS_EVALUATION

    result = runtime.restart_session(overrides=overrides)

    # This code is reached only after the new session started successfully.
    previous_cfg = runtime.get_previous_cfg2()
    previous_metrics = runtime.get_previous_metrics()
    previous_structure = runtime.get_previous_structure()
    previous_evaluation = runtime.get_previous_evaluation()

    PREVIOUS_CFG2 = (
        copy.deepcopy(previous_cfg.get("cfg2"))
        if previous_cfg.get("available")
        else None
    )
    PREVIOUS_METRICS = (
        copy.deepcopy(previous_metrics.get("metrics"))
        if previous_metrics.get("available")
        else None
    )
    PREVIOUS_STRUCTURE = (
        copy.deepcopy(previous_structure.get("model_structure"))
        if previous_structure.get("available")
        else None
    )
    PREVIOUS_EVALUATION = (
        copy.deepcopy(previous_evaluation.get("evaluation"))
        if previous_evaluation.get("available")
        else None
    )
    return _transport_safe(result)


@mcp.tool()
def get_previous_cfg2() -> Dict[str, Any]:
    """Return previous CFG2 only after a successful experiment restart."""
    available = PREVIOUS_CFG2 is not None
    return _transport_safe(
        {
            "available": available,
            "reason": None if available else (
                "No previous experiment exists until restart_session() successfully "
                "replaces an active session."
            ),
            "cfg2": copy.deepcopy(PREVIOUS_CFG2),
            "session": copy.deepcopy(runtime.previous_session),
        }
    )


@mcp.tool()
def get_previous_metrics() -> Dict[str, Any]:
    """Return the complete previous metric snapshot; prefer bounded queries."""
    available = PREVIOUS_METRICS is not None
    return _transport_safe(
        {
            "available": available,
            "reason": None if available else (
                "No previous experiment exists until restart_session() successfully "
                "replaces an active session."
            ),
            "metrics": copy.deepcopy(PREVIOUS_METRICS),
            "session": copy.deepcopy(runtime.previous_session),
        }
    )


@mcp.tool()
def get_previous_structure() -> Dict[str, Any]:
    """Return prior structure only after a successful experiment restart."""
    available = PREVIOUS_STRUCTURE is not None
    return _transport_safe(
        {
            "available": available,
            "reason": None if available else (
                "No previous experiment exists until restart_session() successfully "
                "replaces an active session."
            ),
            "model_structure": copy.deepcopy(PREVIOUS_STRUCTURE),
            "session": copy.deepcopy(runtime.previous_session),
        }
    )


@mcp.tool()
def get_previous_evaluation() -> Dict[str, Any]:
    """Return prior HDBSCAN evaluation if the replaced session was evaluated."""
    available = PREVIOUS_EVALUATION is not None
    return _transport_safe(
        {
            "available": available,
            "reason": None if available else (
                "No previous HDBSCAN evaluation exists. Run evaluate_hdbscan() "
                "before restart_session() if downstream A/B evidence is needed."
            ),
            "evaluation": copy.deepcopy(PREVIOUS_EVALUATION),
            "session": copy.deepcopy(runtime.previous_session),
        }
    )


@mcp.tool()
def evaluate_hdbscan() -> Dict[str, Any]:
    """Evaluate the active model with its selected two-phase HDBSCAN DAGs."""
    return _transport_safe(runtime.evaluate_hdbscan())


@mcp.tool()
def get_context() -> Dict[str, Any]:
    """Return current session, effective config, model structure, probes, and evidence summary."""
    return _transport_safe(runtime.get_context())


@mcp.tool()
def get_experiment_comparison() -> Dict[str, Any]:
    """Return a compact current-vs-previous comparison envelope and matched step window."""
    return _transport_safe(runtime.get_experiment_comparison())


@mcp.tool()
def train_segment(steps: int) -> Dict[str, Any]:
    """Continue current training for 1..6000 steps without recreating session state."""
    return _transport_safe(runtime.train_segment(steps))


@mcp.tool()
def list_metrics(stage: Optional[str] = None) -> Dict[str, Any]:
    """List metrics actually recorded in the current session."""
    return _transport_safe(runtime.list_metrics(stage=stage))


@mcp.tool()
def query_metrics(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Query current metrics/history in one bounded batch (up to 16 queries, 200 records each)."""
    return _transport_safe(runtime.query_metrics(queries))


@mcp.tool()
def list_previous_metrics(stage: Optional[str] = None) -> Dict[str, Any]:
    """List metrics available in the previous-session snapshot."""
    return _transport_safe(runtime.list_previous_metrics(stage=stage))


@mcp.tool()
def query_previous_metrics(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Query previous-session metrics with the same bounded query shape as current metrics."""
    return _transport_safe(runtime.query_previous_metrics(queries))


atexit.register(runtime.close)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
