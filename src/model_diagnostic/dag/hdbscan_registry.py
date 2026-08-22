from __future__ import annotations

from model_diagnostic.dag.operation_registry import OperationRegistry


# Runtime HDBSCAN DAG registry.  Unlike MODEL_BUILDER_REGISTRY, operations in
# this registry execute data/geometry/evaluation work immediately.
HDBSCAN_DAG_REGISTRY = OperationRegistry()
