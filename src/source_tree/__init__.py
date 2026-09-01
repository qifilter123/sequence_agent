"""Deterministic source facts and bounded local query support."""

from .fact_store import (
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    FactStoreError,
    QueryRequest,
    build_fact_database,
    query_facts,
)

__all__ = [
    "DEFAULT_QUERY_LIMIT",
    "MAX_QUERY_LIMIT",
    "FactStoreError",
    "QueryRequest",
    "build_fact_database",
    "query_facts",
]
