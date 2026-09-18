"""Retrieval backend implementations and registry."""

from rq1.backends.registry import get_backend

__all__ = ["get_backend"]
