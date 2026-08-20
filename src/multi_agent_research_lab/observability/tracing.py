"""Tracing hooks supporting LangSmith, Langfuse, and local structured spans."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing execution spans with latency, metadata, and provider hooks."""
    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "running",
    }

    # Optional Langfuse span generation
    langfuse_span = None
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse import Langfuse

            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_base_url or settings.langfuse_host,
            )
            langfuse_span = lf.span(name=name, metadata=attributes or {})
        except Exception as exc:
            logger.debug(f"Langfuse span creation ignored: {exc}")

    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        if langfuse_span:
            langfuse_span.end(level="ERROR", status_message=str(exc))
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if langfuse_span and span.get("status") != "error":
            langfuse_span.end(metadata=span)
