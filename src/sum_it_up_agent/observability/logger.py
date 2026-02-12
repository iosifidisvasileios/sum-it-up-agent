"""Central logging utilities with request correlation.

This module is designed to be imported by any component (agent, MCP servers, etc.)
without duplicating logging setup.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Iterator, Optional


_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "sum_it_up_correlation_id",
    default=None,
)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = _correlation_id_var.get() or "-"
        record.correlation_id = correlation_id
        # Backward-compatible fields for older log formatters/call sites.
        record.request_id = correlation_id
        record.session_uuid = correlation_id
        return True


_configured = False


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure root logging once.

    Safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    root.addHandler(handler)

    log_to_file = os.getenv("SUM_IT_UP_LOG_TO_FILE", "").strip().lower() in {"1", "true", "yes", "on"}
    if log_to_file:
        log_dir = Path(os.getenv("SUM_IT_UP_LOG_DIR", "logs")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        process_name = os.getenv("SUM_IT_UP_PROCESS_NAME", "").strip()
        basename = process_name or Path(sys.argv[0]).stem or "sum_it_up"
        log_path = log_dir / f"{basename}.log"

        append = os.getenv("SUM_IT_UP_LOG_APPEND", "1").strip().lower() in {"1", "true", "yes", "on"}
        file_handler = logging.FileHandler(log_path, mode=("a" if append else "w"), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RequestIdFilter())
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger. Ensure logging is configured."""
    configure_logging()
    return logging.getLogger(name)


def new_request_id() -> str:
    """Generate a new UUID4 correlation id.

    Kept as a backward-compatible name.
    """
    return str(uuid.uuid4())


def get_request_id() -> Optional[str]:
    return _correlation_id_var.get()


def get_session_uuid() -> Optional[str]:
    return _correlation_id_var.get()


def new_correlation_id() -> str:
    return new_request_id()


def get_correlation_id() -> Optional[str]:
    return _correlation_id_var.get()


@contextlib.contextmanager
def bind_request_id(request_id: Optional[str]) -> Iterator[None]:
    """Bind a correlation id to the current context (used by log filter).

    Kept as a backward-compatible name.
    """
    token = _correlation_id_var.set(request_id)
    try:
        yield
    finally:
        _correlation_id_var.reset(token)


@contextlib.contextmanager
def bind_session_uuid(session_uuid: Optional[str]) -> Iterator[None]:
    """Bind a correlation id to the current context (used by log filter).

    Kept as a backward-compatible name.
    """
    token = _correlation_id_var.set(session_uuid)
    try:
        yield
    finally:
        _correlation_id_var.reset(token)


@contextlib.contextmanager
def bind_correlation_id(correlation_id: Optional[str]) -> Iterator[None]:
    token = _correlation_id_var.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id_var.reset(token)
