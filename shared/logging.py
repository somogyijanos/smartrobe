"""
Simple structured logging setup for Smartrobe services.
"""

import sys

from loguru import logger

from .config import get_settings


def setup_logging(service_name: str, settings=None) -> None:
    """Configure basic structured logging for a service."""
    if settings is None:
        settings = get_settings()

    # Remove default logger
    logger.remove()

    # Configure format based on log format setting
    if settings.log_format.lower() == "json":
        # When using serialize=True, loguru handles JSON formatting automatically
        # Use a minimal format string that works with serialization
        format_string = "{message}"
        serialize_logs = True
    else:
        # For text format, use a readable format that includes extra fields
        format_string = "{time:HH:mm:ss} | {level: <8} | {message} | {extra}"
        serialize_logs = False

    # Add console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=settings.log_level,
        serialize=serialize_logs,
        backtrace=True,  # Include variable values in tracebacks
        diagnose=True,   # Show extra diagnosis information
    )

    # Add service context - this will be included in serialized logs automatically
    logger.configure(extra={"service": service_name})


def get_logger(request_id: str = None):
    """Get a logger with optional request ID."""
    if request_id:
        return logger.bind(request_id=request_id)
    return logger
