"""
Simple structured logging setup for Smartrobe services.
"""

import sys

from loguru import logger

from .config import get_settings


def setup_logging(service_name: str) -> None:
    """Configure basic structured logging for a service."""
    settings = get_settings()

    # Remove default logger
    logger.remove()

    # Simple format
    if settings.log_format.lower() == "json":
        format_string = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss}", '
            '"level": "{level}", '
            '"service": "' + service_name + '", '
            '"message": "{message}"}'
        )
    else:
        format_string = "{time:HH:mm:ss} | {level: <8} | {service} | {message}"

    # Add console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=settings.log_level,
        serialize=settings.log_format.lower() == "json",
    )

    # Add service context
    logger.configure(extra={"service": service_name})


def get_logger(request_id: str = None):
    """Get a logger with optional request ID."""
    if request_id:
        return logger.bind(request_id=request_id)
    return logger
