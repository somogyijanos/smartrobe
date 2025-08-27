"""
Configuration management for Smartrobe services.

Provides centralized configuration loading from environment variables
with validation and type safety using Pydantic.
"""

import os

from pydantic import BaseModel, validator

from .schemas import DatabaseConfig, ImageConfig, LogConfig, ServiceConfig


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # =============================================================================
    # SERVICE CONFIGURATION
    # =============================================================================
    orchestrator_port: int
    vision_service_port: int
    heuristic_service_port: int
    llm_service_port: int

    # Service URLs for inter-service communication
    vision_service_url: str
    heuristic_service_url: str
    llm_service_url: str

    # =============================================================================
    # SHARED STORAGE
    # =============================================================================
    shared_storage_path: str

    # =============================================================================
    # IMAGE PROCESSING
    # =============================================================================
    max_image_size_mb: int
    allowed_image_formats: list[str]
    image_download_timeout: int

    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    service_request_timeout: int
    max_concurrent_requests: int

    # =============================================================================
    # LOGGING
    # =============================================================================
    log_level: str
    log_format: str

    # =============================================================================
    # DEVELOPMENT SETTINGS
    # =============================================================================
    debug: bool
    environment: str

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            url=self.database_url, timeout_seconds=self.service_request_timeout
        )

    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get service configuration for a specific service."""
        port_mapping = {
            "orchestrator": self.orchestrator_port,
            "vision_classifier": self.vision_service_port,
            "heuristic_model": self.heuristic_service_port,
            "llm_model": self.llm_service_port,
        }

        port = port_mapping.get(service_name)
        if port is None:
            raise ValueError(f"Unknown service: {service_name}")

        return ServiceConfig(
            port=port,
            timeout_seconds=self.service_request_timeout,
            max_concurrent_requests=self.max_concurrent_requests,
        )

    def get_image_config(self) -> ImageConfig:
        """Get image processing configuration."""
        return ImageConfig(
            max_size_mb=self.max_image_size_mb,
            allowed_formats=self.allowed_image_formats,
            download_timeout_seconds=self.image_download_timeout,
            storage_path=self.shared_storage_path,
        )

    def get_log_config(self) -> LogConfig:
        """Get logging configuration."""
        return LogConfig(level=self.log_level, format=self.log_format)


def get_settings() -> Settings:
    """Get application settings singleton."""
    if not hasattr(get_settings, "_instance"):

        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        def parse_formats(formats_str: str) -> list[str]:
            return [fmt.strip().lower() for fmt in formats_str.split(",")]

        get_settings._instance = Settings(
            database_url=os.environ["DATABASE_URL"],
            postgres_user=os.environ["POSTGRES_USER"],
            postgres_password=os.environ["POSTGRES_PASSWORD"],
            postgres_db=os.environ["POSTGRES_DB"],
            orchestrator_port=int(os.environ["ORCHESTRATOR_PORT"]),
            vision_service_port=int(os.environ["VISION_SERVICE_PORT"]),
            heuristic_service_port=int(os.environ["HEURISTIC_SERVICE_PORT"]),
            llm_service_port=int(os.environ["LLM_SERVICE_PORT"]),
            vision_service_url=os.environ["VISION_SERVICE_URL"],
            heuristic_service_url=os.environ["HEURISTIC_SERVICE_URL"],
            llm_service_url=os.environ["LLM_SERVICE_URL"],
            shared_storage_path=os.environ["SHARED_STORAGE_PATH"],
            max_image_size_mb=int(os.environ["MAX_IMAGE_SIZE_MB"]),
            allowed_image_formats=parse_formats(os.environ["ALLOWED_IMAGE_FORMATS"]),
            image_download_timeout=int(os.environ["IMAGE_DOWNLOAD_TIMEOUT"]),
            service_request_timeout=int(os.environ["SERVICE_REQUEST_TIMEOUT"]),
            max_concurrent_requests=int(os.environ["MAX_CONCURRENT_REQUESTS"]),
            log_level=os.environ["LOG_LEVEL"],
            log_format=os.environ["LOG_FORMAT"],
            debug=parse_bool(os.environ["DEBUG"]),
            environment=os.environ["ENVIRONMENT"],
        )
    return get_settings._instance


def create_request_directory(request_id: str, storage_path: str | None = None) -> str:
    """Create a unique directory for request-scoped image storage."""
    settings = get_settings()
    base_path = storage_path or settings.shared_storage_path
    request_dir = os.path.join(base_path, request_id)
    os.makedirs(request_dir, exist_ok=True)
    return request_dir


def cleanup_request_directory(request_id: str, storage_path: str | None = None) -> None:
    """Clean up request-scoped directory after processing."""
    import shutil

    settings = get_settings()
    base_path = storage_path or settings.shared_storage_path
    request_dir = os.path.join(base_path, request_id)

    if os.path.exists(request_dir):
        shutil.rmtree(request_dir)


def get_service_url(service_name: str) -> str:
    """Get the URL for a specific service."""
    settings = get_settings()
    url_mapping = {
        "vision_classifier": settings.vision_service_url,
        "heuristic_model": settings.heuristic_service_url,
        "llm_model": settings.llm_service_url,
    }

    url = url_mapping.get(service_name)
    if url is None:
        raise ValueError(f"Unknown service: {service_name}")

    return url
