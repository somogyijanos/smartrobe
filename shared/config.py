"""
Configuration management for Smartrobe services.

Provides centralized configuration loading from environment variables
with validation and type safety using Pydantic.
"""

import os

from pydantic import BaseModel, validator

from .schemas import DatabaseConfig, ImageConfig, LogConfig, ServiceConfig


class BaseSettings(BaseModel):
    """Base settings shared by all services."""
    # Common to all services
    shared_storage_path: str
    log_level: str
    log_format: str
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

    def get_log_config(self) -> LogConfig:
        """Get logging configuration."""
        return LogConfig(level=self.log_level, format=self.log_format)


class ModelServiceSettings(BaseSettings):
    """Settings for model services (vision, heuristic, llm)."""
    service_port: int = 8000  # All services use same internal port
    
    def get_service_config(self) -> ServiceConfig:
        """Get service configuration."""
        return ServiceConfig(
            port=self.service_port,
            timeout_seconds=30,  # Default timeout
            max_concurrent_requests=10,  # Default concurrency
        )


class OrchestratorSettings(BaseSettings):
    """Settings for orchestrator service (needs everything)."""
    
    # Database Configuration
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # Service Configuration 
    orchestrator_port: int = 8000  # Same internal port as others
    
    # Service URLs for inter-service communication (new capability-based services)
    heuristics_service_url: str
    llm_multimodal_service_url: str
    fashion_clip_service_url: str

    # Image Processing
    max_image_size_mb: int
    allowed_image_formats: list[str]
    image_download_timeout: int

    # API Configuration
    service_request_timeout: int
    max_concurrent_requests: int

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            url=self.database_url, timeout_seconds=self.service_request_timeout
        )

    def get_service_config(self) -> ServiceConfig:
        """Get orchestrator service configuration."""
        return ServiceConfig(
            port=self.orchestrator_port,
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

    def get_service_url(self, service_name: str) -> str:
        """Get the URL for a specific service."""
        url_mapping = {
            "heuristics": self.heuristics_service_url,
            "llm_multimodal": self.llm_multimodal_service_url,
            "fashion_clip": self.fashion_clip_service_url,
        }
        
        url = url_mapping.get(service_name)
        if url is None:
            raise ValueError(f"Unknown service: {service_name}")
        return url


# Legacy Settings class for backward compatibility during transition
Settings = OrchestratorSettings


def get_model_service_settings() -> ModelServiceSettings:
    """Get model service settings (for vision, heuristic, llm services)."""
    if not hasattr(get_model_service_settings, "_instance"):
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        get_model_service_settings._instance = ModelServiceSettings(
            service_port=int(os.environ.get("SERVICE_PORT", "8000")),
            shared_storage_path=os.environ["SHARED_STORAGE_PATH"],
            log_level=os.environ["LOG_LEVEL"],
            log_format=os.environ["LOG_FORMAT"],
            debug=parse_bool(os.environ["DEBUG"]),
            environment=os.environ["ENVIRONMENT"],
        )
    return get_model_service_settings._instance


def get_orchestrator_settings() -> OrchestratorSettings:
    """Get orchestrator settings (needs full configuration)."""
    if not hasattr(get_orchestrator_settings, "_instance"):
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        def parse_formats(formats_str: str) -> list[str]:
            return [fmt.strip().lower() for fmt in formats_str.split(",")]

        get_orchestrator_settings._instance = OrchestratorSettings(
            database_url=os.environ["DATABASE_URL"],
            postgres_user=os.environ["POSTGRES_USER"],
            postgres_password=os.environ["POSTGRES_PASSWORD"],
            postgres_db=os.environ["POSTGRES_DB"],
            orchestrator_port=int(os.environ.get("ORCHESTRATOR_PORT", "8000")),
            heuristics_service_url=os.environ["HEURISTICS_SERVICE_URL"],
            llm_multimodal_service_url=os.environ["LLM_MULTIMODAL_SERVICE_URL"],
            fashion_clip_service_url=os.environ["FASHION_CLIP_SERVICE_URL"],
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
    return get_orchestrator_settings._instance


def get_settings() -> OrchestratorSettings:
    """Get application settings (legacy - use get_orchestrator_settings)."""
    return get_orchestrator_settings()


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
    settings = get_orchestrator_settings()
    return settings.get_service_url(service_name)
