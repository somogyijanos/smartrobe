"""Unified configuration management for Smartrobe services.

Simple, clean configuration that works for all service types.
"""

import os
from typing import Optional

from pydantic import BaseModel, validator

from .schemas import DatabaseConfig, ImageConfig, LogConfig, ServiceConfig


class ServiceSettings(BaseModel):
    """Unified settings for all Smartrobe services.
    
    Each service uses only the fields it needs - clean and simple.
    """
    
    # ========================================================================
    # BASIC SETTINGS - Required by all services
    # ========================================================================
    shared_storage_path: str
    log_level: str
    log_format: str
    debug: bool
    environment: str
    service_port: int = 8000
    
    # ========================================================================
    # DATABASE SETTINGS - Optional, only used by orchestrator
    # ========================================================================
    database_url: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None
    
    # ========================================================================
    # SERVICE URLS - Optional, only used by services that call other services
    # ========================================================================
    heuristics_service_url: Optional[str] = None
    llm_multimodal_service_url: Optional[str] = None
    fashion_clip_service_url: Optional[str] = None
    microsoft_florence_service_url: Optional[str] = None
    
    # ========================================================================
    # PROCESSING SETTINGS - Optional, only used by orchestrator
    # ========================================================================
    max_image_size_mb: Optional[int] = None
    allowed_image_formats: Optional[list[str]] = None
    image_download_timeout: Optional[int] = None
    service_request_timeout: Optional[int] = 30
    max_concurrent_requests: Optional[int] = 10
    debug_retain_images: Optional[bool] = False

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
    
    def get_service_config(self) -> ServiceConfig:
        """Get service configuration."""
        return ServiceConfig(
            port=self.service_port,
            timeout_seconds=self.service_request_timeout or 30,
            max_concurrent_requests=self.max_concurrent_requests or 10,
        )
        
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        if not self.database_url:
            raise ValueError("Database configuration not available for this service")
        return DatabaseConfig(
            url=self.database_url, 
            timeout_seconds=self.service_request_timeout or 30
        )
    
    def get_image_config(self) -> ImageConfig:
        """Get image processing configuration."""
        if not all([self.max_image_size_mb, self.allowed_image_formats, self.image_download_timeout]):
            raise ValueError("Image configuration not available for this service")
        return ImageConfig(
            max_size_mb=self.max_image_size_mb,
            allowed_formats=self.allowed_image_formats,
            download_timeout_seconds=self.image_download_timeout,
            storage_path=self.shared_storage_path,
        )
    
    def get_service_url(self, service_name: str) -> str:
        """Get URL for another service."""
        url_mapping = {
            "heuristics": self.heuristics_service_url,
            "llm-multimodal": self.llm_multimodal_service_url,
            "fashion-clip": self.fashion_clip_service_url,
            "microsoft-florence": self.microsoft_florence_service_url,
        }
        
        url = url_mapping.get(service_name)
        if url is None:
            raise ValueError(f"Service URL for '{service_name}' not configured")
        return url


# ============================================================================
# UNIFIED SETTINGS LOADER
# ============================================================================

def get_settings() -> ServiceSettings:
    """Get unified service settings.
    
    Loads all environment variables once, services use what they need.
    Clean, simple, no confusion.
    """
    if not hasattr(get_settings, "_instance"):
        def parse_bool(value: str) -> bool:
            return value.lower() in ("true", "1", "yes", "on")

        def parse_formats(formats_str: str) -> list[str]:
            if not formats_str:
                return []
            return [fmt.strip().lower() for fmt in formats_str.split(",")]

        get_settings._instance = ServiceSettings(
            # Basic settings (required)
            shared_storage_path=os.environ["SHARED_STORAGE_PATH"],
            log_level=os.environ["LOG_LEVEL"],
            log_format=os.environ["LOG_FORMAT"],
            debug=parse_bool(os.environ["DEBUG"]),
            environment=os.environ["ENVIRONMENT"],
            service_port=int(os.environ.get("SERVICE_PORT", "8000")),
            
            # Database settings (optional - only for orchestrator)
            database_url=os.environ.get("DATABASE_URL"),
            postgres_user=os.environ.get("POSTGRES_USER"),
            postgres_password=os.environ.get("POSTGRES_PASSWORD"),
            postgres_db=os.environ.get("POSTGRES_DB"),
            
            # Service URLs (optional - only for services that need them)
            heuristics_service_url=os.environ.get("HEURISTICS_SERVICE_URL"),
            llm_multimodal_service_url=os.environ.get("LLM_MULTIMODAL_SERVICE_URL"),
            fashion_clip_service_url=os.environ.get("FASHION_CLIP_SERVICE_URL"),
            microsoft_florence_service_url=os.environ.get("MICROSOFT_FLORENCE_SERVICE_URL"),
            
            # Processing settings (optional - only for orchestrator)
            max_image_size_mb=int(os.environ["MAX_IMAGE_SIZE_MB"]) if os.environ.get("MAX_IMAGE_SIZE_MB") else None,
            allowed_image_formats=parse_formats(os.environ.get("ALLOWED_IMAGE_FORMATS", "")),
            image_download_timeout=int(os.environ["IMAGE_DOWNLOAD_TIMEOUT"]) if os.environ.get("IMAGE_DOWNLOAD_TIMEOUT") else None,
            service_request_timeout=int(os.environ.get("SERVICE_REQUEST_TIMEOUT", "30")),
            max_concurrent_requests=int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10")),
            debug_retain_images=parse_bool(os.environ.get("DEBUG_RETAIN_IMAGES", "false")),
        )
    return get_settings._instance


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    return settings.get_service_url(service_name)


# ============================================================================
# LEGACY COMPATIBILITY - NO MORE CONFUSION!
# ============================================================================

# Legacy aliases for backward compatibility
Settings = ServiceSettings
OrchestratorSettings = ServiceSettings
ModelServiceSettings = ServiceSettings

# Legacy functions for backward compatibility
get_orchestrator_settings = get_settings
get_model_service_settings = get_settings