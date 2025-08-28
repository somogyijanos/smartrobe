"""
Shared Pydantic schemas for the Smartrobe multi-model attribute extraction service.

This module defines all data models used across microservices for:
- API request/response structures
- Inter-service communication
- Database persistence
- Configuration validation
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, validator

# =============================================================================
# ENUMS FOR ATTRIBUTE VALUES
# =============================================================================


class ClothingCategory(str, Enum):
    """Clothing item categories."""

    SHIRT = "shirt"
    PANTS = "pants"
    DRESS = "dress"
    JACKET = "jacket"
    SKIRT = "skirt"
    SWEATER = "sweater"
    BLOUSE = "blouse"
    SHORTS = "shorts"
    SUIT = "suit"
    OTHER = "other"


class Gender(str, Enum):
    """Target gender for clothing."""

    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class SleeveLength(str, Enum):
    """Sleeve length types."""

    SLEEVELESS = "sleeveless"
    SHORT = "short"
    THREE_QUARTER = "three_quarter"
    LONG = "long"


class Neckline(str, Enum):
    """Neckline types."""

    CREW = "crew"
    V_NECK = "v_neck"
    SCOOP = "scoop"
    BOAT = "boat"
    SQUARE = "square"
    HALTER = "halter"
    OFF_SHOULDER = "off_shoulder"
    OTHER = "other"


class ClosureType(str, Enum):
    """Closure mechanisms."""

    BUTTON = "button"
    ZIP = "zip"
    PULLOVER = "pullover"
    TIE = "tie"
    SNAP = "snap"
    VELCRO = "velcro"
    OTHER = "other"


class Fit(str, Enum):
    """Clothing fit types."""

    SLIM = "slim"
    REGULAR = "regular"
    LOOSE = "loose"
    OVERSIZED = "oversized"
    TIGHT = "tight"


class Color(str, Enum):
    """Primary color categories."""

    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    BROWN = "brown"
    BEIGE = "beige"
    RED = "red"
    PINK = "pink"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    NAVY = "navy"
    MULTICOLOR = "multicolor"


class Material(str, Enum):
    """Fabric/material types."""

    COTTON = "cotton"
    POLYESTER = "polyester"
    WOOL = "wool"
    LINEN = "linen"
    SILK = "silk"
    DENIM = "denim"
    LEATHER = "leather"
    SYNTHETIC = "synthetic"
    BLEND = "blend"
    OTHER = "other"


class Pattern(str, Enum):
    """Pattern types."""

    SOLID = "solid"
    STRIPED = "striped"
    CHECKERED = "checkered"
    FLORAL = "floral"
    GEOMETRIC = "geometric"
    ANIMAL_PRINT = "animal_print"
    ABSTRACT = "abstract"
    OTHER = "other"


class Style(str, Enum):
    """Style categories."""

    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    SPORTY = "sporty"
    VINTAGE = "vintage"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    TRENDY = "trendy"
    CLASSIC = "classic"


class Season(str, Enum):
    """Seasonal appropriateness."""

    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class Condition(str, Enum):
    """Item condition assessment."""

    NEW = "new"
    LIKE_NEW = "like_new"
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# =============================================================================
# ATTRIBUTE MODELS
# =============================================================================


class VisionAttributes(BaseModel):
    """Attributes extracted by vision classifier service."""

    category: ClothingCategory | None = None
    gender: Gender | None = None
    sleeve_length: SleeveLength | None = None
    neckline: Neckline | None = None
    closure_type: ClosureType | None = None
    fit: Fit | None = None


class HeuristicAttributes(BaseModel):
    """Attributes extracted by heuristic model service."""

    color: Color | None = None
    material: Material | None = None
    pattern: Pattern | None = None
    brand: str | None = None


class LLMAttributes(BaseModel):
    """Attributes extracted by LLM service."""

    style: Style | None = None
    season: Season | None = None
    condition: Condition | None = None


class AllAttributes(BaseModel):
    """Combined attributes from all model types."""

    # Vision attributes
    category: ClothingCategory | None = None
    gender: Gender | None = None
    sleeve_length: SleeveLength | None = None
    neckline: Neckline | None = None
    closure_type: ClosureType | None = None
    fit: Fit | None = None

    # Heuristic attributes
    color: Color | None = None
    material: Material | None = None
    pattern: Pattern | None = None
    brand: str | None = None

    # LLM attributes
    style: Style | None = None
    season: Season | None = None
    condition: Condition | None = None


# =============================================================================
# SERVICE COMMUNICATION MODELS
# =============================================================================


class ServiceRequest(BaseModel):
    """Base request model for service communication."""

    request_id: UUID
    image_paths: list[str] = Field(..., min_items=1, max_items=4)


class ServiceResponse(BaseModel):
    """Base response model for service communication."""

    request_id: UUID
    success: bool
    attributes: dict[str, Any]
    confidence_scores: dict[str, float] = {}
    processing_time_ms: int
    error_message: str | None = None


class VisionServiceResponse(ServiceResponse):
    """Response from vision classifier service."""

    attributes: VisionAttributes


class HeuristicServiceResponse(ServiceResponse):
    """Response from heuristic model service."""

    attributes: HeuristicAttributes


class LLMServiceResponse(ServiceResponse):
    """Response from LLM service."""

    attributes: LLMAttributes


# =============================================================================
# API MODELS
# =============================================================================


class AnalyzeRequest(BaseModel):
    """API request for item analysis."""

    images: list[HttpUrl] = Field(..., min_items=1, max_items=4)

    @validator("images")
    def validate_image_urls(cls, v):
        """Validate that all URLs are HTTPS."""
        for url in v:
            if url.scheme != "https":
                raise ValueError("All image URLs must use HTTPS")
        return v


class ModelInfo(BaseModel):
    """Information about model processing."""

    model_type: str
    version: str = "1.0.0"
    processing_time_ms: int
    confidence_scores: dict[str, float] = {}
    success: bool
    error_message: str | None = None


class ProcessingInfo(BaseModel):
    """Overall processing metadata."""

    request_id: UUID
    total_processing_time_ms: int
    image_download_time_ms: int
    parallel_processing: bool = True
    timestamp: datetime
    image_count: int


class AnalyzeResponse(BaseModel):
    """API response for item analysis."""

    id: UUID = Field(default_factory=uuid4)
    attributes: AllAttributes
    model_info: dict[str, ModelInfo]
    processing_info: ProcessingInfo


# =============================================================================
# DATABASE MODELS
# =============================================================================


class InferenceResult(BaseModel):
    """Database model for storing inference results."""

    id: UUID = Field(default_factory=uuid4)
    request_data: dict[str, Any]
    attributes: AllAttributes
    model_info: dict[str, ModelInfo]
    processing_info: ProcessingInfo
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# HEALTH CHECK MODELS
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheck(BaseModel):
    """Health check response model."""

    service: str
    status: HealthStatus
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = {}


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str
    max_connections: int = 10
    timeout_seconds: int = 30


class ServiceConfig(BaseModel):
    """Service configuration."""

    host: str = "0.0.0.0"
    port: int
    timeout_seconds: int = 30
    max_concurrent_requests: int = 10


class ImageConfig(BaseModel):
    """Image processing configuration."""

    max_size_mb: int = 10
    allowed_formats: list[str] = ["jpeg", "jpg", "png", "webp"]
    download_timeout_seconds: int = 30
    storage_path: str = "/app/shared_storage"


class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"

    @validator("level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
