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
    """Detailed clothing item categories."""

    # Headwear
    HAT = "hat"
    CAP = "cap"
    BEANIE = "beanie"
    
    # Tops - Basic
    T_SHIRT = "t-shirt"
    SHIRT = "shirt"
    BLOUSE = "blouse"
    TANK_TOP = "tank top"
    POLO_SHIRT = "polo shirt"
    
    # Tops - Warm/Layering
    SWEATER = "sweater"
    HOODIE = "hoodie"
    CARDIGAN = "cardigan"
    
    # Outerwear
    JACKET = "jacket"
    COAT = "coat"
    BLAZER = "blazer"
    VEST = "vest"
    
    # Bottoms - Pants
    JEANS = "jeans"
    PANTS = "pants"
    TROUSERS = "trousers"
    LEGGINGS = "leggings"
    SWEATPANTS = "sweatpants"
    JOGGERS = "joggers"
    CHINOS = "chinos"
    
    # Bottoms - Other
    SHORTS = "shorts"
    SKIRT = "skirt"
    
    # Dresses/One-piece
    DRESS = "dress"
    JUMPSUIT = "jumpsuit"
    ROMPER = "romper"
    
    # Footwear
    SHOES = "shoes"
    SNEAKERS = "sneakers"
    BOOTS = "boots"
    SANDALS = "sandals"
    HEELS = "heels"
    FLATS = "flats"
    LOAFERS = "loafers"
    SLIPPERS = "slippers"
    ATHLETIC_SHOES = "athletic shoes"
    DRESS_SHOES = "dress shoes"
    
    # Undergarments
    SOCKS = "socks"
    UNDERWEAR = "underwear"
    BRA = "bra"
    
    # Swimwear
    SWIMSUIT = "swimsuit"
    BIKINI = "bikini"
    SWIM_TRUNKS = "swim trunks"
    
    # Accessories - Wearable
    SCARF = "scarf"
    GLOVES = "gloves"
    MITTENS = "mittens"
    BELT = "belt"
    TIE = "tie"
    BOW_TIE = "bow tie"
    
    # Accessories - Jewelry
    JEWELRY = "jewelry"
    NECKLACE = "necklace"
    EARRINGS = "earrings"
    BRACELET = "bracelet"
    WATCH = "watch"
    RING = "ring"
    SUNGLASSES = "sunglasses"
    
    # Bags
    PURSE = "purse"
    HANDBAG = "handbag"
    BACKPACK = "backpack"
    WALLET = "wallet"


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
    condition: str | None = None


# =============================================================================
# ATTRIBUTE EXTRACTION MODELS
# =============================================================================


class AttributeRequest(BaseModel):
    """Request for a specific attribute extraction."""

    request_id: UUID
    attribute_name: str
    image_paths: list[str] = Field(..., min_items=1, max_items=4)


class AttributeResponse(BaseModel):
    """Response for individual attribute extraction."""

    request_id: UUID
    attribute_name: str
    attribute_value: Any | None
    confidence_score: float
    processing_time_ms: int
    success: bool
    error_message: str | None = None


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


class AttributeModelInfo(BaseModel):
    """Information about model processing for a specific attribute."""

    service_name: str  # e.g. "heuristics", "llm_multimodal", "fashion_clip"
    service_type: str  # e.g. "heuristic", "llm", "pre_trained_vision"
    version: str = "1.0.0"
    processing_time_ms: int
    confidence_score: float
    success: bool
    error_message: str | None = None


class ProcessingInfo(BaseModel):
    """Overall processing metadata."""

    request_id: UUID
    total_processing_time_ms: int
    image_download_time_ms: int
    timestamp: datetime
    image_count: int
    implemented_attributes: list[str] = []  # Successfully extracted attributes
    skipped_attributes: list[str] = []  # Attributes skipped due to no implementation


class AnalyzeResponse(BaseModel):
    """API response for item analysis."""

    id: UUID = Field(default_factory=uuid4)
    attributes: AllAttributes  # All 13 attributes, some may be None if not implemented
    model_info: dict[str, AttributeModelInfo]  # attribute_name -> model info
    processing: ProcessingInfo  # Renamed from processing_info


# =============================================================================
# DATABASE MODELS
# =============================================================================


class InferenceResult(BaseModel):
    """Database model for storing inference results."""

    id: UUID = Field(default_factory=uuid4)
    request_data: dict[str, Any]
    attributes: AllAttributes
    model_info: dict[str, AttributeModelInfo]
    processing: ProcessingInfo
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
    format: str = "text"

    @validator("level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
