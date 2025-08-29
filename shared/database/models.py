"""
SQLAlchemy database models for Smartrobe inference results.

Defines the database schema for storing and retrieving clothing item
analysis results with full JSONB support for flexible attribute storage.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""

    pass


class InferenceResult(Base):
    """
    Database model for storing clothing item inference results.
    
    Stores the complete analysis pipeline results including:
    - Original request data
    - Extracted attributes from all model types
    - Model processing metadata
    - Processing performance information
    """

    __tablename__ = "inference_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Request data (original image URLs and metadata)
    request_data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Extracted attributes from all model services
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Model processing information and metadata
    model_info: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Processing performance and system metadata
    processing_info: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        nullable=False
    )

    def __repr__(self) -> str:
        """String representation of inference result."""
        return f"<InferenceResult(id={self.id}, created_at={self.created_at})>"

    def to_dict(self) -> dict:
        """Convert model to dictionary representation."""
        return {
            "id": str(self.id),
            "request_data": self.request_data,
            "attributes": self.attributes,
            "model_info": self.model_info,
            "processing_info": self.processing_info,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
