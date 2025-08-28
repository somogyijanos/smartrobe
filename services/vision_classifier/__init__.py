"""
Vision classifier service for extracting visual attributes from clothing images.
"""

from .app import VisionClassifierService, create_app

__all__ = ["VisionClassifierService", "create_app"]
