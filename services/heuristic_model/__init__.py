"""
Heuristic model service for extracting color, material, pattern, and brand
information from clothing images using rule-based analysis.
"""

from .app import HeuristicModelService, create_app

__all__ = ["HeuristicModelService", "create_app"]
