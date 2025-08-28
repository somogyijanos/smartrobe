"""
Orchestrator service for Smartrobe multi-model attribute extraction.

Main service that coordinates image processing across vision classifier,
heuristic model, and LLM services.
"""

from .app import OrchestratorService, create_app

__all__ = ["OrchestratorService", "create_app"]
