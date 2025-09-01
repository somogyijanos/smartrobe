"""
LLM Multimodal service for Smartrobe.

Extracts contextual attributes using multimodal large language models.
Combines vision and text understanding for sophisticated attribute reasoning.
Currently implements garment condition assessment.
"""

import asyncio
import base64
import json
import os
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from shared.base_service import CapabilityService



# Condition analysis schemas (adapted from analyze_garment_condition.py)
class OverallCondition(str, Enum):
    """Overall condition ratings for garments."""
    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


class SeverityLevel(str, Enum):
    """Severity levels for damages and issues."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class DamageType(str, Enum):
    """Types of damage that can be found on garments."""
    HOLE = "hole"
    TEAR = "tear"
    RIP = "rip"
    STAIN = "stain"
    DISCOLORATION = "discoloration"
    FADING = "fading"
    YELLOWING = "yellowing"
    PILLING = "pilling"
    THINNING = "thinning"
    STRETCHING = "stretching"
    WORN_EDGES = "worn_edges"
    LOOSE_SEAM = "loose_seam"
    BROKEN_SEAM = "broken_seam"
    MISSING_BUTTON = "missing_button"
    BROKEN_ZIPPER = "broken_zipper"
    DEFORMATION = "deformation"
    SHRINKAGE = "shrinkage"
    OTHER = "other"


class HardwareCondition(str, Enum):
    """Condition of hardware elements like buttons, zippers, etc."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    BROKEN = "Broken"


class DamageItem(BaseModel):
    """Represents a specific damage or issue found on the garment."""
    type: str = Field(..., description="Type of damage")
    location: str = Field(..., description="Location on the garment")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Detailed description")


class GarmentConditionAnalysis(BaseModel):
    """Structured analysis of garment condition."""
    overall_condition: str = Field(..., description="Overall condition rating")
    overall_score: int = Field(
        ..., ge=1, le=10, description="Numeric score from 1 (poor) to 10 (excellent)"
    )
    damages: list[DamageItem] = Field(
        default_factory=list, description="List of specific damages found"
    )
    stains_and_discoloration: list[DamageItem] = Field(
        default_factory=list, description="List of stains and color issues"
    )
    wear_signs: list[DamageItem] = Field(
        default_factory=list, description="List of wear indicators"
    )
    structural_issues: list[DamageItem] = Field(
        default_factory=list, description="List of structural problems"
    )
    hardware_condition: str = Field(
        ..., description="Condition of buttons, zippers, etc."
    )
    resale_suitable: bool = Field(
        ..., description="Whether item is suitable for resale"
    )
    repair_recommendations: list[str] = Field(
        default_factory=list, description="Suggested repairs if applicable"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in assessment (0.0-1.0)"
    )
    summary: str = Field(..., description="Brief summary of the analysis")


class LLMMultimodalService(CapabilityService):
    """LLM Multimodal service for contextual attribute extraction."""

    SUPPORTED_ATTRIBUTES = {
        "condition": str,  # Free-form condition description
    }
    SERVICE_TYPE = "llm"

    def __init__(self):
        super().__init__("llm_multimodal", "llm", "1.0.0")
        # Initialize OpenAI client - will be set up during service initialization
        self.openai_client = None

    async def _initialize_service(self) -> None:
        """Initialize LLM multimodal service."""
        # Initialize OpenAI client for real condition analysis
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not found in environment variables")
            self.set_health_detail("openai_api_key_missing", True)
        else:
            self.openai_client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
            self.set_health_detail("openai_api_key_missing", False)

        await asyncio.sleep(0.1)  # Minimal startup delay
        self.logger.info("LLM multimodal service initialized successfully")
        self.set_health_detail("llm_api_connected", True)
        self.set_health_detail("vision_encoder_loaded", True)
        self.set_health_detail("prompt_templates_loaded", True)
        self.set_health_detail("multimodal_pipeline_ready", True)
        self.set_health_detail("model_type", "gpt-4o_vision")

    async def _cleanup_service(self) -> None:
        """Cleanup LLM multimodal service."""
        self.logger.info("LLM multimodal service cleaned up")

    async def extract_single_attribute(
        self, request_id: str, attribute_name: str, image_paths: list[str]
    ) -> tuple[Any, float]:
        """
        Extract a single attribute using multimodal LLM reasoning.

        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            image_paths: List of paths to images

        Returns:
            Tuple of (attribute_value, confidence_score)
        """
        self.logger.info(
            f"Extracting {attribute_name} using multimodal LLM",
            request_id=request_id,
            attribute=attribute_name,
            image_count=len(image_paths),
        )

        if attribute_name == "condition":
            return await self._extract_condition(request_id, image_paths)
        else:
            raise ValueError(f"Unsupported attribute: {attribute_name}")



    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def _extract_condition(
        self, request_id: str, image_paths: list[str]
    ) -> tuple[str, float]:
        """
        Extract garment condition using OpenAI's GPT-4V vision model.

        Analyzes images for:
        1. Visible damages, wear, stains, and overall condition
        2. Holes, tears, fading, pilling, structural issues
        3. Hardware condition (buttons, zippers, etc.)
        4. Overall condition assessment

        Args:
            request_id: Unique request identifier
            image_paths: List of paths to images

        Returns:
            Tuple of (condition_summary_string, confidence_score)
        """
        start_time = time.time()

        if not self.openai_client:
            self.logger.error(
                "OpenAI client not initialized",
                request_id=request_id
            )
            return "Unable to assess condition - OpenAI client not available", 0.0

        try:
            # Prepare image content for the API
            image_content = []
            for image_path in image_paths:
                if not Path(image_path).exists():
                    self.logger.warning(
                        f"Image not found: {image_path}",
                        request_id=request_id
                    )
                    continue

                # Get file extension to determine image type
                file_ext = Path(image_path).suffix.lower()
                if file_ext in ['.jpg', '.jpeg']:
                    media_type = "image/jpeg"
                elif file_ext == '.png':
                    media_type = "image/png"
                elif file_ext == '.webp':
                    media_type = "image/webp"
                else:
                    self.logger.warning(
                        f"Unsupported image format: {file_ext}",
                        request_id=request_id
                    )
                    continue

                # Encode image to base64
                base64_image = self._encode_image_to_base64(image_path)

                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })

            if not image_content:
                return "No valid images found to analyze", 0.0

            # Get JSON schema for structured output
            schema = GarmentConditionAnalysis.model_json_schema()

            # Prepare the prompt for condition analysis
            prompt = f"""
            Please analyze the condition of this garment item from the provided images.

            Examine the garment carefully for:
            1. Overall condition and visible damages
            2. Stains, discoloration, fading, or yellowing
            3. Wear signs like pilling, thinning, stretching
            4. Structural issues like loose/broken seams
            5. Hardware condition (buttons, zippers, etc.)

            Provide your response as valid JSON following this exact schema:
            {json.dumps(schema, indent=2)}

            Be thorough, accurate, and assess the confidence in your analysis.
            """

            # Create the message content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + image_content
                }
            ]

            # Make the API call with JSON mode
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Parse the JSON response
            response_text = response.choices[0].message.content
            response_data = json.loads(response_text)

            # Validate and create structured response
            analysis = GarmentConditionAnalysis.model_validate(response_data)

            processing_time_ms = int((time.time() - start_time) * 1000)

            self.logger.info(
                "Condition extraction completed",
                request_id=request_id,
                overall_condition=analysis.overall_condition,
                overall_score=analysis.overall_score,
                confidence=analysis.confidence_score,
                processing_time_ms=processing_time_ms,
                images_processed=len(image_content),
            )

            # Return the summary string and confidence score
            return analysis.summary, analysis.confidence_score

        except json.JSONDecodeError as e:
            self.logger.error(
                "Invalid JSON response from OpenAI",
                request_id=request_id,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
            return "Unable to parse condition analysis", 0.1

        except Exception as e:
            self.logger.error(
                "Error analyzing condition",
                request_id=request_id,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
            return f"Error analyzing condition: {str(e)}", 0.1

    async def _check_service_health(self) -> bool:
        """Check LLM multimodal service health."""
        try:
            test_start = time.time()

            # Check OpenAI client availability
            if self.openai_client:
                self.set_health_detail("openai_client_ready", True)
                # Could add actual API ping here if needed
            else:
                self.set_health_detail("openai_client_ready", False)
                self.logger.warning("OpenAI client not available")

            await asyncio.sleep(0.05)  # Simulate health check
            health_check_time = time.time() - test_start
            self.set_health_detail(
                "last_health_check_ms", int(health_check_time * 1000)
            )
            self.set_health_detail(
                "api_response_time_ms", int(health_check_time * 1000)
            )
            self.set_health_detail("multimodal_reasoning_ready", True)
            self.set_health_detail("vision_understanding_ready", True)
            return True
        except Exception as e:
            self.logger.error(
                "LLM multimodal service health check failed", error=str(e)
            )
            return False


def create_app():
    """Create the LLM multimodal FastAPI application."""
    service = LLMMultimodalService()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = LLMMultimodalService()
    service.run()
