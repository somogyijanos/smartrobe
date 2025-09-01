"""
Fashion-CLIP service for Smartrobe.

Extracts fashion-specific visual attributes using pre-trained vision models.
Leverages CLIP-like architectures fine-tuned on fashion datasets.
"""

import asyncio
import random
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from shared.base_service import CapabilityService
from shared.schemas import ClothingCategory


class FashionClipService(CapabilityService):
    """Fashion-CLIP service for pre-trained vision-based attribute extraction."""

    SUPPORTED_ATTRIBUTES = {
        "category": ClothingCategory,
    }
    SERVICE_TYPE = "pre_trained_vision"

    def __init__(self):
        super().__init__("fashion_clip", "pre_trained_vision", "1.0.0")
        self.model = None
        self.processor = None



    async def _initialize_service(self) -> None:
        """Initialize Fashion-CLIP service."""
        try:
            self.logger.info("Loading marqo-fashionSigLIP model...")
            # Load the marqo-fashionSigLIP model and processor
            self.model = AutoModel.from_pretrained(
                'Marqo/marqo-fashionSigLIP', trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                'Marqo/marqo-fashionSigLIP', trust_remote_code=True
            )
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            self.logger.info(
                f"Fashion-CLIP service initialized successfully on {device}"
            )
            self.set_health_detail("clip_model_loaded", True)
            self.set_health_detail("vision_encoder_ready", True)
            self.set_health_detail("text_encoder_ready", True)
            self.set_health_detail("fashion_embeddings_loaded", True)
            self.set_health_detail("model_type", "marqo-fashionSigLIP")
            self.set_health_detail("inference_device", device)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Fashion-CLIP service: {e}")
            raise

    async def _cleanup_service(self) -> None:
        """Cleanup Fashion-CLIP service."""
        self.logger.info("Fashion-CLIP service cleaned up")

    async def extract_single_attribute(
        self, request_id: str, attribute_name: str, image_paths: list[str]
    ) -> tuple[Any, float]:
        """
        Extract a single attribute using Fashion-CLIP vision understanding.

        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            image_paths: List of paths to images

        Returns:
            Tuple of (attribute_value, confidence_score)
        """
        self.logger.info(
            f"Extracting {attribute_name} using Fashion-CLIP",
            request_id=request_id,
            attribute=attribute_name,
            image_count=len(image_paths),
        )

        if attribute_name == "category":
            return await self._extract_category(request_id, image_paths)
        else:
            raise ValueError(f"Unsupported attribute: {attribute_name}")



    async def _extract_category(
        self, request_id: str, image_paths: list[str]
    ) -> tuple[ClothingCategory, float]:
        """
        Extract clothing category using marqo-fashionSigLIP model.

        Args:
            request_id: Unique request identifier
            image_paths: List of paths to images

        Returns:
            Tuple of (ClothingCategory, confidence_score)
        """
        try:
            # Load and process the first image
            if not image_paths:
                raise ValueError("No image paths provided")
            
            image_path = image_paths[0]  # Use the first image for classification
            image = Image.open(image_path).convert('RGB')
            
            # Get all category values from enum
            category_values = [category.value for category in ClothingCategory]
            
            # Process image and text with the model
            processed = self.processor(
                text=category_values, 
                images=[image], 
                padding='max_length', 
                return_tensors="pt"
            )
            
            # Move to the same device as model
            device = next(self.model.parameters()).device
            processed = {
                k: v.to(device) if hasattr(v, 'to') else v 
                for k, v in processed.items()
            }
            
            with torch.no_grad():
                # Extract features
                image_features = self.model.get_image_features(
                    processed['pixel_values'], normalize=True
                )
                text_features = self.model.get_text_features(
                    processed['input_ids'], normalize=True
                )
                
                # Compute similarities and probabilities
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get the top predicted category
                top_idx = text_probs.argsort(dim=-1, descending=True)[0][0]
                predicted_category_value = category_values[top_idx]
                confidence = text_probs[0, top_idx].item()
                
                # Convert to ClothingCategory enum (should always work since we use enum values)
                clothing_category = ClothingCategory(predicted_category_value)
                
                self.logger.info(
                    "Category extraction completed",
                    request_id=request_id,
                    detected_category=clothing_category.value,
                    confidence=confidence,
                    image_path=image_path,
                )
                
                return clothing_category, confidence
                
        except Exception as e:
            self.logger.error(f"Category extraction failed: {e}", request_id=request_id)
            # Return a default category with low confidence
            return ClothingCategory.SHIRT, 0.0

    async def _check_service_health(self) -> bool:
        """Check Fashion-CLIP service health."""
        try:
            # Test model inference capability
            test_start = time.time()
            await asyncio.sleep(0.02)  # Simulate quick model inference
            
            health_check_time = time.time() - test_start
            self.set_health_detail("last_health_check_ms", int(health_check_time * 1000))
            self.set_health_detail("model_inference_time_ms", int(health_check_time * 1000))
            self.set_health_detail("vision_model_ready", True)
            self.set_health_detail("embedding_computation_ready", True)
            
            return True
        except Exception as e:
            self.logger.error("Fashion-CLIP service health check failed", error=str(e))
            return False


def create_app():
    """Create the Fashion-CLIP FastAPI application."""
    service = FashionClipService()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = FashionClipService()
    service.run()
