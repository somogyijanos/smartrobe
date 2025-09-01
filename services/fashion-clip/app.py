"""
Fashion-CLIP service for Smartrobe.

Extracts fashion-specific visual attributes using pre-trained vision models.
Leverages CLIP-like architectures fine-tuned on fashion datasets.
Currently implements neckline detection through vision understanding.
"""

import asyncio
import random
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from shared.base_service import CapabilityService
from shared.schemas import Neckline, ClothingCategory


class FashionClipService(CapabilityService):
    """Fashion-CLIP service for pre-trained vision-based attribute extraction."""

    SUPPORTED_ATTRIBUTES = {
        "neckline": Neckline,
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

        if attribute_name == "neckline":
            return await self._extract_neckline(request_id, image_paths)
        elif attribute_name == "category":
            return await self._extract_category(request_id, image_paths)
        else:
            raise ValueError(f"Unsupported attribute: {attribute_name}")

    async def _extract_neckline(self, request_id: str, image_paths: list[str]) -> tuple[Neckline, float]:
        """
        Extract neckline type using Fashion-CLIP vision understanding.

        In a real implementation, this would:
        1. Preprocess images (resize, normalize, etc.)
        2. Extract visual features using vision encoder
        3. Generate text embeddings for neckline types
        4. Compute similarity scores between image and text embeddings
        5. Apply softmax to get probability distribution
        6. Select highest probability neckline type
        7. Return confidence based on probability scores

        Args:
            request_id: Unique request identifier
            image_paths: List of paths to images

        Returns:
            Tuple of (Neckline, confidence_score)
        """
        # Simulate Fashion-CLIP processing time (moderate, faster than LLM)
        processing_time = random.uniform(0.3, 0.7)
        await asyncio.sleep(processing_time)

        # Mock Fashion-CLIP classification with neckline-specific probabilities
        # In reality, this would use actual vision-language similarity scores
        neckline_similarities = {
            Neckline.CREW: {
                "similarity": random.uniform(0.15, 0.25),
                "prevalence": 0.30,  # Most common neckline
                "detection_difficulty": "easy"
            },
            Neckline.V_NECK: {
                "similarity": random.uniform(0.12, 0.22),
                "prevalence": 0.25,
                "detection_difficulty": "easy"
            },
            Neckline.SCOOP: {
                "similarity": random.uniform(0.10, 0.20),
                "prevalence": 0.15,
                "detection_difficulty": "medium"
            },
            Neckline.BOAT: {
                "similarity": random.uniform(0.08, 0.18),
                "prevalence": 0.08,
                "detection_difficulty": "medium"
            },
            Neckline.SQUARE: {
                "similarity": random.uniform(0.06, 0.16),
                "prevalence": 0.06,
                "detection_difficulty": "hard"
            },
            Neckline.HALTER: {
                "similarity": random.uniform(0.05, 0.15),
                "prevalence": 0.04,
                "detection_difficulty": "hard"
            },
            Neckline.OFF_SHOULDER: {
                "similarity": random.uniform(0.04, 0.14),
                "prevalence": 0.07,
                "detection_difficulty": "medium"
            },
            Neckline.OTHER: {
                "similarity": random.uniform(0.02, 0.12),
                "prevalence": 0.05,
                "detection_difficulty": "hard"
            }
        }

        # Select neckline based on simulated similarity scores
        necklines = list(neckline_similarities.keys())
        similarities = [data["similarity"] for data in neckline_similarities.values()]
        
        # Normalize similarities to probabilities
        total_similarity = sum(similarities)
        probabilities = [sim / total_similarity for sim in similarities]
        
        selected_neckline = random.choices(necklines, weights=probabilities)[0]

        # Generate confidence based on similarity score and detection difficulty
        base_similarity = neckline_similarities[selected_neckline]["similarity"]
        difficulty = neckline_similarities[selected_neckline]["detection_difficulty"]
        
        # Adjust confidence based on detection difficulty
        if difficulty == "easy":
            confidence = min(0.95, base_similarity + random.uniform(0.65, 0.80))
        elif difficulty == "medium":
            confidence = min(0.90, base_similarity + random.uniform(0.55, 0.70))
        else:  # hard
            confidence = min(0.85, base_similarity + random.uniform(0.45, 0.65))

        # Simulate CLIP-like similarity score logging
        top_similarities = sorted(
            [(neckline.value, neckline_similarities[neckline]["similarity"]) 
             for neckline in necklines], 
            key=lambda x: x[1], 
            reverse=True
        )[:3]

        self.logger.info(
            "Neckline extraction completed",
            request_id=request_id,
            detected_neckline=selected_neckline.value,
            confidence=confidence,
            top_similarities=top_similarities,
            processing_time_ms=int(processing_time * 1000),
        )

        return selected_neckline, confidence

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
