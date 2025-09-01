"""
Microsoft Florence-2 service for Smartrobe.

Provides garment detection, bounding box extraction, and image cropping functionality
using Microsoft's Florence-2 vision model for open vocabulary object detection.
"""

import asyncio
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from fastapi import HTTPException

from shared.base_service import BaseService
from shared.schemas import ServiceRequest, ServiceResponse


class MicrosoftFlorenceService(BaseService):
    """Microsoft Florence-2 service for garment detection and cropping."""

    def __init__(self):
        super().__init__("microsoft-florence", "1.0.0")
        self.model = None
        self.processor = None
        self.model_id = 'microsoft/Florence-2-base'

    async def _initialize_service(self) -> None:
        """Initialize the Florence-2 model."""
        self.logger.info("Loading Microsoft Florence-2 model...")
        
        try:
            # Load model and processor
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            self.logger.info("Microsoft Florence-2 model loaded successfully")
            self.set_health_detail("model_loaded", True)
            self.set_health_detail("processor_loaded", True)
            
        except Exception as e:
            self.logger.error(f"Failed to load Florence-2 model: {str(e)}")
            raise

    async def _cleanup_service(self) -> None:
        """Cleanup Florence-2 service."""
        self.model = None
        self.processor = None
        self.logger.info("Microsoft Florence-2 service cleaned up")

    def _add_routes(self, app) -> None:
        """Add Florence-2 specific routes."""
        
        @app.post("/detect_garment")
        async def detect_garment(request: Dict[str, Any]) -> Dict[str, Any]:
            """
            Detect garment in a single image and return bounding box information.
            
            Request format:
            {
                "image_path": "/path/to/image.jpg"
            }
            
            Response format:
            {
                "success": bool,
                "bboxes": [[x1, y1, x2, y2], ...],
                "bboxes_labels": ["garment", ...],
                "is_close_up": bool,
                "processing_time_ms": int,
                "error_message": str | None
            }
            """
            try:
                image_path = request.get("image_path")
                if not image_path:
                    raise ValueError("image_path is required")
                
                start_time = time.time()
                
                # Load image
                image = Image.open(image_path)
                
                # Detect garment
                result = await self._run_garment_detection(image)
                
                # Check if it's a close-up
                is_close_up = self._is_close_up(image, result)
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "bboxes": result.get("bboxes", []),
                    "bboxes_labels": result.get("bboxes_labels", []),
                    "is_close_up": is_close_up,
                    "processing_time_ms": processing_time_ms,
                    "error_message": None
                }
                
            except Exception as e:
                self.logger.error(f"Garment detection failed: {str(e)}")
                return {
                    "success": False,
                    "bboxes": [],
                    "bboxes_labels": [],
                    "is_close_up": False,
                    "processing_time_ms": 0,
                    "error_message": str(e)
                }

        @app.post("/crop_garment")
        async def crop_garment(request: Dict[str, Any]) -> Dict[str, Any]:
            """
            Detect garment and return cropped image path.
            
            Request format:
            {
                "image_path": "/path/to/image.jpg",
                "output_path": "/path/to/output.jpg"  # optional
            }
            
            Response format:
            {
                "success": bool,
                "cropped_image_path": str | None,
                "is_close_up": bool,
                "bbox": [x1, y1, x2, y2] | None,
                "processing_time_ms": int,
                "error_message": str | None
            }
            """
            try:
                image_path = request.get("image_path")
                output_path = request.get("output_path")
                
                if not image_path:
                    raise ValueError("image_path is required")
                    
                start_time = time.time()
                
                # Load image
                image = Image.open(image_path)
                
                # Detect garment
                result = await self._run_garment_detection(image)
                
                # Check if it's a close-up
                is_close_up = self._is_close_up(image, result)
                
                # Crop garment if found
                cropped_image, bbox = self._crop_to_garment(image, result)
                
                cropped_image_path = None
                if cropped_image is not None:
                    if output_path is None:
                        # Generate output path based on input path
                        import os
                        base, ext = os.path.splitext(image_path)
                        output_path = f"{base}_cropped{ext}"
                    
                    cropped_image.save(output_path)
                    cropped_image_path = output_path
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "cropped_image_path": cropped_image_path,
                    "is_close_up": is_close_up,
                    "bbox": bbox,
                    "processing_time_ms": processing_time_ms,
                    "error_message": None
                }
                
            except Exception as e:
                self.logger.error(f"Garment cropping failed: {str(e)}")
                return {
                    "success": False,
                    "cropped_image_path": None,
                    "is_close_up": False,
                    "bbox": None,
                    "processing_time_ms": 0,
                    "error_message": str(e)
                }

        @app.post("/process_images")
        async def process_images(request: Dict[str, Any]) -> Dict[str, Any]:
            """
            Process multiple images: filter non-close-ups and crop garments.
            
            Request format:
            {
                "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg", ...],
                "output_dir": "/path/to/output/dir"  # optional
            }
            
            Response format:
            {
                "success": bool,
                "processed_images": [
                    {
                        "original_path": str,
                        "cropped_path": str | None,
                        "is_close_up": bool,
                        "bbox": [x1, y1, x2, y2] | None
                    },
                    ...
                ],
                "non_close_up_cropped_paths": [str, ...],  # Only non-close-up cropped images
                "processing_time_ms": int,
                "error_message": str | None
            }
            """
            try:
                image_paths = request.get("image_paths", [])
                output_dir = request.get("output_dir")
                
                if not image_paths:
                    raise ValueError("image_paths is required")
                    
                start_time = time.time()
                processed_images = []
                non_close_up_cropped_paths = []
                
                for i, image_path in enumerate(image_paths):
                    try:
                        # Load image
                        image = Image.open(image_path)
                        
                        # Detect garment
                        result = await self._run_garment_detection(image)
                        
                        # Check if it's a close-up
                        is_close_up = self._is_close_up(image, result)
                        
                        # Crop garment if found
                        cropped_image, bbox = self._crop_to_garment(image, result)
                        
                        cropped_path = None
                        if cropped_image is not None:
                            if output_dir:
                                import os
                                filename = os.path.basename(image_path)
                                base, ext = os.path.splitext(filename)
                                cropped_path = os.path.join(output_dir, f"{base}_cropped{ext}")
                            else:
                                # Generate output path based on input path
                                import os
                                base, ext = os.path.splitext(image_path)
                                cropped_path = f"{base}_cropped{ext}"
                            
                            cropped_image.save(cropped_path)
                            
                            # Add to non-close-up list if it's not a close-up
                            if not is_close_up:
                                non_close_up_cropped_paths.append(cropped_path)
                        
                        processed_images.append({
                            "original_path": image_path,
                            "cropped_path": cropped_path,
                            "is_close_up": is_close_up,
                            "bbox": bbox
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {image_path}: {str(e)}")
                        processed_images.append({
                            "original_path": image_path,
                            "cropped_path": None,
                            "is_close_up": False,
                            "bbox": None
                        })
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "processed_images": processed_images,
                    "non_close_up_cropped_paths": non_close_up_cropped_paths,
                    "processing_time_ms": processing_time_ms,
                    "error_message": None
                }
                
            except Exception as e:
                self.logger.error(f"Image processing failed: {str(e)}")
                return {
                    "success": False,
                    "processed_images": [],
                    "non_close_up_cropped_paths": [],
                    "processing_time_ms": 0,
                    "error_message": str(e)
                }

    async def _run_garment_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Run Florence-2 garment detection on an image."""
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        text_input = "garment"
        
        if self.processor is None or self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Prepare prompt
        prompt = task_prompt + text_input
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Generate
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        # Decode and parse
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer.get('<OPEN_VOCABULARY_DETECTION>', {})

    def _crop_to_garment(self, image: Image.Image, result: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[List[float]]]:
        """Crop image to garment bounding box if garment is detected."""
        bboxes_labels = result.get('bboxes_labels', [])
        bboxes = result.get('bboxes', [])
        
        if 'garment' in bboxes_labels:
            label_idx = bboxes_labels.index('garment')
            bbox = bboxes[label_idx]
            x1, y1, x2, y2 = bbox
            cropped_image = image.crop((x1, y1, x2, y2))
            return cropped_image, bbox
        else:
            return None, None

    def _is_close_up(self, image: Image.Image, result: Dict[str, Any]) -> bool:
        """Determine if the image is a close-up based on garment bounding box size."""
        bboxes_labels = result.get('bboxes_labels', [])
        bboxes = result.get('bboxes', [])
        
        if 'garment' in bboxes_labels and bboxes:
            img_size = image.size
            bbox = bboxes[0]  # Use first garment bbox
            x1, y1, x2, y2 = bbox
            
            diff_x = (x2 - x1) / img_size[0]
            diff_y = (y2 - y1) / img_size[1]
            
            # Consider it a close-up if garment takes up >95% of image in both dimensions
            return diff_x > 0.95 and diff_y > 0.95
        else:
            return False

    async def _check_service_health(self) -> bool:
        """Check Microsoft Florence-2 service health."""
        try:
            if self.model is None or self.processor is None:
                return False
            
            # Test with a small dummy image
            test_image = Image.new('RGB', (64, 64), color='red')
            
            start_time = time.time()
            await self._run_garment_detection(test_image)
            health_check_time = time.time() - start_time
            
            self.set_health_detail("last_health_check_ms", int(health_check_time * 1000))
            self.set_health_detail("model_ready", True)
            
            return True
        except Exception as e:
            self.logger.error("Microsoft Florence-2 service health check failed", error=str(e))
            return False


def create_app():
    """Create the Microsoft Florence-2 FastAPI application."""
    service = MicrosoftFlorenceService()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = MicrosoftFlorenceService()
    service.run()
