"""
Segmentation service for Smartrobe.

Extracts garments from background using rembg for improved attribute detection.
Provides both batch and individual image segmentation capabilities.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import rembg
from PIL import Image

from shared.base_service import BaseService
from shared.config import get_model_service_settings
from shared.schemas import SegmentationRequest, SegmentationResponse


class SegmentationRembgService(BaseService):
    """Segmentation service for garment background removal using rembg."""

    def __init__(self):
        super().__init__("segmentation_rembg", "1.0.0", settings=get_model_service_settings())
        self.rembg_session = None
        self.model_name = "u2net"  # Default model for general purpose segmentation
        
        # Configure shared model storage
        self.shared_storage_path = Path(self.settings.shared_storage_path)
        self.models_dir = self.shared_storage_path / "models" / "rembg"
        
        # Set rembg to use shared storage for models
        os.environ["U2NET_HOME"] = str(self.models_dir)

    async def _initialize_service(self) -> None:
        """Initialize segmentation service and load rembg model."""
        self.logger.info("Initializing rembg segmentation service")
        
        # Create models directory in shared storage
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Using shared storage for models",
            models_dir=str(self.models_dir),
            exists=self.models_dir.exists(),
        )
        
        try:
            # Check if model already exists in shared storage
            model_file = self.models_dir / f"{self.model_name}.onnx"
            model_exists = model_file.exists()
            
            if model_exists:
                self.logger.info(
                    "Using cached model from shared storage",
                    model=self.model_name,
                    model_path=str(model_file),
                    model_size_mb=round(model_file.stat().st_size / (1024 * 1024), 1),
                )
            else:
                self.logger.info(
                    "Model not found in cache, will download",
                    model=self.model_name,
                    expected_path=str(model_file),
                )
            
            # Initialize rembg session - this downloads the model if not cached
            loop = asyncio.get_event_loop()
            self.rembg_session = await loop.run_in_executor(
                None, rembg.new_session, self.model_name
            )
            
            # Log final status
            if model_exists:
                self.logger.info(
                    "Segmentation model loaded from cache",
                    model=self.model_name,
                )
            else:
                self.logger.info(
                    "Segmentation model downloaded and loaded",
                    model=self.model_name,
                    model_size="~176MB",
                )
                # Verify the model was actually saved to shared storage
                if model_file.exists():
                    self.logger.info(
                        "Model cached to shared storage for future use",
                        model_path=str(model_file),
                    )
            
            self.set_health_detail("model_loaded", True)
            self.set_health_detail("model_name", self.model_name)
            self.set_health_detail("model_storage", str(self.models_dir))
            self.set_health_detail("model_cached", model_exists)
            self.set_health_detail("rembg_version", rembg.__version__)
            
        except Exception as e:
            self.logger.error("Failed to load segmentation model", error=str(e))
            self.set_unhealthy(f"Model loading failed: {str(e)}")
            raise

    async def _cleanup_service(self) -> None:
        """Cleanup segmentation service."""
        self.logger.info("Segmentation service cleaned up")
        # rembg sessions don't need explicit cleanup

    def _add_routes(self, app) -> None:
        """Add segmentation routes."""
        
        @app.post("/segment", response_model=SegmentationResponse)
        async def segment_images(request: SegmentationRequest) -> SegmentationResponse:
            """
            Segment garments from background in batch of images.
            
            Processes all images and returns both original and segmented paths.
            If segmentation fails for any image, marks it as failed but continues with others.
            """
            start_time = time.time()
            
            self.logger.info(
                "Segmentation request received",
                request_id=str(request.request_id),
                image_count=len(request.image_paths),
                model=request.model or self.model_name,
            )

            try:
                segmented_paths, success_mask = await self._segment_image_batch(
                    str(request.request_id), 
                    request.image_paths,
                    request.model or self.model_name,
                    request.output_format
                )
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                successful_count = sum(success_mask)
                
                self.logger.info(
                    "Segmentation completed",
                    request_id=str(request.request_id),
                    successful_segmentations=successful_count,
                    total_images=len(request.image_paths),
                    processing_time_ms=processing_time_ms,
                )

                return SegmentationResponse(
                    request_id=request.request_id,
                    original_paths=request.image_paths,
                    segmented_paths=segmented_paths,
                    success_mask=success_mask,
                    processing_time_ms=processing_time_ms,
                    success=True,
                )

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                self.logger.error(
                    "Segmentation request failed",
                    request_id=str(request.request_id),
                    error=str(e),
                    processing_time_ms=processing_time_ms,
                )
                
                # Return failed response with original paths as fallback
                return SegmentationResponse(
                    request_id=request.request_id,
                    original_paths=request.image_paths,
                    segmented_paths=request.image_paths,  # Fallback to originals
                    success_mask=[False] * len(request.image_paths),
                    processing_time_ms=processing_time_ms,
                    success=False,
                    error_message=str(e),
                )

    async def _segment_image_batch(
        self, 
        request_id: str, 
        image_paths: list[str], 
        model: str,
        output_format: str
    ) -> tuple[list[str], list[bool]]:
        """
        Segment a batch of images.
        
        Args:
            request_id: Unique request identifier
            image_paths: List of paths to original images
            model: rembg model to use (for future extensibility)
            output_format: Output format (png/jpg)
            
        Returns:
            Tuple of (segmented_paths, success_mask)
        """
        segmented_paths = []
        success_mask = []
        
        # Process images concurrently but limit concurrency to avoid memory issues
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent segmentations
        
        async def segment_single_image(i: int, image_path: str) -> tuple[str, bool]:
            """Segment a single image with concurrency control."""
            async with semaphore:
                return await self._segment_single_image(
                    request_id, i, image_path, output_format
                )
        
        # Create tasks for all images
        tasks = [
            segment_single_image(i, path) 
            for i, path in enumerate(image_paths)
        ]
        
        # Execute all segmentations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Segmentation failed for image {i}: {type(result).__name__}: {str(result)}",
                    request_id=request_id,
                    image_path=image_paths[i],
                    error_type=type(result).__name__,
                    error_details=str(result),
                )
                segmented_paths.append(image_paths[i])  # Fallback to original
                success_mask.append(False)
            else:
                segmented_path, success = result
                segmented_paths.append(segmented_path)
                success_mask.append(success)
        
        return segmented_paths, success_mask

    async def _segment_single_image(
        self, request_id: str, image_index: int, image_path: str, output_format: str
    ) -> tuple[str, bool]:
        """
        Segment a single image using rembg.
        
        Args:
            request_id: Unique request identifier
            image_index: Index of image in batch
            image_path: Path to original image
            output_format: Output format
            
        Returns:
            Tuple of (segmented_path, success)
        """
        try:
            # Validate input file exists and is readable
            input_path = Path(image_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")
            
            if not input_path.is_file():
                raise ValueError(f"Input path is not a file: {image_path}")
            
            # Check file size
            file_size = input_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Input image file is empty: {image_path}")
            
            # Generate output path
            output_path = input_path.parent / f"{input_path.stem}_segmented.{output_format}"
            
            self.logger.debug(
                f"Segmenting image {image_index}",
                request_id=request_id,
                input_path=str(image_path),
                output_path=str(output_path),
                input_size_kb=file_size // 1024,
                input_exists=input_path.exists(),
            )

            # Load input image
            loop = asyncio.get_event_loop()
            try:
                input_image = await loop.run_in_executor(None, Image.open, image_path)
                self.logger.debug(
                    f"Image {image_index} loaded successfully",
                    request_id=request_id,
                    image_format=input_image.format,
                    image_mode=input_image.mode,
                    image_size=input_image.size,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load image with PIL: {str(e)}") from e
            
            # Perform segmentation (CPU intensive, run in executor)
            try:
                output_image = await loop.run_in_executor(
                    None, rembg.remove, input_image, self.rembg_session
                )
                self.logger.debug(
                    f"Image {image_index} segmentation processing completed",
                    request_id=request_id,
                    output_mode=output_image.mode,
                    output_size=output_image.size,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to segment image with rembg: {str(e)}") from e
            
            # Save segmented image
            try:
                await loop.run_in_executor(None, output_image.save, str(output_path))
                self.logger.debug(
                    f"Image {image_index} saved successfully",
                    request_id=request_id,
                    output_path=str(output_path),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to save segmented image: {str(e)}") from e
            
            # Validate output
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ValueError("Segmented image is empty or not created")
            
            self.logger.debug(
                f"Image {image_index} segmented successfully",
                request_id=request_id,
                output_path=str(output_path),
                output_size_kb=output_path.stat().st_size // 1024,
            )
            
            return str(output_path), True
            
        except Exception as e:
            # Log with detailed error information in the main message
            self.logger.error(
                f"Failed to segment image {image_index}: {type(e).__name__}: {str(e)}",
                request_id=request_id,
                input_path=image_path,
                error_type=type(e).__name__,
                error_details=str(e),
            )
            # Also log the full traceback for debugging
            import traceback
            self.logger.debug(
                f"Full traceback for image {image_index} segmentation failure",
                request_id=request_id,
                traceback=traceback.format_exc(),
            )
            return image_path, False  # Return original path as fallback

    async def _check_service_health(self) -> bool:
        """Check segmentation service health."""
        try:
            # Test that rembg session is available
            if self.rembg_session is None:
                self.set_health_detail("model_status", "not_loaded")
                return False
            
            # Quick health check
            self.set_health_detail("model_status", "loaded")
            self.set_health_detail("last_health_check", time.time())
            
            return True
            
        except Exception as e:
            self.logger.error("Segmentation service health check failed", error=str(e))
            self.set_health_detail("health_check_error", str(e))
            return False


def create_app():
    """Create the segmentation rembg FastAPI application."""
    service = SegmentationRembgService()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = SegmentationRembgService()
    service.run()
