"""
Main orchestrator service for Smartrobe multi-model attribute extraction.

Coordinates image processing across all model services, handles parallel 
execution, and aggregates results for the final API response.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image

from database import InferenceRepository, init_database
from shared.base_service import BaseService
from shared.config import (
    cleanup_request_directory,
    create_request_directory,
    get_service_url,
    get_settings,
)
from shared.logging import get_logger
from shared.schemas import (
    AllAttributes,
    AnalyzeRequest,
    AnalyzeResponse,
    HeuristicAttributes,
    LLMAttributes,
    ModelInfo,
    ProcessingInfo,
    ServiceRequest,
    VisionAttributes,
)


class OrchestratorService(BaseService):
    """Main orchestrator service for coordinating multi-model processing."""

    def __init__(self):
        super().__init__("orchestrator", "1.0.0")
        self.settings = get_settings()
        self.client = httpx.AsyncClient(
            timeout=self.settings.service_request_timeout,
            limits=httpx.Limits(
                max_connections=self.settings.max_concurrent_requests,
                max_keepalive_connections=10,
            ),
        )

    def _add_routes(self, app: FastAPI) -> None:
        """Add orchestrator-specific routes."""

        @app.post("/v1/items/analyze", response_model=AnalyzeResponse)
        async def analyze_item(request: AnalyzeRequest) -> AnalyzeResponse:
            """
            Analyze clothing item from 4 images using all model services.
            
            Process images through vision classifier, heuristic model, and LLM
            services in parallel, then aggregate results with metadata.
            """
            start_time = time.time()
            request_id = uuid.uuid4()
            
            image_count = len(request.images)
            self.logger.info(
                "Item analysis request received",
                request_id=str(request_id),
                image_count=image_count,
            )

            try:
                # Download and validate images
                download_start = time.time()
                image_paths = await self._download_and_validate_images(
                    request_id, request.images
                )
                download_time_ms = int((time.time() - download_start) * 1000)

                # Process through all model services in parallel
                processing_start = time.time()
                
                vision_result, heuristic_result, llm_result = await asyncio.gather(
                    self._call_vision_service(request_id, image_paths),
                    self._call_heuristic_service(request_id, image_paths),
                    self._call_llm_service(request_id, image_paths),
                    return_exceptions=True,
                )

                processing_time_ms = int((time.time() - processing_start) * 1000)

                # Aggregate results
                attributes, model_info = self._aggregate_results(
                    vision_result, heuristic_result, llm_result
                )

                total_time_ms = int((time.time() - start_time) * 1000)

                # Create response
                response = AnalyzeResponse(
                    id=request_id,
                    attributes=attributes,
                    model_info=model_info,
                    processing_info=ProcessingInfo(
                        request_id=request_id,
                        total_processing_time_ms=total_time_ms,
                        image_download_time_ms=download_time_ms,
                        parallel_processing=True,
                        timestamp=datetime.utcnow(),
                        image_count=len(request.images),
                    ),
                )

                # Store in database
                try:
                    await InferenceRepository.create_inference_result(
                        response, {"images": [str(url) for url in request.images]}
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to store inference result",
                        request_id=str(request_id),
                        error=str(e),
                    )
                    # Continue without failing the request

                self.logger.info(
                    "Item analysis completed successfully",
                    request_id=str(request_id),
                    total_time_ms=total_time_ms,
                    download_time_ms=download_time_ms,
                    processing_time_ms=processing_time_ms,
                )

                return response

            except Exception as e:
                total_time_ms = int((time.time() - start_time) * 1000)
                
                self.logger.error(
                    "Item analysis failed",
                    request_id=str(request_id),
                    error=str(e),
                    total_time_ms=total_time_ms,
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )

            finally:
                # Cleanup request directory
                try:
                    cleanup_request_directory(str(request_id))
                except Exception as e:
                    self.logger.warning(
                        "Failed to cleanup request directory",
                        request_id=str(request_id),
                        error=str(e),
                    )

    async def _initialize_service(self) -> None:
        """Initialize orchestrator service."""
        # Initialize database
        await init_database(self.settings.database_url)
        
        # Ensure shared storage directory exists
        os.makedirs(self.settings.shared_storage_path, exist_ok=True)
        
        # Test service connectivity
        await self._test_service_connectivity()

    async def _cleanup_service(self) -> None:
        """Cleanup orchestrator service."""
        await self.client.aclose()

    async def _check_service_health(self) -> bool:
        """Check health of all dependent services."""
        try:
            # Check database
            from database import get_database_manager
            db_manager = get_database_manager()
            if not await db_manager.health_check():
                return False

            # Check model services
            services = ["vision_classifier", "heuristic_model", "llm_model"]
            
            for service_name in services:
                try:
                    service_url = get_service_url(service_name)
                    response = await self.client.get(f"{service_url}/health")
                    if response.status_code != 200:
                        self.logger.warning(
                            f"Service {service_name} health check failed",
                            status_code=response.status_code,
                        )
                        return False
                except Exception as e:
                    self.logger.warning(
                        f"Service {service_name} unreachable",
                        error=str(e),
                    )
                    return False

            return True

        except Exception:
            return False

    async def _download_and_validate_images(
        self, request_id: uuid.UUID, image_urls: list[str]
    ) -> list[str]:
        """
        Download and validate images from URLs.
        
        Args:
            request_id: Unique request identifier
            image_urls: List of HTTPS image URLs
            
        Returns:
            List of local file paths to downloaded images
        """
        # Create request-specific directory
        request_dir = create_request_directory(str(request_id))
        image_paths = []

        try:
            for i, url in enumerate(image_urls):
                self.logger.debug(
                    "Downloading image",
                    request_id=str(request_id),
                    image_index=i,
                    url=str(url),
                )

                # Download image
                response = await self.client.get(
                    str(url),
                    timeout=self.settings.image_download_timeout,
                )
                response.raise_for_status()

                # Check file size
                content_length = len(response.content)
                max_size_bytes = self.settings.max_image_size_mb * 1024 * 1024
                
                if content_length > max_size_bytes:
                    raise ValueError(
                        f"Image {i} too large: {content_length} bytes "
                        f"(max: {max_size_bytes} bytes)"
                    )

                # Save image
                image_path = os.path.join(request_dir, f"image_{i}.jpg")
                with open(image_path, "wb") as f:
                    f.write(response.content)

                # Validate image format
                try:
                    with Image.open(image_path) as img:
                        # Convert to RGB if needed and save as JPEG
                        if img.format.lower() not in self.settings.allowed_image_formats:
                            raise ValueError(
                                f"Image {i} format not supported: {img.format}"
                            )
                        
                        # Ensure RGB format for consistency
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                            img.save(image_path, "JPEG", quality=95)

                except Exception as e:
                    raise ValueError(f"Image {i} validation failed: {str(e)}")

                image_paths.append(image_path)

                self.logger.debug(
                    "Image downloaded and validated",
                    request_id=str(request_id),
                    image_index=i,
                    file_size=content_length,
                    local_path=image_path,
                )

            return image_paths

        except Exception:
            # Cleanup on failure
            cleanup_request_directory(str(request_id))
            raise

    async def _call_vision_service(
        self, request_id: uuid.UUID, image_paths: list[str]
    ) -> dict[str, Any]:
        """Call vision classifier service."""
        try:
            service_url = get_service_url("vision_classifier")
            
            request_data = ServiceRequest(
                request_id=request_id, image_paths=image_paths
            )

            response = await self.client.post(
                f"{service_url}/extract",
                json=request_data.model_dump(mode='json'),
            )
            response.raise_for_status()
            
            return response.json()

        except Exception as e:
            self.logger.error(
                "Vision service call failed",
                request_id=str(request_id),
                error=str(e),
            )
            # Return error result instead of raising
            return {
                "request_id": str(request_id),
                "success": False,
                "attributes": {},
                "confidence_scores": {},
                "processing_time_ms": 0,
                "error_message": str(e),
            }

    async def _call_heuristic_service(
        self, request_id: uuid.UUID, image_paths: list[str]
    ) -> dict[str, Any]:
        """Call heuristic model service."""
        try:
            service_url = get_service_url("heuristic_model")
            
            request_data = ServiceRequest(
                request_id=request_id, image_paths=image_paths
            )

            response = await self.client.post(
                f"{service_url}/extract",
                json=request_data.model_dump(mode='json'),
            )
            response.raise_for_status()
            
            return response.json()

        except Exception as e:
            self.logger.error(
                "Heuristic service call failed",
                request_id=str(request_id),
                error=str(e),
            )
            return {
                "request_id": str(request_id),
                "success": False,
                "attributes": {},
                "confidence_scores": {},
                "processing_time_ms": 0,
                "error_message": str(e),
            }

    async def _call_llm_service(
        self, request_id: uuid.UUID, image_paths: list[str]
    ) -> dict[str, Any]:
        """Call LLM service."""
        try:
            service_url = get_service_url("llm_model")
            
            request_data = ServiceRequest(
                request_id=request_id, image_paths=image_paths
            )

            response = await self.client.post(
                f"{service_url}/extract",
                json=request_data.model_dump(mode='json'),
            )
            response.raise_for_status()
            
            return response.json()

        except Exception as e:
            self.logger.error(
                "LLM service call failed",
                request_id=str(request_id),
                error=str(e),
            )
            return {
                "request_id": str(request_id),
                "success": False,
                "attributes": {},
                "confidence_scores": {},
                "processing_time_ms": 0,
                "error_message": str(e),
            }

    def _aggregate_results(
        self, 
        vision_result: dict[str, Any],
        heuristic_result: dict[str, Any],
        llm_result: dict[str, Any],
    ) -> tuple[AllAttributes, dict[str, ModelInfo]]:
        """
        Aggregate results from all model services.
        
        Args:
            vision_result: Results from vision classifier
            heuristic_result: Results from heuristic model
            llm_result: Results from LLM service
            
        Returns:
            Tuple of (aggregated_attributes, model_info)
        """
        # Handle service failures gracefully
        def safe_get_attributes(result, default_class):
            if isinstance(result, Exception) or not result.get("success", False):
                return default_class().model_dump(), {}
            return result.get("attributes", {}), result.get("confidence_scores", {})

        vision_attrs, vision_confidence = safe_get_attributes(vision_result, VisionAttributes)
        heuristic_attrs, heuristic_confidence = safe_get_attributes(heuristic_result, HeuristicAttributes)
        llm_attrs, llm_confidence = safe_get_attributes(llm_result, LLMAttributes)

        # Combine all attributes
        all_attributes = AllAttributes(
            # Vision attributes
            **vision_attrs,
            # Heuristic attributes
            **heuristic_attrs,
            # LLM attributes
            **llm_attrs,
        )

        # Create model info
        model_info = {
            "vision_classifier": ModelInfo(
                model_type="vision_classifier",
                processing_time_ms=vision_result.get("processing_time_ms", 0),
                confidence_scores=vision_confidence,
                success=vision_result.get("success", False),
                error_message=vision_result.get("error_message"),
            ),
            "heuristic_model": ModelInfo(
                model_type="heuristic_model",
                processing_time_ms=heuristic_result.get("processing_time_ms", 0),
                confidence_scores=heuristic_confidence,
                success=heuristic_result.get("success", False),
                error_message=heuristic_result.get("error_message"),
            ),
            "llm_model": ModelInfo(
                model_type="llm_model",
                processing_time_ms=llm_result.get("processing_time_ms", 0),
                confidence_scores=llm_confidence,
                success=llm_result.get("success", False),
                error_message=llm_result.get("error_message"),
            ),
        }

        return all_attributes, model_info

    async def _test_service_connectivity(self) -> None:
        """Test connectivity to all model services."""
        services = ["vision_classifier", "heuristic_model", "llm_model"]
        
        for service_name in services:
            try:
                service_url = get_service_url(service_name)
                response = await self.client.get(f"{service_url}/health")
                
                if response.status_code == 200:
                    self.logger.info(f"Service {service_name} is reachable")
                else:
                    self.logger.warning(
                        f"Service {service_name} returned status {response.status_code}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Service {service_name} is not reachable: {str(e)}"
                )


def create_app() -> FastAPI:
    """Create the orchestrator FastAPI application."""
    service = OrchestratorService()
    return service.app


# For development/testing
if __name__ == "__main__":
    service = OrchestratorService()
    service.run()
