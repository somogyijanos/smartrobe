"""
New capability-based orchestrator service for Smartrobe.

Routes attribute extraction requests to appropriate services based on YAML configuration.
Handles unimplemented attributes with warnings and supports dynamic service routing.
"""

import asyncio
import os
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image

from shared.database import InferenceRepository, init_database
from shared.base_service import BaseService
from shared.config import create_request_directory, cleanup_request_directory, get_settings
from shared.schemas import (
    AllAttributes,
    AnalyzeRequest,
    AnalyzeResponse,
    AttributeModelInfo,
    AttributeRequest,
    ProcessingInfo,
    SegmentationRequest,
    SegmentationResponse,
)


class CapabilityOrchestrator(BaseService):
    """Capability-based orchestrator for dynamic attribute routing."""

    # All 13 required attributes
    ALL_ATTRIBUTES = [
        "category", "gender", "sleeve_length", "neckline", "closure_type", "fit",
        "color", "material", "pattern", "brand", "style", "season", "condition"
    ]

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
        
        # Load attribute routing configuration
        self.attribute_config = self._load_attribute_config()
        self.service_urls = self._build_service_url_mapping()

    def _load_attribute_config(self) -> dict[str, Any]:
        """Load attribute routing configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "attribute_routing.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(
                "Attribute routing configuration loaded",
                config_file=str(config_path),
                implemented_attributes=list(config["attribute_mappings"].keys()),
                total_services=len(config["services"])
            )
            
            return config
            
        except Exception as e:
            self.logger.error(
                "Failed to load attribute routing configuration",
                config_file=str(config_path),
                error=str(e)
            )
            # Return minimal config to avoid complete failure
            return {
                "attribute_mappings": {},
                "services": [],
                "routing_config": {
                    "unimplemented_behavior": "return_null_with_warning",
                    "default_timeout_seconds": 30
                }
            }

    def _build_service_url_mapping(self) -> dict[str, str]:
        """Build mapping of service names to their URLs from environment variables."""
        # Use environment variables for service URLs (12-factor app approach)
        url_mapping = {
            "heuristics": self.settings.heuristics_service_url,
            "llm_multimodal": self.settings.llm_multimodal_service_url,
            "fashion_clip": self.settings.fashion_clip_service_url,
            "segmentation_rembg": self.settings.segmentation_rembg_service_url,
        }
            
        self.logger.info(
            "Service URL mapping built from environment variables",
            services=list(url_mapping.keys()),
            urls=list(url_mapping.values())
        )
        
        return url_mapping

    def _add_routes(self, app: FastAPI) -> None:
        """Add orchestrator routes."""

        @app.post("/v1/items/analyze", response_model=AnalyzeResponse)
        async def analyze_item(request: AnalyzeRequest) -> AnalyzeResponse:
            """
            Analyze clothing item using capability-based routing.
            
            Routes each attribute to its configured service and handles
            unimplemented attributes with warnings.
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

                # Route attributes to services (includes parallel segmentation)
                processing_start = time.time()
                
                attributes, model_info = await self._route_attributes_to_services(
                    request_id, image_paths
                )
                
                processing_time_ms = int((time.time() - processing_start) * 1000)
                total_time_ms = int((time.time() - start_time) * 1000)

                # Determine implemented vs skipped attributes
                implemented_attrs = [attr for attr in self.ALL_ATTRIBUTES 
                                   if getattr(attributes, attr) is not None]
                skipped_attrs = [attr for attr in self.ALL_ATTRIBUTES 
                               if getattr(attributes, attr) is None]

                # Create response
                response = AnalyzeResponse(
                    id=request_id,
                    attributes=attributes,
                    model_info=model_info,
                    processing=ProcessingInfo(
                        request_id=request_id,
                        total_processing_time_ms=total_time_ms,
                        image_download_time_ms=download_time_ms,
                        timestamp=datetime.utcnow(),
                        image_count=image_count,
                        implemented_attributes=implemented_attrs,
                        skipped_attributes=skipped_attrs,
                    ),
                )

                # Store in database
                database_stored = False
                try:
                    await InferenceRepository.create_inference_result(
                        response, {"images": [str(url) for url in request.images]}
                    )
                    database_stored = True
                except Exception as e:
                    # Log with just the essential error details
                    self.logger.error(
                        f"Failed to store inference result: {type(e).__name__}: {str(e)}",
                        request_id=str(request_id),
                    )
                    # Don't swallow the exception - let it bubble up or handle appropriately
                    # For now, we'll continue processing but mark the issue clearly

                # Note: Even if database storage failed, the analysis itself succeeded
                self.logger.info(
                    "Item analysis completed",
                    request_id=str(request_id),
                    total_time_ms=total_time_ms,
                    implemented_count=len(implemented_attrs),
                    skipped_count=len(skipped_attrs),
                    database_stored=database_stored,
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
                # Cleanup request directory (unless debug mode retains images)
                if not self.settings.debug_retain_images:
                    try:
                        cleanup_request_directory(str(request_id))
                    except Exception as e:
                        self.logger.warning(
                            "Failed to cleanup request directory",
                            request_id=str(request_id),
                            error=str(e),
                        )
                else:
                    self.logger.info(
                        "Debug mode: retaining images in request directory",
                        request_id=str(request_id),
                        directory=f"{self.settings.shared_storage_path}/{request_id}",
                    )

    async def _route_attributes_to_services(
        self, request_id: uuid.UUID, image_paths: list[str]
    ) -> tuple[AllAttributes, dict[str, AttributeModelInfo]]:
        """
        Route attributes to their configured services with parallel segmentation.
        
        Returns:
            Tuple of (attributes, model_info)
        """
        attribute_mappings = self.attribute_config["attribute_mappings"]
        
        # Separate implemented from unimplemented attributes
        implemented_attrs = []
        unimplemented_attrs = []
        
        for attr in self.ALL_ATTRIBUTES:
            if attr in attribute_mappings:
                implemented_attrs.append(attr)
            else:
                unimplemented_attrs.append(attr)

        self.logger.info(
            "Routing attributes to services with segmentation support",
            request_id=str(request_id),
            implemented=implemented_attrs,
            unimplemented=unimplemented_attrs,
        )

        # Check if segmentation is needed for any services
        segmentation_needed = self._check_segmentation_needed(implemented_attrs)
        
        # Handle segmentation first if needed
        segmentation_response = None
        if segmentation_needed:
            segmentation_response = await self._call_segmentation_service(request_id, image_paths)
        
        # Start attribute extraction tasks with segmentation result
        attr_tasks = []
        for attr in implemented_attrs:
            service_name = attribute_mappings[attr]
            task = self._call_attribute_service_with_segmentation(
                request_id, attr, service_name, image_paths, segmentation_response
            )
            attr_tasks.append(task)

        # Execute all attribute extraction tasks in parallel
        attr_results = await asyncio.gather(*attr_tasks, return_exceptions=True)

        # Process results
        attribute_values = {}
        model_info = {}

        # Handle implemented attributes
        for i, (attr, result) in enumerate(zip(implemented_attrs, attr_results)):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Attribute extraction failed for {attr}",
                    request_id=str(request_id),
                    attribute=attr,
                    error=str(result),
                )
                attribute_values[attr] = None
            else:
                attr_response, model_info_item = result
                attribute_values[attr] = attr_response.attribute_value
                model_info[attr] = model_info_item

        # Handle unimplemented attributes - just set to None
        for attr in unimplemented_attrs:
            attribute_values[attr] = None

        # Create AllAttributes object
        attributes = AllAttributes(**attribute_values)

        return attributes, model_info

    def _check_segmentation_needed(self, implemented_attrs: list[str]) -> bool:
        """
        Check if segmentation is needed for any of the implemented attributes.
        
        Args:
            implemented_attrs: List of attributes to be extracted
            
        Returns:
            True if segmentation is needed, False otherwise
        """
        segmentation_config = self.attribute_config.get("segmentation_config", {})
        if not segmentation_config.get("enabled", False):
            return False
        
        attribute_mappings = self.attribute_config["attribute_mappings"]
        
        for attr in implemented_attrs:
            service_name = attribute_mappings[attr]
            # Check if the service for this attribute has segmentation enabled
            for service in self.attribute_config["services"]:
                if service["name"] == service_name:
                    if service.get("segmentation_enabled", False):
                        self.logger.debug(
                            f"Segmentation needed for attribute '{attr}' (service: {service_name})"
                        )
                        return True
        
        return False

    async def _call_segmentation_service(
        self, request_id: uuid.UUID, image_paths: list[str]
    ) -> SegmentationResponse:
        """
        Call the segmentation service to segment images.
        
        Args:
            request_id: Unique request identifier
            image_paths: List of paths to original images
            
        Returns:
            SegmentationResponse with segmented image paths
        """
        segmentation_config = self.attribute_config.get("segmentation_config", {})
        
        # Get segmentation service URL from environment variables  
        segmentation_url = self.service_urls.get("segmentation_rembg")
        if not segmentation_url:
            raise ValueError("Segmentation service URL not found in environment variables")

        endpoint_url = f"{segmentation_url}/segment"
        
        request_data = SegmentationRequest(
            request_id=request_id,
            image_paths=image_paths,
            model=segmentation_config.get("model", "u2net"),
            output_format=segmentation_config.get("output_format", "png"),
        )

        self.logger.info(
            "Calling segmentation service",
            request_id=str(request_id),
            endpoint_url=endpoint_url,
            image_count=len(image_paths),
        )

        start_time = time.time()
        
        try:
            response = await self.client.post(
                endpoint_url,
                json=request_data.model_dump(mode='json'),
            )
            response.raise_for_status()
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            result = response.json()
            
            segmentation_response = SegmentationResponse(**result)
            
            successful_count = sum(segmentation_response.success_mask)
            self.logger.info(
                "Segmentation service call completed",
                request_id=str(request_id),
                successful_segmentations=successful_count,
                total_images=len(image_paths),
                processing_time_ms=processing_time_ms,
            )

            return segmentation_response

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.error(
                "Segmentation service call failed",
                request_id=str(request_id),
                error=str(e),
                processing_time_ms=processing_time_ms,
            )
            
            # Return fallback response with original images
            return SegmentationResponse(
                request_id=request_id,
                original_paths=image_paths,
                segmented_paths=image_paths,  # Fallback to originals
                success_mask=[False] * len(image_paths),
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e),
            )

    async def _call_attribute_service_with_segmentation(
        self, 
        request_id: uuid.UUID, 
        attribute_name: str, 
        service_name: str, 
        image_paths: list[str], 
        segmentation_response: SegmentationResponse | None
    ) -> tuple[Any, AttributeModelInfo]:
        """
        Call an attribute service with optional segmented images.
        
        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            service_name: Name of the service to call
            image_paths: List of paths to original images
            segmentation_response: Optional segmentation response (if segmentation was performed)
            
        Returns:
            Tuple of (AttributeResponse, AttributeModelInfo)
        """
        # Check if this service needs segmentation
        service_needs_segmentation = False
        for service in self.attribute_config["services"]:
            if service["name"] == service_name:
                service_needs_segmentation = service.get("segmentation_enabled", False)
                break

        # Determine which image paths to use
        segmented_image_paths = None
        use_segmented = False
        
        if service_needs_segmentation and segmentation_response:
            if segmentation_response.success:
                # Use mixed mode: segmented where available, original as fallback
                mixed_paths = []
                for i, (original, segmented, success) in enumerate(zip(
                    segmentation_response.original_paths,
                    segmentation_response.segmented_paths,
                    segmentation_response.success_mask
                )):
                    mixed_paths.append(segmented if success else original)
                
                segmented_image_paths = mixed_paths
                use_segmented = True
                
                self.logger.debug(
                    f"Using segmented images for {attribute_name}",
                    request_id=str(request_id),
                    successful_segmentations=sum(segmentation_response.success_mask),
                    total_images=len(image_paths),
                )
            else:
                self.logger.warning(
                    f"Segmentation failed for {attribute_name}, using original images",
                    request_id=str(request_id),
                    error=segmentation_response.error_message,
                )

        # Call the attribute service
        return await self._call_attribute_service(
            request_id, attribute_name, service_name, image_paths, 
            segmented_image_paths, use_segmented
        )

    async def _call_attribute_service(
        self, 
        request_id: uuid.UUID, 
        attribute_name: str, 
        service_name: str, 
        image_paths: list[str],
        segmented_image_paths: list[str] | None = None,
        use_segmented: bool = False
    ) -> tuple[Any, AttributeModelInfo]:
        """
        Call a specific service for a single attribute.
        
        Returns:
            Tuple of (AttributeResponse, AttributeModelInfo)
        """
        service_url = self.service_urls.get(service_name)
        if not service_url:
            raise ValueError(f"Service URL not found for service: {service_name}")

        # Get service type from config
        service_type = None
        for service in self.attribute_config["services"]:
            if service["name"] == service_name:
                service_type = service["type"]
                break

        endpoint_url = f"{service_url}/extract/{attribute_name}"
        
        request_data = AttributeRequest(
            request_id=request_id,
            attribute_name=attribute_name,
            image_paths=image_paths,
            segmented_image_paths=segmented_image_paths,
            use_segmented=use_segmented
        )

        self.logger.debug(
            f"Calling service for attribute '{attribute_name}'",
            request_id=str(request_id),
            service_name=service_name,
            endpoint_url=endpoint_url,
        )

        start_time = time.time()
        
        try:
            response = await self.client.post(
                endpoint_url,
                json=request_data.model_dump(mode='json'),
            )
            response.raise_for_status()
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            result = response.json()
            
            # Create AttributeModelInfo
            model_info = AttributeModelInfo(
                service_name=service_name,
                service_type=service_type or "unknown",
                processing_time_ms=processing_time_ms,
                confidence_score=result.get("confidence_score", 0.0),
                success=result.get("success", False),
                error_message=result.get("error_message"),
            )

            self.logger.debug(
                f"Service call successful for attribute '{attribute_name}'",
                request_id=str(request_id),
                service_name=service_name,
                processing_time_ms=processing_time_ms,
                confidence=result.get("confidence_score", 0.0),
            )

            # Parse response into AttributeResponse-like object
            from types import SimpleNamespace
            attr_response = SimpleNamespace(
                request_id=result["request_id"],
                attribute_name=result["attribute_name"],
                attribute_value=result["attribute_value"],
                confidence_score=result["confidence_score"],
                processing_time_ms=result["processing_time_ms"],
                success=result["success"],
                error_message=result.get("error_message")
            )

            return attr_response, model_info

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.error(
                f"Service call failed for attribute '{attribute_name}'",
                request_id=str(request_id),
                service_name=service_name,
                error=str(e),
                processing_time_ms=processing_time_ms,
            )
            
            raise e

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
            # Cleanup on failure (unless debug mode retains images)
            if not self.settings.debug_retain_images:
                cleanup_request_directory(str(request_id))
            raise

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
        """Check health of all configured services."""
        try:
            # Check database
            from shared.database import get_database_manager
            db_manager = get_database_manager()
            if not await db_manager.health_check():
                return False

            # Check configured services
            for service_name, service_url in self.service_urls.items():
                try:
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

    async def _test_service_connectivity(self) -> None:
        """Test connectivity to all configured services."""
        for service_name, service_url in self.service_urls.items():
            try:
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
    """Create the capability-based orchestrator FastAPI application."""
    service = CapabilityOrchestrator()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = CapabilityOrchestrator()
    service.run()
