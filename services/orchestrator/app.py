"""
Simplified orchestrator service for Smartrobe.

Clean design: receives n images → determines attributes to extract → calls services in parallel → aggregates results.
"""

import asyncio
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

from shared.database import InferenceRepository, init_database
from shared.base_service import BaseService
from shared.config import create_request_directory, cleanup_request_directory, get_settings
from shared.schemas import (
    AllAttributes,
    AnalyzeRequest,
    AnalyzeResponse,
    AttributeModelInfo,
    ProcessingInfo,
)


class SimplifiedOrchestrator(BaseService):
    """Simplified orchestrator for clean attribute routing."""

    # All supported attributes (can be extended)
    ALL_ATTRIBUTES = [
        "category", "gender", "sleeve_length", "neckline", "closure_type", "fit",
        "color", "material", "pattern", "brand", "style", "season", "condition"
    ]

    def __init__(self):
        super().__init__("orchestrator", "2.0.0")  # Version bump for simplified design
        # Create client without timeout - we'll set per-request timeouts
        self.client = httpx.AsyncClient()
        
        # Load simplified configuration
        self.config = self._load_simple_config()
        self.service_urls = self._build_service_urls()

    def _load_simple_config(self) -> Dict[str, Any]:
        """Load simplified routing configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "attribute_routing.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract service names from config
            services = set()
            for attr_config in config["attributes"].values():
                services.add(attr_config.get("service"))
            
            self.logger.info(
                "Simplified routing configuration loaded",
                attributes=list(config["attributes"].keys()),
                services=list(services),
            )
            return config
            
        except Exception as e:
            self.logger.error("Failed to load configuration", error=str(e))
            # Minimal fallback config
            return {
                "attributes": {},
                "timeouts": {"default": 30}
            }

    def _build_service_urls(self) -> Dict[str, str]:
        """Build service URL mapping from environment variables."""
        return {
            "heuristics": self.settings.heuristics_service_url,
            "llm-multimodal": self.settings.llm_multimodal_service_url,
            "fashion-clip": self.settings.fashion_clip_service_url,
            "microsoft-florence": self.settings.microsoft_florence_service_url,
        }

    def _add_routes(self, app: FastAPI) -> None:
        """Add simplified orchestrator routes."""

        @app.post("/v1/items/analyze", response_model=AnalyzeResponse)
        async def analyze_item(request: AnalyzeRequest) -> AnalyzeResponse:
            """
            Simplified item analysis: download images → route to services → aggregate results.
            """
            start_time = time.time()
            request_id = uuid.uuid4()
            
            self.logger.info(
                "Analysis request received",
                request_id=str(request_id),
                image_count=len(request.images),
            )

            try:
                # 1. Download and validate images
                image_paths = await self._download_images(request_id, request.images)
                
                # 2. Classify images (close-up vs non-close-up) and get garment bboxes
                image_metadata = await self._classify_images(request_id, image_paths)
                
                # 3. Extract attributes in parallel with image filtering
                attributes, model_info = await self._extract_attributes_parallel(
                    request_id, image_paths, image_metadata
                )
                
                # 3. Create response
                total_time_ms = int((time.time() - start_time) * 1000)
                response = AnalyzeResponse(
                    id=request_id,
                    attributes=attributes,
                    model_info=model_info,
                    processing=ProcessingInfo(
                        request_id=request_id,
                        total_processing_time_ms=total_time_ms,
                        image_download_time_ms=0,  # Could track separately if needed
                        timestamp=datetime.utcnow(),
                        image_count=len(request.images),
                        implemented_attributes=[k for k, v in model_info.items() if v.success],
                        skipped_attributes=[attr for attr in self.ALL_ATTRIBUTES 
                                          if attr not in self.config["attributes"]],
                    ),
                )

                # Store in database
                try:
                    await InferenceRepository.create_inference_result(
                        response, {"images": [str(url) for url in request.images]}
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to store inference result: {type(e).__name__}: {str(e)}",
                        request_id=str(request_id),
                    )

                self.logger.info(
                    "Analysis completed",
                    request_id=str(request_id),
                    total_time_ms=total_time_ms,
                    successful_attributes=len([v for v in model_info.values() if v.success]),
                )

                return response

            except Exception as e:
                self.logger.error("Analysis failed", request_id=str(request_id), error=str(e))
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

            finally:
                # Cleanup
                if not self.settings.debug_retain_images:
                    cleanup_request_directory(str(request_id))

    async def _classify_images(self, request_id: uuid.UUID, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify images using Florence service to determine close-up vs non-close-up 
        and get garment bounding boxes.
        
        Returns list of metadata dicts for each image:
        [
            {
                "image_path": str,
                "is_close_up": bool,
                "garment_bbox": [x1, y1, x2, y2] | None,
                "success": bool
            }, ...
        ]
        """
        florence_url = self.service_urls.get("microsoft-florence")
        if not florence_url:
            self.logger.warning("Florence service not available for image classification")
            # Return default metadata
            return [
                {
                    "image_path": path,
                    "is_close_up": False,
                    "garment_bbox": None,
                    "success": False
                } for path in image_paths
            ]

        self.logger.info(
            "Classifying images with Florence service",
            request_id=str(request_id),
            image_count=len(image_paths),
        )

        # Call Florence service for each image
        tasks = []
        for image_path in image_paths:
            task = self._classify_single_image(florence_url, image_path)
            tasks.append(task)

        # Execute classification tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        image_metadata = []
        for i, (image_path, result) in enumerate(zip(image_paths, results)):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Image classification failed for {image_path}",
                    request_id=str(request_id),
                    error=str(result),
                )
                metadata = {
                    "image_path": image_path,
                    "is_close_up": False,
                    "garment_bbox": None,
                    "success": False
                }
            else:
                # Extract first garment bbox if available
                bboxes = result.get("bboxes", [])
                garment_bbox = bboxes[0] if bboxes else None
                
                metadata = {
                    "image_path": image_path,
                    "is_close_up": result.get("is_close_up", False),
                    "garment_bbox": garment_bbox,
                    "success": result.get("success", False)
                }
            image_metadata.append(metadata)

        close_up_count = sum(1 for m in image_metadata if m["is_close_up"])
        self.logger.info(
            "Image classification completed",
            request_id=str(request_id),
            close_up_images=close_up_count,
            non_close_up_images=len(image_metadata) - close_up_count,
        )

        # Store metadata to shared storage for debugging
        await self._store_image_metadata(request_id, image_metadata)

        return image_metadata

    async def _store_image_metadata(self, request_id: uuid.UUID, image_metadata: List[Dict[str, Any]]) -> None:
        """Store image metadata to shared storage for debugging purposes."""
        try:
            import json
            from pathlib import Path
            from shared.config import get_settings
            
            # Use the same shared storage path as images
            settings = get_settings()
            request_dir = Path(settings.shared_storage_path) / str(request_id)
            metadata_file = request_dir / "image_metadata.json"
            
            # Ensure directory exists
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Store metadata with pretty formatting for easy debugging
            with open(metadata_file, 'w') as f:
                json.dump(image_metadata, f, indent=2)
            
            self.logger.info(
                "Image metadata stored for debugging",
                request_id=str(request_id),
                metadata_file=str(metadata_file),
                image_count=len(image_metadata)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to store image metadata",
                request_id=str(request_id),
                error=str(e)
            )
            # Don't fail the request if metadata storage fails

    def _filter_images_for_attribute(self, image_paths: List[str], image_metadata: List[Dict[str, Any]], image_filter: str) -> List[str]:
        """
        Filter images based on attribute requirements.
        
        Args:
            image_paths: All available image paths
            image_metadata: Metadata for each image (including is_close_up flag)
            image_filter: Filter type - "all", "close_up", or "non_close_up"
            
        Returns:
            List of filtered image paths
        """
        if image_filter == "all":
            return image_paths
        
        filtered_paths = []
        for meta in image_metadata:
            image_path = meta["image_path"]
            is_close_up = meta["is_close_up"]
            
            if image_filter == "close_up" and is_close_up:
                filtered_paths.append(image_path)
            elif image_filter == "non_close_up" and not is_close_up:
                filtered_paths.append(image_path)
                
        return filtered_paths

    async def _classify_single_image(self, florence_url: str, image_path: str) -> Dict[str, Any]:
        """Classify a single image using Florence service."""
        endpoint_url = f"{florence_url}/detect_garment"
        request_data = {"image_path": image_path}
        
        try:
            response = await self.client.post(endpoint_url, json=request_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to classify image {image_path}: {str(e)}")
            return {
                "success": False,
                "is_close_up": False,
                "garment_bbox": None,
                "error_message": str(e)
            }

    async def _extract_attributes_parallel(
        self, request_id: uuid.UUID, image_paths: List[str], image_metadata: List[Dict[str, Any]]
    ) -> tuple[AllAttributes, Dict[str, AttributeModelInfo]]:
        """
        Extract all configured attributes in parallel.
        
        Simple logic:
        1. Group attributes by service
        2. Call each service once with its attributes
        3. Aggregate results
        """
        # Group attributes by service with image filtering
        service_tasks = {}
        for attr, attr_config in self.config["attributes"].items():
            # Extract service and filter from config
            service = attr_config["service"]
            image_filter = attr_config.get("image_filter", "all")
            
            # Filter images based on attribute requirements
            filtered_images = self._filter_images_for_attribute(image_paths, image_metadata, image_filter)
            
            if service not in service_tasks:
                service_tasks[service] = []
            service_tasks[service].append({
                "attribute": attr,
                "image_filter": image_filter,
                "filtered_images": filtered_images,
                "image_metadata": [meta for meta in image_metadata if meta["image_path"] in filtered_images]
            })

        self.logger.info(
            "Starting parallel attribute extraction",
            request_id=str(request_id),
            service_tasks=service_tasks,
        )

        # Create tasks for each service
        tasks = [
            self._call_service(request_id, service, attr_configs)
            for service, attr_configs in service_tasks.items()
        ]

        # Execute all service calls in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_attributes = {}
        model_info = {}

        for i, (service, attr_configs) in enumerate(service_tasks.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                self.logger.error(
                    f"Service {service} failed",
                    request_id=str(request_id),
                    error=str(result),
                )
                # Set all attributes for this service to None
                for attr_config in attr_configs:
                    attr = attr_config["attribute"]
                    all_attributes[attr] = None
                    model_info[attr] = AttributeModelInfo(
                        service_name=service,
                        service_type="unknown",
                        processing_time_ms=0,
                        confidence_score=0.0,
                        success=False,
                        error_message=str(result),
                    )
            else:
                service_results, service_model_info = result
                all_attributes.update(service_results)
                model_info.update(service_model_info)

        # Fill in unimplemented attributes as None
        for attr in self.ALL_ATTRIBUTES:
            if attr not in all_attributes:
                all_attributes[attr] = None

        return AllAttributes(**all_attributes), model_info

    async def _call_service(
        self, request_id: uuid.UUID, service_name: str, attr_configs: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, AttributeModelInfo]]:
        """
        Call a service to extract multiple attributes with filtered images and metadata.
        
        Each service receives attributes with their filtered images and bbox metadata.
        """
        service_url = self.service_urls.get(service_name)
        if not service_url:
            raise ValueError(f"Service URL not found: {service_name}")

        endpoint_url = f"{service_url}/extract_batch"
        
        # Enhanced request format with filtered images and metadata
        attributes = [config["attribute"] for config in attr_configs]
        
        # Collect all unique filtered images and their metadata
        all_filtered_images = set()
        image_metadata_map = {}
        
        for config in attr_configs:
            for img_path in config["filtered_images"]:
                all_filtered_images.add(img_path)
            
            # Build metadata map for these images
            for meta in config["image_metadata"]:
                image_metadata_map[meta["image_path"]] = meta
        
        # Convert to list for JSON serialization
        image_paths = list(all_filtered_images)
        image_metadata = [image_metadata_map[path] for path in image_paths]
        
        request_data = {
            "request_id": str(request_id),
            "attributes": attributes,
            "image_paths": image_paths,
            "image_metadata": image_metadata,  # Include bbox and close-up info
            "attribute_configs": attr_configs,  # Include attribute filtering info
        }

        # Get service-specific timeout
        service_timeout = self.config.get("timeouts", {}).get(
            service_name, 
            self.config.get("timeouts", {}).get("default", 30)
        )

        self.logger.debug(
            f"Calling service {service_name}",
            request_id=str(request_id),
            attributes=attributes,
            endpoint=endpoint_url,
            timeout_seconds=service_timeout,
            image_count=len(image_paths),
            has_metadata=len(image_metadata) > 0,
        )

        start_time = time.time()
        
        try:
            response = await self.client.post(
                endpoint_url, 
                json=request_data, 
                timeout=service_timeout
            )
            response.raise_for_status()
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            result = response.json()
            
            # Parse service response
            service_results = result.get("attributes", {})
            confidence_scores = result.get("confidence_scores", {})
            
            # Create model info for each attribute
            model_info = {}
            for attr in attributes:
                model_info[attr] = AttributeModelInfo(
                    service_name=service_name,
                    service_type=result.get("service_type", "unknown"),
                    processing_time_ms=processing_time_ms,
                    confidence_score=confidence_scores.get(attr, 0.0),
                    success=result.get("success", False),
                    error_message=result.get("error_message"),
                )

            self.logger.debug(
                f"Service {service_name} completed",
                request_id=str(request_id),
                processing_time_ms=processing_time_ms,
                extracted_attributes=list(service_results.keys()),
                processed_images=len(image_paths),
            )

            return service_results, model_info

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(
                f"Service {service_name} call failed",
                request_id=str(request_id),
                error=str(e),
                processing_time_ms=processing_time_ms,
            )
            raise

    async def _download_images(self, request_id: uuid.UUID, image_urls: List[str]) -> List[str]:
        """Download and validate images (simplified version)."""
        request_dir = create_request_directory(str(request_id))
        image_paths = []

        for i, url in enumerate(image_urls):
            # Download image
            response = await self.client.get(str(url))
            response.raise_for_status()

            # Save image
            image_path = f"{request_dir}/image_{i}.jpg"
            with open(image_path, "wb") as f:
                f.write(response.content)
            
            image_paths.append(image_path)

        return image_paths

    async def _initialize_service(self) -> None:
        """Initialize simplified orchestrator."""
        # Initialize database
        await init_database(self.settings.database_url)
        self.logger.info("Simplified orchestrator initialized")

    async def _cleanup_service(self) -> None:
        """Cleanup orchestrator."""
        await self.client.aclose()

    async def _check_service_health(self) -> bool:
        """Check health of configured services."""
        for service_name, service_url in self.service_urls.items():
            try:
                response = await self.client.get(f"{service_url}/health")
                if response.status_code != 200:
                    return False
            except Exception:
                return False
        return True


def create_app() -> FastAPI:
    """Create the simplified orchestrator FastAPI application."""
    service = SimplifiedOrchestrator()
    return service.app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    service = SimplifiedOrchestrator()
    service.run()