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
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=self.settings.service_request_timeout)
        
        # Load simplified configuration
        self.config = self._load_simple_config()
        self.service_urls = self._build_service_urls()

    def _load_simple_config(self) -> Dict[str, Any]:
        """Load simplified routing configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "attribute_routing.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(
                "Simplified routing configuration loaded",
                attributes=list(config["attributes"].keys()),
                services=set(config["attributes"].values()),
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
                
                # 2. Extract attributes in parallel
                attributes, model_info = await self._extract_attributes_parallel(
                    request_id, image_paths
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

    async def _extract_attributes_parallel(
        self, request_id: uuid.UUID, image_paths: List[str]
    ) -> tuple[AllAttributes, Dict[str, AttributeModelInfo]]:
        """
        Extract all configured attributes in parallel.
        
        Simple logic:
        1. Group attributes by service
        2. Call each service once with its attributes
        3. Aggregate results
        """
        # Group attributes by service
        service_tasks = {}
        for attr, service in self.config["attributes"].items():
            if service not in service_tasks:
                service_tasks[service] = []
            service_tasks[service].append(attr)

        self.logger.info(
            "Starting parallel attribute extraction",
            request_id=str(request_id),
            service_tasks=service_tasks,
        )

        # Create tasks for each service
        tasks = [
            self._call_service(request_id, service, attrs, image_paths)
            for service, attrs in service_tasks.items()
        ]

        # Execute all service calls in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_attributes = {}
        model_info = {}

        for i, (service, attrs) in enumerate(service_tasks.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                self.logger.error(
                    f"Service {service} failed",
                    request_id=str(request_id),
                    error=str(result),
                )
                # Set all attributes for this service to None
                for attr in attrs:
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
        self, request_id: uuid.UUID, service_name: str, attributes: List[str], image_paths: List[str]
    ) -> tuple[Dict[str, Any], Dict[str, AttributeModelInfo]]:
        """
        Call a service to extract multiple attributes.
        
        Each service now receives all attributes it should extract in one call.
        """
        service_url = self.service_urls.get(service_name)
        if not service_url:
            raise ValueError(f"Service URL not found: {service_name}")

        endpoint_url = f"{service_url}/extract_batch"
        
        # New simplified request format
        request_data = {
            "request_id": str(request_id),
            "attributes": attributes,
            "image_paths": image_paths,
        }

        self.logger.debug(
            f"Calling service {service_name}",
            request_id=str(request_id),
            attributes=attributes,
            endpoint=endpoint_url,
        )

        start_time = time.time()
        
        try:
            response = await self.client.post(endpoint_url, json=request_data)
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