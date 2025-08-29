"""
Base service class for Smartrobe microservices.

Provides common functionality including:
- FastAPI application setup
- Health check endpoints
- Logging configuration
- Error handling
- Service lifecycle management
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_model_service_settings, get_settings
from .logging import get_logger, setup_logging
from .schemas import HealthCheck, HealthStatus


class BaseService(ABC):
    """Base class for all Smartrobe microservices."""

    def __init__(self, service_name: str, version: str = "1.0.0", settings=None):
        self.service_name = service_name
        self.version = version
        self.settings = settings if settings is not None else get_settings()

        # Setup logging
        setup_logging(service_name, self.settings)
        self.logger = get_logger(f"{service_name}.base")

        # Create FastAPI app
        self.app = self._create_app()

        # Service state
        self._startup_time = None
        self._is_healthy = True
        self._health_details = {}

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title=f"Smartrobe {self.service_name.title()} Service",
            description=f"Smartrobe {self.service_name} service",
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add exception handler
        app.add_exception_handler(Exception, self._global_exception_handler)

        # Add startup and shutdown events
        app.add_event_handler("startup", self._startup_handler)
        app.add_event_handler("shutdown", self._shutdown_handler)

        # Add health check endpoint
        app.add_api_route("/health", self.health_check, methods=["GET"])

        # Add service-specific routes
        self._add_routes(app)

        return app

    @abstractmethod
    def _add_routes(self, app: FastAPI) -> None:
        """Add service-specific routes to the FastAPI app."""
        pass

    async def _startup_handler(self) -> None:
        """Handle service startup."""
        self._startup_time = time.time()
        self.logger.info(f"{self.service_name} service starting up")

        try:
            await self._initialize_service()
            self._is_healthy = True
            self.logger.info(f"{self.service_name} service started successfully")
        except Exception as e:
            self._is_healthy = False
            self.logger.error(
                f"Failed to start {self.service_name} service", error=str(e)
            )
            raise

    async def _shutdown_handler(self) -> None:
        """Handle service shutdown."""
        self.logger.info(f"{self.service_name} service shutting down")

        try:
            await self._cleanup_service()
            self.logger.info(f"{self.service_name} service shutdown complete")
        except Exception as e:
            self.logger.error(
                f"Error during {self.service_name} service shutdown", error=str(e)
            )

    async def _global_exception_handler(self, request, exc: Exception) -> JSONResponse:
        """Global exception handler for unhandled errors."""
        self.logger.error(
            "Unhandled exception",
            path=str(request.url.path),
            method=request.method,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "service": self.service_name,
                "timestamp": time.time(),
            },
        )

    async def health_check(self) -> HealthCheck:
        """Health check endpoint."""
        try:
            # Perform service-specific health checks
            service_health = await self._check_service_health()

            # Determine overall status
            status = (
                HealthStatus.HEALTHY
                if (self._is_healthy and service_health)
                else HealthStatus.UNHEALTHY
            )

            # Collect health details
            details = {
                "startup_time": self._startup_time,
                "uptime_seconds": time.time() - self._startup_time
                if self._startup_time
                else 0,
                **self._health_details,
            }

            return HealthCheck(
                service=self.service_name,
                status=status,
                version=self.version,
                details=details,
            )

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return HealthCheck(
                service=self.service_name,
                status=HealthStatus.UNHEALTHY,
                version=self.version,
                details={"error": str(e)},
            )

    @abstractmethod
    async def _initialize_service(self) -> None:
        """Initialize service-specific components."""
        pass

    @abstractmethod
    async def _cleanup_service(self) -> None:
        """Cleanup service-specific components."""
        pass

    async def _check_service_health(self) -> bool:
        """
        Perform service-specific health checks.

        Returns:
            True if service is healthy, False otherwise
        """
        return True

    def set_health_detail(self, key: str, value: Any) -> None:
        """Set a health check detail."""
        self._health_details[key] = value

    def set_unhealthy(self, reason: str) -> None:
        """Mark service as unhealthy."""
        self._is_healthy = False
        self.set_health_detail("unhealthy_reason", reason)
        self.logger.warning(f"Service marked as unhealthy: {reason}")

    def set_healthy(self) -> None:
        """Mark service as healthy."""
        self._is_healthy = True
        if "unhealthy_reason" in self._health_details:
            del self._health_details["unhealthy_reason"]
        self.logger.info("Service marked as healthy")

    def run(self, host: str = None, port: int = None) -> None:
        """Run the service."""
        import uvicorn

        # Get configuration
        service_config = self.settings.get_service_config()

        run_host = host or service_config.host
        run_port = port or service_config.port

        self.logger.info(
            f"Starting {self.service_name} service",
            host=run_host,
            port=run_port,
            debug=self.settings.debug,
        )

        uvicorn.run(
            self.app,
            host=run_host,
            port=run_port,
            log_config=None,  # Use our custom logging
            access_log=False,  # Disable uvicorn access logs
            reload=self.settings.debug,
        )


class CapabilityService(BaseService):
    """Base class for capability-aware model services."""

    # Subclasses should define their supported attributes
    SUPPORTED_ATTRIBUTES: dict[str, type] = {}
    SERVICE_TYPE: str = ""

    def __init__(self, service_name: str, service_type: str, version: str = "1.0.0"):
        self.service_type = service_type
        super().__init__(service_name, version, settings=get_model_service_settings())

    @abstractmethod
    async def extract_single_attribute(
        self, request_id: str, attribute_name: str, image_paths: list[str]
    ) -> tuple[Any, float]:
        """
        Extract a single attribute from images.

        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            image_paths: List of paths to images

        Returns:
            Tuple of (attribute_value, confidence_score)
        """
        pass

    def get_supported_attributes(self) -> list[str]:
        """Get list of supported attributes."""
        return list(self.SUPPORTED_ATTRIBUTES.keys())

    def _add_routes(self, app: FastAPI) -> None:
        """Add capability-aware routes."""
        from .schemas import AttributeResponse

        # Add individual attribute endpoints
        for attr_name in self.SUPPORTED_ATTRIBUTES.keys():
            endpoint_path = f"/extract/{attr_name}"
            endpoint_func = self._create_attribute_endpoint(attr_name)
            app.add_api_route(
                endpoint_path,
                endpoint_func,
                methods=["POST"],
                response_model=AttributeResponse,
                name=f"extract_{attr_name}"
            )

        # Add capabilities endpoint
        @app.get("/capabilities")
        async def get_capabilities():
            """Get service capabilities."""
            return {
                "service_name": self.service_name,
                "service_type": self.service_type,
                "supported_attributes": self.get_supported_attributes(),
                "version": self.version,
                "endpoints": [
                    f"/extract/{attr}" for attr in self.get_supported_attributes()
                ]
            }

    def _create_attribute_endpoint(self, attribute_name: str):
        """Create an endpoint function for a specific attribute."""
        from .schemas import AttributeRequest, AttributeResponse

        async def extract_attribute_endpoint(
            request: AttributeRequest,
        ) -> AttributeResponse:
            """Extract a specific attribute from images."""
            start_time = time.time()

            try:
                self.logger.info(
                    f"Attribute '{attribute_name}' extraction request received",
                    request_id=str(request.request_id),
                    attribute=attribute_name,
                    image_count=len(request.image_paths),
                )

                # Determine which image paths to use
                image_paths_to_use = request.image_paths
                if request.use_segmented and request.segmented_image_paths:
                    image_paths_to_use = request.segmented_image_paths
                    self.logger.debug(
                        f"Using segmented images for {attribute_name}",
                        request_id=str(request.request_id),
                        segmented_count=len(request.segmented_image_paths),
                    )

                # Extract the specific attribute
                attribute_value, confidence_score = await self.extract_single_attribute(
                    str(request.request_id), attribute_name, image_paths_to_use
                )

                processing_time_ms = int((time.time() - start_time) * 1000)

                self.logger.info(
                    f"Attribute '{attribute_name}' extraction completed",
                    request_id=str(request.request_id),
                    attribute=attribute_name,
                    processing_time_ms=processing_time_ms,
                    confidence_score=confidence_score,
                )

                return AttributeResponse(
                    request_id=request.request_id,
                    attribute_name=attribute_name,
                    attribute_value=attribute_value,
                    confidence_score=confidence_score,
                    processing_time_ms=processing_time_ms,
                    success=True,
                )

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)

                self.logger.error(
                    f"Attribute '{attribute_name}' extraction failed",
                    request_id=str(request.request_id),
                    attribute=attribute_name,
                    error=str(e),
                    processing_time_ms=processing_time_ms,
                )

                return AttributeResponse(
                    request_id=request.request_id,
                    attribute_name=attribute_name,
                    attribute_value=None,
                    confidence_score=0.0,
                    processing_time_ms=processing_time_ms,
                    success=False,
                    error_message=str(e),
                )

        return extract_attribute_endpoint
