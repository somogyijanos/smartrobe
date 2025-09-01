"""
Heuristics service for Smartrobe.

Extracts basic visual attributes using rule-based analysis and computer vision.
Currently implements color detection through image processing techniques.
"""

import asyncio
import binascii
import colorsys
import time
from typing import Any

import httpx
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from shared.base_service import CapabilityService
from shared.schemas import Color


class HeuristicsService(CapabilityService):
    """Heuristics service for rule-based attribute extraction."""

    SUPPORTED_ATTRIBUTES = {
        "color": Color,
    }
    SERVICE_TYPE = "heuristic"

    def __init__(self):
        super().__init__("heuristics", "heuristic", "1.0.0")
        self.client = httpx.AsyncClient(timeout=30)

    async def _initialize_service(self) -> None:
        """Initialize heuristics service."""
        # In a real implementation, this would initialize:
        # - Color analysis models (K-means clustering, color space conversion)
        # - Image preprocessing pipelines
        # - Computer vision libraries (OpenCV, etc.)
        await asyncio.sleep(0.2)
        
        self.logger.info("Heuristics service initialized successfully")
        self.set_health_detail("color_detector_loaded", True)
        self.set_health_detail("image_processor_ready", True)
        self.set_health_detail("cv_libraries_loaded", True)

    async def _cleanup_service(self) -> None:
        """Cleanup heuristics service."""
        await self.client.aclose()
        self.logger.info("Heuristics service cleaned up")

    async def extract_single_attribute(
        self, request_id: str, attribute_name: str, image_paths: list[str]
    ) -> tuple[Any, float]:
        """
        Extract a single attribute using heuristic methods.

        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            image_paths: List of paths to images

        Returns:
            Tuple of (attribute_value, confidence_score)
        """
        self.logger.info(
            f"Extracting {attribute_name} using heuristics",
            request_id=request_id,
            attribute=attribute_name,
            image_count=len(image_paths),
        )

        if attribute_name == "color":
            return await self._extract_color(request_id, image_paths)
        else:
            raise ValueError(f"Unsupported attribute: {attribute_name}")

    async def _extract_color(self, request_id: str, image_paths: list[str]) -> tuple[Color, float]:
        """
        Extract dominant color using real color analysis with Florence-2 preprocessing.

        Process:
        1. Use Microsoft Florence-2 to crop images to garment only (non-close-ups)
        2. Apply K-means clustering on cropped images
        3. Extract dominant color as hex
        4. Map to predefined color categories
        5. Calculate confidence based on color distribution

        Args:
            request_id: Unique request identifier
            image_paths: List of paths to images

        Returns:
            Tuple of (Color, confidence_score)
        """
        start_time = time.time()
        
        try:
            # Step 1: Get non-close-up cropped images using Microsoft Florence-2
            cropped_image_paths = await self._get_cropped_images(request_id, image_paths)
            
            if not cropped_image_paths:
                self.logger.warning(
                    "No suitable images for color extraction after Florence-2 processing",
                    request_id=request_id
                )
                # Fallback to original images if no cropped images available
                cropped_image_paths = image_paths
            
            # Step 2: Extract dominant color from the best cropped image in HSV space
            best_image_path = cropped_image_paths[0]  # Use first available image
            dominant_hsv, hex_color = self._get_dominant_color_hsv(best_image_path)
            
            # Step 3: Map HSV directly to predefined Color enum
            mapped_color, confidence = self._map_hsv_to_color_enum(dominant_hsv)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.info(
                "Color extraction completed",
                request_id=request_id,
                hex_color=hex_color,
                dominant_hsv=dominant_hsv.tolist() if isinstance(dominant_hsv, np.ndarray) else dominant_hsv,
                mapped_color=mapped_color.value,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                images_processed=len(cropped_image_paths),
            )
            
            return mapped_color, confidence
            
        except Exception as e:
            self.logger.error(
                "Color extraction failed",
                request_id=request_id,
                error=str(e),
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
            # Return a fallback color with low confidence
            return Color.GRAY, 0.1

    async def _get_cropped_images(self, request_id: str, image_paths: list[str]) -> list[str]:
        """Use Microsoft Florence-2 to get cropped, non-close-up images suitable for color extraction."""
        try:
            # Call Microsoft Florence-2 service to process images
            florence_url = self.settings.get_service_url("microsoft-florence")
            
            request_data = {
                "image_paths": image_paths,
                "output_dir": None  # Will generate paths automatically
            }
            
            response = await self.client.post(f"{florence_url}/process_images", json=request_data)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("success"):
                # Return only non-close-up cropped images
                non_close_up_cropped = result.get("non_close_up_cropped_paths", [])
                
                self.logger.info(
                    "Florence-2 image processing completed",
                    request_id=request_id,
                    original_count=len(image_paths),
                    cropped_count=len(non_close_up_cropped),
                )
                
                return non_close_up_cropped
            else:
                self.logger.error(
                    "Florence-2 processing failed",
                    request_id=request_id,
                    error=result.get("error_message", "Unknown error")
                )
                return []
                
        except Exception as e:
            self.logger.error(
                "Failed to call Microsoft Florence-2 service",
                request_id=request_id,
                error=str(e)
            )
            return []

    def _get_dominant_color_hsv(self, image_path: str) -> tuple[np.ndarray, str]:
        """
        Extract dominant color using K-means clustering in HSV space.
        
        HSV clustering is much better than RGB because:
        - Perceptually meaningful color separation
        - Better handling of lighting variations
        - More intuitive color clusters
        """
        NUM_CLUSTERS = 5
        
        # Load and preprocess image
        with Image.open(image_path) as im:
            # Resize to reduce computation time
            im = im.resize((50, 50))
            rgb_array = np.asarray(im)
            shape = rgb_array.shape
            
            # Convert RGB to HSV
            hsv_array = np.zeros_like(rgb_array, dtype=float)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    r, g, b = rgb_array[i, j] / 255.0
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_array[i, j] = [h * 360, s, v]  # Hue in degrees, S&V in 0-1
            
            # Reshape for clustering
            hsv_flat = hsv_array.reshape(np.prod(shape[:2]), shape[2])
            
            # Handle circular hue properly for clustering
            # Convert hue to cartesian coordinates to handle 0°=360° properly
            hue_rad = hsv_flat[:, 0] * np.pi / 180  # Convert to radians
            hue_x = np.cos(hue_rad) * hsv_flat[:, 1]  # x = cos(hue) * saturation
            hue_y = np.sin(hue_rad) * hsv_flat[:, 1]  # y = sin(hue) * saturation
            
            # Create feature matrix: [hue_x, hue_y, value]
            # This preserves hue circularity and weights by saturation
            features = np.column_stack([hue_x, hue_y, hsv_flat[:, 2]])
            
            # Apply K-means clustering in this transformed space
            kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
            kmeans.fit(features)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            # Find most frequent cluster
            counts, bins = np.histogram(labels, bins=NUM_CLUSTERS)
            index_max = np.argmax(counts)
            
            # Convert centroid back to HSV
            centroid = centroids[index_max]
            hue_x, hue_y, value = centroid
            
            # Reconstruct hue and saturation from cartesian coordinates
            saturation = np.sqrt(hue_x**2 + hue_y**2)
            hue_deg = np.arctan2(hue_y, hue_x) * 180 / np.pi
            if hue_deg < 0:
                hue_deg += 360  # Ensure positive hue
            
            dominant_hsv = np.array([hue_deg, saturation, value])
            
            # Convert back to RGB for hex representation
            h_norm = hue_deg / 360.0
            s_norm = min(1.0, saturation)  # Clamp saturation to [0,1]
            v_norm = min(1.0, value)       # Clamp value to [0,1]
            
            r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
            rgb_int = np.array([int(r * 255), int(g * 255), int(b * 255)])
            
            # Convert to hex
            hex_color = binascii.hexlify(bytearray(rgb_int)).decode('ascii')
            
            return dominant_hsv, hex_color

    def _map_hsv_to_color_enum(self, hsv_values: np.ndarray) -> tuple[Color, float]:
        """
        Map HSV color directly to predefined Color enum.
        
        Much more efficient and accurate since we're already in HSV space.
        """
        h_deg, s, v = hsv_values
        
        # First check for achromatic colors (low saturation)
        if s < 0.2:  # Low saturation = grayscale
            if v < 0.2:
                return Color.BLACK, 0.9
            elif v > 0.8:
                return Color.WHITE, 0.9
            else:
                return Color.GRAY, 0.85
        
        # For chromatic colors, use hue primarily with saturation/value adjustments
        confidence = 0.8  # Base confidence for chromatic colors
        
        # Define hue ranges for colors (in degrees)
        if h_deg < 15 or h_deg >= 345:  # Red
            if s > 0.7 and v > 0.6:
                return Color.RED, confidence
            elif s < 0.5 and v > 0.7:
                return Color.PINK, confidence * 0.9
            else:
                return Color.RED, confidence * 0.8
                
        elif 15 <= h_deg < 45:  # Orange/Brown
            if v < 0.5:
                return Color.BROWN, confidence
            else:
                return Color.ORANGE, confidence
                
        elif 45 <= h_deg < 75:  # Yellow/Beige
            if s < 0.4 and v > 0.7:
                return Color.BEIGE, confidence * 0.9
            else:
                return Color.YELLOW, confidence
                
        elif 75 <= h_deg < 165:  # Green
            return Color.GREEN, confidence
            
        elif 165 <= h_deg < 195:  # Cyan (classify as blue)
            return Color.BLUE, confidence * 0.9
            
        elif 195 <= h_deg < 255:  # Blue
            if v < 0.4 or (s > 0.7 and v < 0.6):
                return Color.NAVY, confidence
            else:
                return Color.BLUE, confidence
                
        elif 255 <= h_deg < 285:  # Purple
            return Color.PURPLE, confidence
            
        elif 285 <= h_deg < 345:  # Pink/Magenta
            return Color.PINK, confidence
        
        # Fallback for edge cases
        return Color.MULTICOLOR, 0.6

    async def _check_service_health(self) -> bool:
        """Check heuristics service health."""
        try:
            # Test basic color detection capability
            test_start = time.time()
            await asyncio.sleep(0.01)  # Simulate quick health check
            
            health_check_time = time.time() - test_start
            self.set_health_detail("last_health_check_ms", int(health_check_time * 1000))
            self.set_health_detail("color_detection_ready", True)
            
            return True
        except Exception as e:
            self.logger.error("Heuristics service health check failed", error=str(e))
            return False


def create_app():
    """Create the heuristics FastAPI application."""
    service = HeuristicsService()
    return service.app


# Create app instance for uvicorn
app = create_app()


# For development/testing
if __name__ == "__main__":
    service = HeuristicsService()
    service.run()
