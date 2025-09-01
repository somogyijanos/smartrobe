"""
Microsoft Florence-2 service for Smartrobe.

Provides garment detection, bounding box extraction, and image cropping functionality
using Microsoft's Florence-2 vision model for open vocabulary object detection.
"""

import asyncio
import io
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from fastapi import HTTPException

from shared.base_service import CapabilityService
from shared.schemas import ServiceRequest, ServiceResponse


class MicrosoftFlorenceService(CapabilityService):
    """Microsoft Florence-2 service for garment detection and cropping."""

    # Define supported attributes
    SUPPORTED_ATTRIBUTES = {"brand": str}
    SERVICE_TYPE = "vision_ocr"

    def __init__(self):
        super().__init__("microsoft-florence", "vision_ocr", "1.0.0")
        self.model = None
        self.processor = None
        self.model_id = 'microsoft/Florence-2-base'
        # Lock to prevent concurrent access to the model during inference (created in async init)
        self._model_lock = None

    async def _initialize_service(self) -> None:
        """Initialize the Florence-2 model."""
        self.logger.info("Loading Microsoft Florence-2 model...")
        
        try:
            # Create the asyncio lock for model concurrency protection
            self._model_lock = asyncio.Lock()
            
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
        self._model_lock = None
        self.logger.info("Microsoft Florence-2 service cleaned up")

    def _add_routes(self, app) -> None:
        """Add Florence-2 specific routes."""
        
        # First add the capability routes (extract_batch, etc.)
        super()._add_routes(app)
        
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

    async def extract_single_attribute(
        self, request_id: str, attribute_name: str, image_paths: list[str],
        image_metadata: list[dict], attribute_configs: list[dict]
    ) -> tuple[Any, float]:
        """
        Extract a single attribute from images with metadata.
        
        Args:
            request_id: Unique request identifier
            attribute_name: Name of the attribute to extract
            image_paths: List of paths to images
            image_metadata: Metadata including bboxes and close-up classification
            attribute_configs: Attribute-specific configurations
            
        Returns:
            Tuple of (attribute_value, confidence_score)
        """
        if attribute_name == "brand":
            return await self._extract_brand(image_paths, image_metadata)
        else:
            raise ValueError(f"Unsupported attribute: {attribute_name}")

    async def _extract_brand(self, image_paths: List[str], image_metadata: list[dict]) -> Tuple[Optional[str], float]:
        """
        Extract brand from multiple images using OCR approach from brand.ipynb.
        
        With metadata support:
        - Images are already filtered as close-ups by orchestrator
        - Bounding box information is available for additional cropping if needed
        
        This follows the EXACT approach from brand.ipynb on close-up images:
        1. Use pre-filtered close-up images from orchestrator
        2. Run OCR only on close-up images (brands are only visible in close-ups)
        3. Order OCR results by text size (largest first)  
        4. Find overlapping brands across close-up images
        5. Return the best candidate
        
        Returns:
            Tuple of (brand_name, confidence_score)
        """
        try:
            self.logger.info(
                f"Starting brand extraction from {len(image_paths)} images - filtering for close-ups first",
                image_count=len(image_paths)
            )
            
            # Step 1: Use pre-filtered close-up images from orchestrator
            close_up_images = []
            close_up_paths = []
            
            self.logger.info(f"Using pre-filtered close-up images from orchestrator metadata")
            for image_path in image_paths:
                try:
                    image = Image.open(image_path)
                    close_up_images.append(image)
                    close_up_paths.append(image_path)
                    self.logger.debug(f"Added pre-filtered close-up image: {image_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load pre-filtered image {image_path}: {str(e)}")
            
            if not close_up_images:
                self.logger.warning("No close-up images found - brand extraction requires close-up garment images")
                return None, 0.0
            
            self.logger.info(
                f"Found {len(close_up_images)} close-up images out of {len(image_paths)} total",
                close_up_count=len(close_up_images),
                close_up_paths=close_up_paths
            )
            
            # Step 2: Run OCR only on close-up images (like brand.ipynb)
            all_ocr_labels = []
            
            for i, (image, image_path) in enumerate(zip(close_up_images, close_up_paths)):
                try:
                    start_time = time.time()
                    
                    # Run OCR on close-up image
                    ocr_result = await self._run_ocr_with_region(image)
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    if ocr_result and 'quad_boxes' in ocr_result and 'labels' in ocr_result:
                        # Order labels by bbox size (largest text first) - like brand.ipynb
                        labels_ordered = self._select_greatest_bbox(
                            ocr_result['quad_boxes'], 
                            ocr_result['labels']
                        )
                        all_ocr_labels.append(labels_ordered)
                        
                        # Clean the top labels for debugging
                        top_labels_cleaned = [self._clean_ocr_label(label) for label in labels_ordered[:5]]
                        
                        self.logger.debug(
                            f"OCR completed for close-up image {i+1}/{len(close_up_images)}",
                            image_path=image_path,
                            processing_time_ms=processing_time,
                            labels_found=len(labels_ordered),
                            raw_top_labels=labels_ordered[:3],  # Show raw OCR results
                            cleaned_top_labels=top_labels_cleaned[:3]  # Show cleaned results
                        )
                    else:
                        self.logger.warning(f"No OCR data for close-up image {i+1}: {image_path}")
                        
                except Exception as e:
                    self.logger.error(f"OCR failed for close-up image {i+1} ({image_path}): {str(e)}")
                    continue
            
            if not all_ocr_labels:
                self.logger.warning("No OCR labels extracted from any image")
                return None, 0.0
            
            self.logger.info(
                f"OCR completed on {len(all_ocr_labels)} close-up images",
                successful_ocr_images=len(all_ocr_labels),
                total_close_ups=len(close_up_images),
                total_images=len(image_paths)
            )
            
            # Debug: Show all OCR results before overlap detection
            self.logger.info("OCR results summary for debugging:")
            for i, labels in enumerate(all_ocr_labels):
                cleaned_labels = [self._clean_ocr_label(label) for label in labels[:5]]
                self.logger.info(
                    f"Image {i+1} top labels",
                    raw_labels=labels[:3],
                    cleaned_labels=cleaned_labels[:3]
                )
            
            # Step 3: Find overlapping brand labels across close-up images (like brand.ipynb)
            brand_candidates = self._select_overlapping_labels(all_ocr_labels)
            
            self.logger.info(
                f"Overlap detection results",
                candidates_found=len(brand_candidates),
                candidates=brand_candidates[:3] if brand_candidates else []
            )
            
            if brand_candidates:
                # Return the best candidate with high confidence
                best_brand = brand_candidates[0]
                
                # Calculate confidence based on detection consistency
                matching_images = 0
                for labels in all_ocr_labels:
                    # Check if brand appears in top 3 labels of this image
                    top_labels = [self._clean_ocr_label(label).lower() for label in labels[:3]]
                    if any(best_brand.lower() in label for label in top_labels if label):
                        matching_images += 1
                
                confidence = min(0.95, matching_images / len(all_ocr_labels))
                
                self.logger.info(
                    f"Brand extraction successful via overlap detection",
                    brand=best_brand,
                    confidence=confidence,
                    candidate_count=len(brand_candidates),
                    matching_images=matching_images,
                    total_images=len(all_ocr_labels)
                )
                return best_brand, confidence
                
            else:
                # Fallback: use data from the image with the least text found in it
                self.logger.info("No overlapping brands found, trying fallback approach using image with least text")
                
                if all_ocr_labels:
                    # Find the image with the least amount of text
                    min_text_count = float('inf')
                    min_text_index = 0
                    
                    for i, labels in enumerate(all_ocr_labels):
                        text_count = len(labels)
                        self.logger.debug(
                            f"Image {i+1} text count: {text_count}",
                            image_path=close_up_paths[i] if i < len(close_up_paths) else f"unknown_{i}",
                            labels=labels[:3]
                        )
                        
                        if text_count < min_text_count:
                            min_text_count = text_count
                            min_text_index = i
                    
                    self.logger.info(
                        f"Selected image {min_text_index + 1} with least text count: {min_text_count}",
                        selected_image_path=close_up_paths[min_text_index] if min_text_index < len(close_up_paths) else f"unknown_{min_text_index}"
                    )
                    
                    # Get the largest text from the image with least text as fallback
                    if all_ocr_labels[min_text_index]:
                        fallback_brand = self._clean_ocr_label(all_ocr_labels[min_text_index][0])
                        if fallback_brand and len(fallback_brand) > 2:
                            self.logger.info(
                                f"Brand extraction fallback successful",
                                brand=fallback_brand,
                                confidence=0.5,
                                source=f"largest_text_from_least_text_image_{min_text_index + 1}",
                                text_count=min_text_count
                            )
                            return fallback_brand, 0.5
                
                self.logger.warning("No brand could be extracted from images")
                return None, 0.0
                
        except Exception as e:
            self.logger.error(f"Brand extraction failed: {str(e)}")
            return None, 0.0

    async def _run_ocr_with_region(self, image: Image.Image) -> Dict[str, Any]:
        """Run Florence-2 OCR with region detection on an image."""
        task_prompt = '<OCR_WITH_REGION>'
        
        if self.processor is None or self.model is None or self._model_lock is None:
            raise RuntimeError("Model not initialized")
        
        # Use lock to prevent concurrent model access
        async with self._model_lock:
            # Process inputs
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
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
        
        return parsed_answer.get('<OCR_WITH_REGION>', {})

    def _select_greatest_bbox(self, bboxes: List[List[float]], labels: List[str]) -> List[str]:
        """Select labels ordered by bbox height (largest first)."""
        bbox_data = []
        
        for idx, bbox in enumerate(bboxes):
            # bbox is a polygon with 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
            # Convert to min/max coordinates to calculate height
            y_coords = [bbox[i] for i in range(1, len(bbox), 2)]
            
            min_y, max_y = min(y_coords), max(y_coords)
            height = max_y - min_y
            
            bbox_data.append({
                'label': labels[idx],
                'height': height
            })
        
        # Sort by height in descending order
        bbox_data_sorted = sorted(bbox_data, key=lambda x: x['height'], reverse=True)
        
        # Return simple list of labels ordered by height
        return [item['label'] for item in bbox_data_sorted]

    def _select_overlapping_labels(self, all_labels: List[List[str]]) -> List[str]:
        """
        Find the most prioritized and similar labels across multiple lists.
        
        Args:
            all_labels: List of label lists, where each inner list is ordered by priority
        
        Returns:
            List of matching labels ordered by their combined priority and similarity score
        """
        if len(all_labels) < 2:
            return []
        
        def calculate_similarity(text1: str, text2: str) -> float:
            """Calculate similarity between two text strings"""
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            # Basic sequence similarity
            seq_similarity = SequenceMatcher(None, text1_lower, text2_lower).ratio()
            
            # Substring bonus: if one is contained in the other
            if text1_lower in text2_lower or text2_lower in text1_lower:
                shorter_len = min(len(text1_lower), len(text2_lower))
                longer_len = max(len(text1_lower), len(text2_lower))
                substring_bonus = shorter_len / longer_len * 0.3
                seq_similarity = min(1.0, seq_similarity + substring_bonus)
            
            return seq_similarity
        
        def calculate_priority_score(positions: List[int], total_lists: int) -> float:
            """Calculate priority score based on positions in each list"""
            # Lower positions (earlier in list) get higher scores
            # Missing from a list gets penalty
            if len(positions) < total_lists:
                coverage_penalty = len(positions) / total_lists
            else:
                coverage_penalty = 1.0
            
            # Average inverse position (1/position) for priority
            avg_priority = sum(1.0 / (pos + 1) for pos in positions) / len(positions)
            
            return avg_priority * coverage_penalty
        
        # Create candidate matches by comparing labels across all lists
        candidates = {}
        similarity_threshold = 0.6
        
        for i, list1 in enumerate(all_labels):
            for pos1, label1 in enumerate(list1):
                clean1 = self._clean_ocr_label(label1)
                if not clean1:
                    continue
                    
                # Compare with labels from other lists
                for j, list2 in enumerate(all_labels):
                    if i >= j:  # Only compare each pair once
                        continue
                        
                    for pos2, label2 in enumerate(list2):
                        clean2 = self._clean_ocr_label(label2)
                        if not clean2:
                            continue
                        
                        similarity = calculate_similarity(clean1, clean2)
                        
                        if similarity >= similarity_threshold:
                            # Use the shorter/cleaner label as the canonical form
                            if len(clean1) <= len(clean2):
                                canonical_label = clean1
                                original_label = label1
                            else:
                                canonical_label = clean2
                                original_label = label2
                            
                            canonical_key = canonical_label.lower()
                            
                            if canonical_key not in candidates:
                                candidates[canonical_key] = {
                                    'label': canonical_label,
                                    'original': original_label,
                                    'positions': {},
                                    'similarities': [],
                                    'list_count': 0
                                }
                            
                            # Track positions in each list
                            candidates[canonical_key]['positions'][i] = pos1
                            candidates[canonical_key]['positions'][j] = pos2
                            candidates[canonical_key]['similarities'].append(similarity)
                            
                            # Update list count
                            candidates[canonical_key]['list_count'] = len(candidates[canonical_key]['positions'])
        
        # Score and rank candidates
        scored_candidates = []
        
        for key, candidate in candidates.items():
            positions = list(candidate['positions'].values())
            avg_similarity = sum(candidate['similarities']) / len(candidate['similarities'])
            priority_score = calculate_priority_score(positions, len(all_labels))
            
            # Combined score: priority * similarity * coverage
            coverage_boost = candidate['list_count'] / len(all_labels)
            final_score = priority_score * avg_similarity * (1 + coverage_boost)
            
            scored_candidates.append({
                'label': candidate['label'],
                'original': candidate['original'],
                'score': final_score,
                'list_count': candidate['list_count'],
                'avg_similarity': avg_similarity,
                'positions': positions
            })
        
        # Sort by score (highest first) and return labels
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return [candidate['label'] for candidate in scored_candidates]

    def _clean_ocr_label(self, label: str) -> str:
        """Clean OCR artifacts and normalize text"""
        if not label:
            return ""
        
        # Remove OCR tokens like </s> and other angle bracket tokens
        clean = re.sub(r'<[^>]*/?>', '', label)
        # Remove extra whitespace and strip
        clean = re.sub(r'\s+', ' ', clean.strip())
        return clean

    async def _run_garment_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Run Florence-2 garment detection on an image."""
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        text_input = "garment"
        
        if self.processor is None or self.model is None or self._model_lock is None:
            raise RuntimeError("Model not initialized")
        
        # Prepare prompt
        prompt = task_prompt + text_input
        
        # Use lock to prevent concurrent model access
        async with self._model_lock:
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
        """
        Determine if the image is a close-up based on garment bounding box proximity to edges.
        
        Logic: If the bbox comes near to at least 3 out of 4 image borders, it's a close-up.
        "Near" means relative position <5% or >95% depending on which border.
        """
        bboxes_labels = result.get('bboxes_labels', [])
        bboxes = result.get('bboxes', [])
        
        if 'garment' in bboxes_labels and bboxes:
            img_width, img_height = image.size
            bbox = bboxes[0]  # Use first garment bbox
            x1, y1, x2, y2 = bbox
            
            # Calculate relative positions (0.0 to 1.0)
            left_rel = x1 / img_width
            right_rel = x2 / img_width  
            top_rel = y1 / img_height
            bottom_rel = y2 / img_height
            
            # Check how many borders the garment touches (within 5% threshold)
            borders_touched = 0
            
            if left_rel < 0.05:     # Touches left border
                borders_touched += 1
            if right_rel > 0.95:    # Touches right border  
                borders_touched += 1
            if top_rel < 0.05:      # Touches top border
                borders_touched += 1
            if bottom_rel > 0.95:   # Touches bottom border
                borders_touched += 1
            
            # Close-up if garment touches at least 3 borders
            is_close_up = borders_touched >= 3
            
            self.logger.debug(
                "Close-up detection analysis",
                bbox=[x1, y1, x2, y2],
                image_size=[img_width, img_height],
                relative_positions={
                    "left": f"{left_rel:.3f}",
                    "right": f"{right_rel:.3f}", 
                    "top": f"{top_rel:.3f}",
                    "bottom": f"{bottom_rel:.3f}"
                },
                borders_touched=borders_touched,
                is_close_up=is_close_up
            )
            
            return is_close_up
        else:
            return False

    async def _check_service_health(self) -> bool:
        """Check Microsoft Florence-2 service health."""
        try:
            if self.model is None or self.processor is None or self._model_lock is None:
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
