#!/usr/bin/env python3
"""
Script to analyze garment condition using OpenAI's vision model.
Looks for damages, wear, stains, and overall condition of clothing items.
"""

import base64
import json
import os
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field


class OverallCondition(str, Enum):
    """Overall condition ratings for garments."""
    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


class SeverityLevel(str, Enum):
    """Severity levels for damages and issues."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class DamageType(str, Enum):
    """Types of damage that can be found on garments."""
    HOLE = "hole"
    TEAR = "tear"
    RIP = "rip"
    STAIN = "stain"
    DISCOLORATION = "discoloration"
    FADING = "fading"
    YELLOWING = "yellowing"
    PILLING = "pilling"
    THINNING = "thinning"
    STRETCHING = "stretching"
    WORN_EDGES = "worn_edges"
    LOOSE_SEAM = "loose_seam"
    BROKEN_SEAM = "broken_seam"
    MISSING_BUTTON = "missing_button"
    BROKEN_ZIPPER = "broken_zipper"
    DEFORMATION = "deformation"
    SHRINKAGE = "shrinkage"
    OTHER = "other"


class HardwareCondition(str, Enum):
    """Condition of hardware elements like buttons, zippers, etc."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    BROKEN = "Broken"


class DamageItem(BaseModel):
    """Represents a specific damage or issue found on the garment."""
    type: DamageType = Field(..., description="Type of damage")
    location: str = Field(..., description="Location on the garment")
    severity: SeverityLevel = Field(..., description="Severity level")
    description: str = Field(..., description="Detailed description")


class GarmentConditionAnalysis(BaseModel):
    """Structured analysis of garment condition."""
    overall_condition: OverallCondition = Field(
        ..., description="Overall condition rating"
    )
    overall_score: int = Field(
        ..., ge=1, le=10, description="Numeric score from 1 (poor) to 10 (excellent)"
    )
    damages: list[DamageItem] = Field(
        default_factory=list, description="List of specific damages found"
    )
    stains_and_discoloration: list[DamageItem] = Field(
        default_factory=list, description="List of stains and color issues"
    )
    wear_signs: list[DamageItem] = Field(
        default_factory=list, description="List of wear indicators"
    )
    structural_issues: list[DamageItem] = Field(
        default_factory=list, description="List of structural problems"
    )
    hardware_condition: HardwareCondition = Field(
        ..., description="Condition of buttons, zippers, etc."
    )
    resale_suitable: bool = Field(
        ..., description="Whether item is suitable for resale"
    )
    repair_recommendations: list[str] = Field(
        default_factory=list, description="Suggested repairs if applicable"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in assessment (0.0-1.0)"
    )
    summary: str = Field(
        ..., description="Brief summary of the analysis"
    )


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_garment_condition(
    image_paths: list[str], api_key: str = None
) -> GarmentConditionAnalysis:
    """
    Analyze garment condition from multiple images using OpenAI's vision model.

    Args:
        image_paths: List of paths to garment images
        api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)

    Returns:
        Structured analysis result as GarmentConditionAnalysis
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

    # Prepare image content for the API
    image_content = []
    for image_path in image_paths:
        if not Path(image_path).exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Get file extension to determine image type
        file_ext = Path(image_path).suffix.lower()
        if file_ext in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif file_ext == '.png':
            media_type = "image/png"
        elif file_ext == '.webp':
            media_type = "image/webp"
        else:
            print(f"Warning: Unsupported image format: {file_ext}")
            continue

        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)

        image_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{base64_image}",
                "detail": "high"
            }
        })

    if not image_content:
        raise ValueError("No valid images found to analyze.")

    # Get JSON schema for structured output
    schema = GarmentConditionAnalysis.model_json_schema()

    # Prepare the prompt
    prompt = f"""
    Please analyze the condition of this garment item from the provided images.

    Examine the garment carefully for:
    1. Overall condition: Rate using ONLY these values:
       {[e.value for e in OverallCondition]}
    2. Visible damages: categorize each damage using these types:
       {[e.value for e in DamageType]}
    3. Hardware condition: Rate using ONLY these values:
       {[e.value for e in HardwareCondition]}

    For each issue found, categorize it into the appropriate array and include:
    - Type: Use ONLY the predefined damage types from the schema
    - Location: Specific location on the garment
      (e.g., "left sleeve", "front hem", "collar")
    - Severity: Use ONLY these levels: {[e.value for e in SeverityLevel]}
    - Description: Detailed description of the issue

    Provide your response as valid JSON following this exact schema:
    {json.dumps(schema, indent=2)}

    Be thorough, accurate, and assess the confidence in your analysis.
    """

    # Create the message content
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ] + image_content
        }
    ]

    try:
        # Make the API call with JSON mode
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,  # Increased for structured output
            temperature=0.1,  # Lower temperature for consistent analysis
            response_format={"type": "json_object"}  # Force JSON output
        )

        # Parse the JSON response
        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)

        # Validate and create structured response
        return GarmentConditionAnalysis.model_validate(response_data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from OpenAI: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Error analyzing images: {str(e)}") from e


def main():
    """Main function to run the garment analysis."""

    # Set your image paths here
    image_paths = [
        # Example paths - replace with your actual image paths
        "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/00/2025_08_2009_01_320011.JPG",
        "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/00/2025_08_2009_01_490012.JPG",
        "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/00/2025_08_2009_02_020040.JPG",
        "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/00/2025_08_2009_02_150041.JPG"
    ]
    # image_paths = [
    #     "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/"
    #     "public-vinted-damaged/image-1.jpg",
    #     "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/"
    #     "public-vinted-damaged/image-2.jpeg",
    #     "/Users/somogyijanos/Repos/somogyijanos/smartrobe/test-images/"
    #     "public-vinted-damaged/image-3.jpeg"
    # ]

    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OPENAI_API_KEY environment variable")
        print("You can set it by running: export OPENAI_API_KEY='your-api-key-here'")
        return

    print("Analyzing garment condition...")
    print(f"Processing {len(image_paths)} images...")

    try:
        # Analyze the garment condition
        result = analyze_garment_condition(image_paths)

        print("\n" + "=" * 50)
        print("GARMENT CONDITION ANALYSIS")
        print("=" * 50)

        # Display structured results
        print(f"Overall Condition: {result.overall_condition.value}")
        print(f"Overall Score: {result.overall_score}/10")
        print(f"Hardware Condition: {result.hardware_condition.value}")
        print(f"Resale Suitable: {'Yes' if result.resale_suitable else 'No'}")
        print(f"Confidence Score: {result.confidence_score:.2f}")

        if result.damages:
            print(f"\nDAMAGES FOUND ({len(result.damages)}):")
            for i, damage in enumerate(result.damages, 1):
                print(f"  {i}. {damage.type.value} at {damage.location}")
                print(f"     Severity: {damage.severity.value}")
                print(f"     Description: {damage.description}")

        if result.stains_and_discoloration:
            print(f"\nSTAINS & DISCOLORATION ({len(result.stains_and_discoloration)}):")
            for i, stain in enumerate(result.stains_and_discoloration, 1):
                print(f"  {i}. {stain.type.value} at {stain.location}")
                print(f"     Severity: {stain.severity.value}")
                print(f"     Description: {stain.description}")

        if result.wear_signs:
            print(f"\nWEAR SIGNS ({len(result.wear_signs)}):")
            for i, wear in enumerate(result.wear_signs, 1):
                print(f"  {i}. {wear.type.value} at {wear.location}")
                print(f"     Severity: {wear.severity.value}")
                print(f"     Description: {wear.description}")

        if result.structural_issues:
            print(f"\nSTRUCTURAL ISSUES ({len(result.structural_issues)}):")
            for i, issue in enumerate(result.structural_issues, 1):
                print(f"  {i}. {issue.type.value} at {issue.location}")
                print(f"     Severity: {issue.severity.value}")
                print(f"     Description: {issue.description}")

        if result.repair_recommendations:
            print("\nREPAIR RECOMMENDATIONS:")
            for i, rec in enumerate(result.repair_recommendations, 1):
                print(f"  {i}. {rec}")

        print("\nSUMMARY:")
        print(result.summary)

        print("\n" + "=" * 50)
        print("JSON OUTPUT:")
        print("=" * 50)
        print(result.model_dump_json(indent=2))
        print("=" * 50)

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check your API key and image paths.")


if __name__ == "__main__":
    main()
