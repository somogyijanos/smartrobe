# AI Usage in Smartrobe Development

Transparency document detailing AI tool usage in this project's development.

## Overview

This document outlines how AI tools were used during the development of the Smartrobe multi-model attribute extraction service, as required for transparency and reproducibility.

## AI Tools Used

### 1. Claude Sonnet 4 Thinking (Anthropic) via Cursor

**Primary AI Assistant for Development**

**Usage Scope:**
- ✅ Code architecture planning and design decisions
- ✅ Implementation of microservices structure  
- ✅ Docker containerization setup
- ✅ Database schema design and SQLAlchemy models
- ✅ API endpoint implementation (FastAPI)
- ✅ Error handling and logging patterns
- ✅ Documentation writing (README, ARCHITECTURE, this file)
- ✅ Code review and optimization suggestions


### 2. AI Models Within the Application

**OpenAI GPT-4o**

**Integration:**
```python
# services/llm-multimodal/app.py
# GPT-4o is used for condition assessment of clothing items
async def extract_condition(self, image_paths: List[str]) -> str:
    # Sends clothing images to GPT-4o for subjective assessment
    # Returns: summary of the condition
```

**Usage Context:**
- Part of the actual product functionality
- Handles subjective attributes requiring human-like reasoning
- Processes multiple clothing images for condition assessment

**Other AI Models (Planned/Framework):**
- **Fashion-CLIP:** Vision transformer for category classification
- **Microsoft Florence-2:** Object detection and OCR for brand extraction
- **Custom Heuristics:** K-means clustering for color detection (traditional ML)