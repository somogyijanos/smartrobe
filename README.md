# 🧥 Smartrobe - Multi-Model Attribute Extraction Service

A high-performance microservices system that analyzes clothing images using multiple AI models to extract comprehensive product attributes for second-hand clothing marketplaces.

## 🎯 Overview

Smartrobe processes 1-4 photos of clothing items and extracts 13 different attributes using three specialized model types:

- **Vision Classifier**: Analyzes visual patterns for category, gender, sleeve length, neckline, closure type, and fit
- **Heuristic Model**: Uses rule-based algorithms for color, material, pattern, and brand detection  
- **LLM Model**: Leverages language model reasoning for style, season, and condition assessment

### Key Features

- 🚀 **REST API**: Single endpoint `/v1/items/analyze` for complete item analysis
- 🔄 **Parallel Processing**: All models process images simultaneously for optimal performance
- 💾 **Database Persistence**: PostgreSQL storage of all inference results
- 🐳 **Containerized**: Docker-based microservices architecture
- 📊 **Comprehensive Monitoring**: Health checks and structured JSON logging
- ⚡ **High Performance**: Target response time of 2-8 seconds

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   Orchestrator   │───▶│   PostgreSQL    │
│                 │    │    (Port 8000)   │    │    Database     │
└─────────────────┘    └──────────┬───────┘    └─────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
            │ Vision Class. │ │ Heuristic     │ │ LLM Model     │
            │ (Port 8001)   │ │ (Port 8002)   │ │ (Port 8003)   │
            └───────────────┘ └───────────────┘ └───────────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                        ┌─────────────────┐
                        │ Shared Storage  │
                        │    Volume       │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM (for optimal performance)
- Ports 8000-8003 and 5432 available

### 1. Clone and Setup

```bash
git clone <repository-url>
cd smartrobe
cp .env.example .env
# Edit .env if needed for your environment
```

### 2. Start All Services

```bash
# Use the convenient startup script
./scripts/start-dev.sh

# Or manually with docker compose
docker compose up --build
```

### 3. Test the API

```bash
curl -X POST "http://localhost:8000/v1/items/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "https://picsum.photos/600/400?random=1",
      "https://picsum.photos/600/400?random=2", 
      "https://picsum.photos/600/400?random=3",
      "https://picsum.photos/600/400?random=4"
    ]
  }'

# Example with fewer images (reduced accuracy)
curl -X POST "http://localhost:8000/v1/items/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "https://picsum.photos/600/400?random=1",
      "https://picsum.photos/600/400?random=2"
    ]
  }'
```

### 4. Explore the API

- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Service Status**: Check individual services at ports 8001-8003

## 📋 API Reference

### Analyze Item Endpoint

**POST** `/v1/items/analyze`

Analyzes 1-4 clothing images and returns extracted attributes.

#### Request Body

```json
{
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg", 
    "https://example.com/image3.jpg",
    "https://example.com/image4.jpg"
  ]
}
```

**Requirements:**
- 1-4 HTTPS image URLs (4 recommended for optimal accuracy)
- Images ≤ 10MB each
- Supported formats: JPEG, PNG, WebP

#### Response

```json
{
  "id": "uuid-v4",
  "attributes": {
    // Vision attributes (6)
    "category": "shirt",
    "gender": "unisex", 
    "sleeve_length": "long",
    "neckline": "crew",
    "closure_type": "button",
    "fit": "regular",
    
    // Heuristic attributes (4)
    "color": "blue",
    "material": "cotton",
    "pattern": "solid", 
    "brand": "Nike",
    
    // LLM attributes (3)
    "style": "casual",
    "season": "all_season",
    "condition": "good"
  },
  "model_info": {
    "vision_classifier": {
      "model_type": "vision_classifier",
      "version": "1.0.0",
      "processing_time_ms": 450,
      "confidence_scores": { "category": 0.95, ... },
      "success": true
    },
    // ... other models
  },
  "processing_info": {
    "request_id": "uuid-v4",
    "total_processing_time_ms": 1250,
    "image_download_time_ms": 300,
    "parallel_processing": true,
    "timestamp": "2024-01-01T12:00:00Z",
    "image_count": 4
  }
}
```

**Note**: The `image_count` field shows how many images were processed. While 1-4 images are accepted, using 4 images provides the best accuracy for attribute extraction.

## 🧪 Testing

### Integration Tests

```bash
# Start services first
docker compose up -d

# Run integration tests
pytest tests/ -v

# Or run specific test file
pytest tests/test_integration.py::test_analyze_endpoint_success -v
```

### Manual Testing

```bash
# Check all service health
curl http://localhost:8000/health
curl http://localhost:8001/health  # Vision service
curl http://localhost:8002/health  # Heuristic service  
curl http://localhost:8003/health  # LLM service

# Test with sample images
curl -X POST "http://localhost:8000/v1/items/analyze" \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

## 🔧 Development

### Project Structure

```
smartrobe/
├── orchestrator/          # Main API service
├── services/
│   ├── vision_classifier/ # Visual attribute extraction
│   ├── heuristic_model/   # Rule-based analysis
│   └── llm_model/         # Language model reasoning
├── shared/                # Common utilities and schemas
├── database/              # Database models and migrations
├── tests/                 # Integration tests
├── scripts/               # Development scripts
└── compose.yml     # Service orchestration
```

### Adding New Attributes

1. Update `shared/schemas.py` with new enum values
2. Modify appropriate service in `services/` to extract the attribute
3. Update database schema if needed
4. Add tests for the new attribute

### Environment Configuration

Key environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/smartrobe

# Service URLs (for inter-service communication)
VISION_SERVICE_URL=http://vision-classifier:8001
HEURISTIC_SERVICE_URL=http://heuristic-model:8002  
LLM_SERVICE_URL=http://llm-model:8003

# Image Processing
MAX_IMAGE_SIZE_MB=10
ALLOWED_IMAGE_FORMATS=jpeg,jpg,png,webp

# Performance
SERVICE_REQUEST_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=10
```

## 📊 Monitoring & Observability

### Health Checks

All services expose `/health` endpoints returning:

```json
{
  "service": "orchestrator",
  "status": "healthy",
  "version": "1.0.0", 
  "timestamp": "2024-01-01T12:00:00Z",
  "details": {
    "uptime_seconds": 3600,
    "database_connected": true
  }
}
```

### Structured Logging

All services emit structured JSON logs:

```json
{
  "time": "2024-01-01 12:00:00",
  "level": "INFO",
  "service": "orchestrator", 
  "message": "Item analysis completed successfully",
  "request_id": "uuid-v4",
  "total_time_ms": 1250
}
```

### Performance Metrics

- **Target Response Time**: 2-8 seconds
- **Concurrent Requests**: Up to 10 simultaneous 
- **Image Processing**: 1-4 images up to 10MB each (4 recommended)
- **Database Storage**: All results persisted with metadata

## 🚀 Production Deployment

### Scaling Considerations

- **Orchestrator**: Scale horizontally behind load balancer
- **Model Services**: Scale independently based on processing time
- **Database**: Use managed PostgreSQL (Cloud SQL, RDS, etc.)
- **Storage**: Implement shared storage (NFS, S3, etc.)

### Security

- Use HTTPS in production
- Secure database connections
- Implement rate limiting
- Add authentication/authorization as needed
- Validate and sanitize all inputs

### Monitoring

- Set up log aggregation (ELK, Splunk, etc.)
- Monitor response times and error rates
- Alert on service failures
- Track database performance

## 🛠️ Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker resources
docker system df
docker system prune  # Clean up if needed

# Check logs
docker compose logs orchestrator
```

**Database connection issues:**
```bash
# Verify PostgreSQL is running
docker compose ps postgres

# Check database logs
docker compose logs postgres
```

**Slow responses:**
```bash
# Check individual service health
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health

# Monitor resource usage
docker stats
```

### Debug Mode

Enable debug logging:

```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker compose restart
```

## 📄 License

[Add your license information]

## 🤝 Contributing

[Add contributing guidelines]

---

**Built with ❤️ for modern clothing marketplaces**

Extracting attributes from second-hand clothing items using computer vision, machine learning, and LLMs.