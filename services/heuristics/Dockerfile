FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared code (database, schemas, config, etc.)
COPY ./shared ./shared

# Copy service code
COPY ./services/heuristics ./services/heuristics

# Install uv for faster package management
RUN pip install uv

# Install dependencies using uv
RUN uv pip install --system -e ./services/heuristics

# Create shared storage directory
RUN mkdir -p /app/shared_storage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "services.heuristics.app:app", "--host", "0.0.0.0", "--port", "8000"]
