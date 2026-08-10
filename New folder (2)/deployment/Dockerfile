# Dockerfile for the FastAPI prediction service.
# Build:  docker build -t rossmann-api -f deployment/Dockerfile .
# Run:    docker run -p 8000:8000 -e JWT_SECRET_KEY=change-this-in-prod rossmann-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/processed/sales_model.joblib data/processed/sales_model.joblib

ENV JWT_SECRET_KEY=""
EXPOSE 8000

# Basic container-level health check, useful for orchestrators (Docker Swarm/K8s)
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
