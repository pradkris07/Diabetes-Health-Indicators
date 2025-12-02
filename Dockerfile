# ---------- Stage 1: Build environment ----------
FROM python:3.12-slim AS build

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# ---------- Stage 2: Runtime environment ----------
FROM python:3.12-slim

# Install uv again (small footprint)
RUN pip install --no-cache-dir uv

# Set workdir and copy app from build stage
WORKDIR /app

# Copy the actual source code
COPY flask_app/ /app/
COPY models/final.yaml /app/models/final.yaml
COPY models/preprocessing.pkl /app/models/preprocessing.pkl

# Expose Flask port
EXPOSE 10000

# Default command to run Flask
#CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "app:app"]