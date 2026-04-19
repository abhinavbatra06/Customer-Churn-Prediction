FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer if pyproject.toml unchanged)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]