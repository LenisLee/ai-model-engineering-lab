FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard]

# Copy source
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY pyproject.toml .
RUN pip install -e .

# Model is mounted at runtime or baked in
# COPY outputs/checkpoints/ outputs/checkpoints/

ENV MODEL_PATH=outputs/checkpoints
EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
