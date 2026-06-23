# vision-forge container image.
# Builds a CPU-capable image. For GPU, swap the base for an NVIDIA CUDA image
# and install the matching torch wheel.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VF_HOST=0.0.0.0 \
    VF_PORT=8000

# System libs needed by opencv-python-headless at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the application.
COPY . .
RUN pip install --no-deps -e .

EXPOSE 8000

# Healthcheck hits the FastAPI liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "visionforge.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
