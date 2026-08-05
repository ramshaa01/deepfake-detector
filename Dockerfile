FROM python:3.11-slim

WORKDIR /app

# Install system deps: libGL needed by any opencv that ends up installed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user

COPY requirements.txt requirements-deploy.txt ./

# Install in specific order:
# 1. CPU-only torch first (much smaller than default CUDA build)
# 2. Install headless opencv BEFORE grad-cam to prevent grad-cam overriding it
# 3. Then install everything else (grad-cam will see headless already installed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir opencv-python-headless==4.10.0.84 && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user:user . /app
USER user

ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV PORT=8000

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn inference.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=\"*\""]
