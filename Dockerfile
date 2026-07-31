FROM python:3.11-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user that HuggingFace Spaces requires
RUN useradd -m -u 1000 user

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Render sets the PORT environment variable dynamically
ENV PORT=8000
EXPOSE $PORT

# Command to run the FastAPI app via Uvicorn
CMD ["sh", "-c", "uvicorn inference.main:app --host 0.0.0.0 --port ${PORT}"]
