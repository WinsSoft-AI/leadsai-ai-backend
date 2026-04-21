# ═══════════════════════════════════════════════════════════════════════════════
# OmniChat — AI Backend Dockerfile
# Heavy ML: Whisper STT + TTS + YOLOv8 + CLIP + ChromaDB + Gemini
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# System dependencies for ML workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (heavy layer — cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py db_init.py models.py gemini_client.py rag_engine.py \
     stt_service.py tts_service.py cv_service.py ./
COPY yolov8n.pt ./

# Create persistent directories
RUN mkdir -p /app/chroma_data /app/data

# Expose the port (internal only — NOT publicly accessible)
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run with single worker (ML models are memory-heavy)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--timeout-keep-alive", "120"]
