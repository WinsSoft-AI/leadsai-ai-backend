# ═══════════════════════════════════════════════════════════════════════════════
# LeadsAI — AI Backend Dockerfile
# Heavy ML: Whisper STT + TTS + YOLOv8 + CLIP + ChromaDB + Gemini
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py db_init.py models.py gemini_client.py rag_engine.py \
     stt_service.py tts_service.py cv_service.py ./
COPY yolov8n.pt ./

RUN mkdir -p /app/chroma_data /app/data

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]