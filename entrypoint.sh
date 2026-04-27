#!/bin/sh
set -e

echo "Downloading .env from S3..."
aws s3 cp s3://winssoft-leadsai/ai-b/.env /app/.env

echo "Loading environment variables..."
export $(grep -v '^#' /app/.env | grep '=' | xargs)

echo "Starting ai-backend..."
exec uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1 --timeout-keep-alive 120