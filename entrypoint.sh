#!/bin/sh
set -e

echo "Downloading .env from S3..."
aws s3 cp s3://winssoft-leadsai/ai-b/.env /app/.env

echo "Loading environment variables..."
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  case "$line" in
    *=*)
      key="${line%%=*}"
      value="${line#*=}"
      value="${value%% #*}"
      value="${value#\"}"
      value="${value%\"}"
      value="${value#\'}"
      value="${value%\'}"
      export "$key=$value"
      ;;
  esac
done < /app/.env

echo "Starting ai-backend..."
exec uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1 --timeout-keep-alive 120