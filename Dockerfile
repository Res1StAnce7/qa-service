# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config

# Ensure a default config exists even if override is mounted later
RUN cp config/settings.example.yaml /app/config/settings.yaml

ENV QA_SERVICE_CONFIG=/app/config/settings.yaml

EXPOSE 8080
CMD ["sh", "-c", "test -f $QA_SERVICE_CONFIG || { echo 'Missing config' >&2; exit 1; }; uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
