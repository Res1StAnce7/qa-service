# Cloud Run Deployment Guide

This repo includes everything you need to run the concierge QA API locally or on Google Cloud Run.

## 1. Configure settings

```bash
cp config/settings.example.yaml config/settings.yaml
```
Edit `config/settings.yaml` with:
- `openai.api_key`
- `messages_api.base_url`
- any other tuning parameters

## 2. Local run (no Docker)

```bash
make install
QA_SERVICE_CONFIG=config/settings.yaml make run   # production-style
# or
QA_SERVICE_CONFIG=config/settings.yaml make run-dev
```

## 3. Local Docker run (optional)

```bash
make docker-build
make docker-run
```

## 4. Cloud Run deployment

1. Authenticate + set project:
   ```bash
   gcloud auth login
   gcloud config set project qa-system-569c0
   ```
2. Build & push container with Cloud Build + Artifact Registry:
   ```bash
   make gcloud-build REGION=us-central1 PROJECT=qa-system-569c0
   ```
3. Deploy to Cloud Run:
   ```bash
   make gcloud-deploy REGION=us-central1 PROJECT=qa-system-569c0
   ```
   The Dockerfile already honors Cloud Run's `PORT` env var, so no additional config is required.
4. Smoke tests:
   - `curl https://<cloud-run-url>/home`
   - `curl "https://<cloud-run-url>/ask?question=Who%20needs%20a%20payment%20check"`

## 5. Updating the service

1. Pull latest code
2. Rebuild image: `make gcloud-build`
3. Redeploy: `make gcloud-deploy`
