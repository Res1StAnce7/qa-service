# QA Service

A FastAPI application that answers concierge questions using the latest member messages. It ships with an interactive web demo, `/ask` API endpoint, and a `/messages` explorer for debugging.

## Features
- **FastAPI backend** with `/ask`, `/messages`, `/home`, `/demo` routes
- **Retrieval + LLM pipeline** (OpenAI responses API + member-message context)
- **Interactive demo UI** that mimics ChatGPT (stop/resend/edit flows)
- **Message explorer** paginated via `/messages`
- **Ready for Cloud Run** (Dockerfile + Makefile + docs/DEPLOYMENT.md)

## Project Layout
```
app/
  main.py          # FastAPI entrypoint and HTML demo
  service.py       # Retrieval + LLM orchestration
  llm.py           # OpenAI client wrapper
  message_client.py# Messages API client
  schemas.py       # Pydantic response schemas
config/
  settings.example.yaml  # Sample config; copy to settings.yaml
requirements.txt
Dockerfile
Makefile
README.md
```

## Requirements
- Python 3.11+
- OpenAI API key with access to the configured models
- Upstream messages API endpoint
- (Optional) Docker + gcloud CLI for Cloud Run deployment

## Getting Started
1. Install dependencies and configure settings:
   ```bash
   cp config/settings.example.yaml config/settings.yaml
   # edit the file with real OpenAI + messages API values
   make install
   ```
2. Run locally (no Docker):
   ```bash
   QA_SERVICE_CONFIG=config/settings.yaml make run        # production-style
   # or
   QA_SERVICE_CONFIG=config/settings.yaml make run-dev    # auto reload
   ```
   Open `http://localhost:8000/demo` for the chat UI or `http://localhost:8000/docs` for the OpenAPI UI.

## Cloud Run Deployment
See `docs/DEPLOYMENT.md` for full instructions. TL;DR:
```bash
REGION=us-central1
PROJECT=qa-system-569c0
make gcloud-build REGION=$REGION PROJECT=$PROJECT   # builds + pushes via Cloud Build
make gcloud-deploy REGION=$REGION PROJECT=$PROJECT # deploys to Cloud Run (service name qa-service)
```
Cloud Run will expose a URL like `https://qa-service-xxxxxx-uc.a.run.app`. Use `/home`, `/demo`, `/ask`, `/messages` there.

## Configuration
- `config/settings.yaml` controls OpenAI models, messages API base URL, retrieval parameters, etc.
- Override the config path by setting `QA_SERVICE_CONFIG=/path/to/config.yaml` (Makefile and Dockerfile already set/forward this).
- When deploying to Cloud Run, the Dockerfile copies `settings.example.yaml` into the image as a default. Replace it with secure secrets via Secret Manager or a different config path before production.

## Useful Commands
- `make run` / `make run-dev`: start FastAPI via uvicorn
- `make docker-build` / `make docker-run`: local Docker smoke test
- `make gcloud-build` / `make gcloud-deploy`: Cloud Run pipeline
- `make clean`: remove the virtualenv
