# QA Service

A FastAPI application that answers concierge questions using the latest member messages. It ships with an interactive web demo, `/ask` API endpoint. 

Home Page: https://qa-service-780002810623.us-central1.run.app/home 
Demo Page: https://qa-service-780002810623.us-central1.run.app/demo
Ask API Endpoint: https://qa-service-780002810623.us-central1.run.app/ask

## Features
- **FastAPI backend** with `/ask`, `/messages`, `/home`, `/demo` routes
- **Retrieval + LLM pipeline** (OpenAI responses API + member-message context)
- **Interactive demo UI** that mimics ChatGPT (stop/resend/edit flows)
- **Message explorer** paginated via `/messages`
- **Cloud Run** (Dockerfile + Makefile + docs/DEPLOYMENT.md)

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
   make run        # production-style
   # or
   make run-dev    # auto reload
   ```
   Open `http://localhost:8000/demo` for the chat UI.

## Cloud Run Deployment
See `docs/DEPLOYMENT.md` for full instructions. TL;DR:
```bash
REGION=us-central1
PROJECT=$PROJECTID
make gcloud-build REGION=$REGION PROJECT=$PROJECT   
make gcloud-deploy REGION=$REGION PROJECT=$PROJECT 
```

## Configuration
- `config/settings.yaml` controls OpenAI models, messages API base URL, retrieval parameters, etc.
- Override the config path by setting `QA_SERVICE_CONFIG=/path/to/config.yaml` (Makefile and Dockerfile already set/forward this).
- When deploying to Cloud Run, the Dockerfile copies `settings.example.yaml` into the image as a default. Replace it with secure secrets via Secret Manager or a different config path before production.

## Useful Commands
- `make run` / `make run-dev`: start FastAPI via uvicorn
- `make docker-build` / `make docker-run`: local Docker smoke test
- `make gcloud-build` / `make gcloud-deploy`: Cloud Run pipeline
- `make clean`: remove the virtualenv

## Bonus 1: Design Notes
- **Hybrid retrieval + generation**: Consider LangChian framework with a sparse BM25 + dense embedding rerank stack to keep latency low while remaining resilient when embeddings (store in vector database like Milvus) miss keywords.
- **Pre-computed answer snippets**: Explore maintaining curated templates per intent and only letting the LLM stitch them together, which would have improved determinism but required higher ops overhead to keep the snippets fresh.
- **Agentic tool-use workflow**: Evaluatd introducing a lightweight planner + tools (messages API, policy lookup, fallback answers) so the system could verify facts before responding.

## Bonus 2: Data Insights
The samples dataset obtained from the provided API contains 1,000 records across 10 members, with no duplicate IDs, name mismatches, or timestamp anomalies. However, 29 messages include potential personally identifiable information (PII) such as phone numbers, credit card or passport mentions, and third-party contacts. We will a need for redaction and structured profile storage. Over 100 messages contain ambiguous temporal terms like “tomorrow,” “next week,” and “this Friday,” which could lead to scheduling errors if not normalized to absolute dates. There are reference inconsistencies, for example, Lily O’Sullivan alternately requests both window and aisle seats, while Fatima El-Tahir lists both smoking and hypoallergenic/low-scent preferences, which may conflict. To improve data reliability, member preferences and contact details should be standardized in structured fields, date expressions normalized on ingestion, and a PII-detection layer introduced to flag sensitive content automatically.
