VENV?=.venv
PYTHON?=python3
PIP?=$(VENV)/bin/pip
UVICORN?=$(VENV)/bin/uvicorn
QA_SERVICE_CONFIG?=config/settings.yaml
REGION?=us-central1
PROJECT?=qa-system-569c0
IMAGE?=$(REGION)-docker.pkg.dev/$(PROJECT)/qa-service/qa-service:latest

.PHONY: help install run run-dev docker-build docker-run gcloud-build gcloud-deploy clean

help:
	@echo "Targets:"
	@echo "  make install        # Create venv + install deps"
	@echo "  make run            # Run with uvicorn (prod-style)"
	@echo "  make run-dev        # Run with autoreload"
	@echo "  make docker-build   # Build Docker image locally"
	@echo "  make docker-run     # Run Docker image locally"
	@echo "  make gcloud-build   # Build & push image via Cloud Build"
	@echo "  make gcloud-deploy  # Deploy latest image to Cloud Run"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install: $(VENV)

run: $(VENV)
	QA_SERVICE_CONFIG=$(QA_SERVICE_CONFIG) $(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --workers 2

run-dev: $(VENV)
	QA_SERVICE_CONFIG=$(QA_SERVICE_CONFIG) $(UVICORN) app.main:app --reload --host 127.0.0.1 --port 8000

docker-build:
	docker build -t qa-service:latest .

docker-run:
	docker run --rm -p 8000:8000 \
	  -e QA_SERVICE_CONFIG=/app/config/settings.yaml \
	  -v $(PWD)/config/settings.yaml:/app/config/settings.yaml:ro \
	  qa-service:latest

gcloud-build:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet || true
	gcloud artifacts repositories create qa-service \
	  --repository-format=docker --location=$(REGION) \
	  --description="QA Service images" || true
	gcloud builds submit --region=$(REGION) --tag $(IMAGE) .

gcloud-deploy:
	gcloud run deploy qa-service \
	  --image $(IMAGE) \
	  --region $(REGION) \
	  --allow-unauthenticated \
	  --set-env-vars QA_SERVICE_CONFIG=/app/config/settings.yaml

clean:
	rm -rf $(VENV)
