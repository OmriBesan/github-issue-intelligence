# GitHub Issue Intelligence - Deployment Plan

## 1. Overview
This document outlines the strategy for taking the GitHub Issue Intelligence system from a local prototype to a production-ready application. The production system will automatically triage incoming GitHub issues, classify them (e.g., Bug, Enhancement, Documentation), and flag potential duplicates.

## 2. Architecture & Containerization
To ensure consistency across environments, both the backend API and frontend UI will be containerized using **Docker**.

- **Backend (FastAPI)**: Packaged in a lightweight Python Docker image (e.g., `python:3.10-slim`). The image will copy the pre-trained models (`data/models/*.pkl`) to ensure fast startup times.
- **Frontend (Streamlit)**: Packaged in a separate Docker image, communicating with the backend via internal network routing.

*Docker Compose* can be used to spin up the entire stack locally or on a single VM.

## 3. GitHub Integration (Webhooks & Actions)
To make the system truly autonomous, it needs to integrate directly with GitHub.

### Option A: GitHub Webhooks (Real-time)
1. **Webhook Registration**: Configure the target GitHub repository to send a `POST` request to our FastAPI backend whenever a new issue is opened (`issues` event, action `opened`).
2. **API Endpoint**: We will add a new endpoint (e.g., `POST /webhook/github`) that parses the incoming GitHub payload, extracts the issue title and body, and passes them to our internal `predict` and `search` functions.
3. **Automated Tagging**: The backend uses the GitHub REST API (using a bot Personal Access Token) to automatically add the predicted label (e.g., `bug`) to the issue.
4. **Duplicate Commenting**: If a duplicate is found with a high confidence score, the bot will automatically comment on the issue: *"This issue looks very similar to #123. Maintainers will review it shortly."*

### Option B: GitHub Actions (Event-Driven Pipeline)
Alternatively, we can create a custom GitHub Action. When an issue is opened, the Action runner sends the issue text to our hosted API, receives the prediction, and uses the standard `actions/github-script` to apply labels and comments. This avoids the need to expose our API directly to the public internet for webhooks.

## 4. MLOps: Model Monitoring & Retraining
Machine learning models degrade over time as the nature of the data changes (Data Drift).

- **Logging**: The FastAPI backend will log all incoming requests and predictions to a database (e.g., PostgreSQL or Elasticsearch).
- **Feedback Loop**: If a maintainer changes an automatically applied label (e.g., the bot labeled it `bug`, but a human changes it to `enhancement`), this change is recorded as "ground truth".
- **Automated Retraining**: A CI/CD pipeline (e.g., GitHub Actions or Airflow) runs a monthly job that pulls the new data, retrains the models (Classical and Transformer), re-evaluates the metrics, and if the new model performs better, deploys it to production.

## 5. Cloud Hosting
The system can be deployed to various cloud providers:
- **Compute**: Google Cloud Run, AWS App Runner, or Heroku are perfect for hosting the stateless Docker containers. 
- **Storage**: Model artifacts (`.pkl` files or Hugging Face weights) should be stored in a central object storage (like AWS S3 or Google Cloud Storage) and pulled during the CI/CD build phase.

## 6. Security Considerations
- **Authentication**: The FastAPI server should secure its endpoints (especially the webhook endpoint) using API keys or verifying GitHub webhook secret signatures.
- **Rate Limiting**: Prevent abuse by limiting the number of requests per minute per IP.
