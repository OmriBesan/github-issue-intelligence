"""FastAPI application factory and endpoints."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from issue_intelligence.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SimilarRequest,
    SimilarResponse,
)
from issue_intelligence.api.service import InferenceService
from issue_intelligence.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_MODEL_PATH = Path("models/classical/final_linear_svc.joblib")
DEFAULT_RETRIEVAL_PATH = Path("models/retrieval/similar_issues.joblib")


def create_app(model_path: Path | None = None, retrieval_path: Path | None = None) -> FastAPI:
    """Application factory for testing and production."""

    # Resolve the model path
    if model_path is None:
        env_path = os.environ.get("ISSUE_MODEL_PATH")
        if env_path:
            resolved_path = Path(env_path)
        else:
            resolved_path = DEFAULT_MODEL_PATH
    else:
        resolved_path = model_path

    if retrieval_path is None:
        env_retrieval_path = os.environ.get("ISSUE_RETRIEVAL_PATH")
        if env_retrieval_path:
            resolved_retrieval_path = Path(env_retrieval_path)
        else:
            resolved_retrieval_path = DEFAULT_RETRIEVAL_PATH
    else:
        resolved_retrieval_path = retrieval_path

    service = InferenceService(resolved_path)
    retrieval_service = RetrievalService(resolved_retrieval_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        service.load_model()
        retrieval_service.load_artifact()
        yield
        # Shutdown
        pass

    app = FastAPI(
        title="GitHub Issue Intelligence API",
        description="Text classification for GitHub Issues",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        resp = {
            "status": "not_ready",
            "model_loaded": service.is_ready,
            "retrieval_loaded": retrieval_service.is_ready
        }
        if service.is_ready:
            resp["status"] = "ok"
        return resp

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info():
        """Return non-sensitive model metadata."""
        if not service.is_ready:
            return {
                "model_loaded": False,
                "expected_fields": ["title", "body"],
                "labels": None,
                "vocabulary_size": None,
                "model_type": None,
            }

        return {
            "model_loaded": True,
            "expected_fields": ["title", "body"],
            "labels": service.classes_,
            "vocabulary_size": service.vocab_size,
            "model_type": "TF-IDF + LinearSVC",
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Predict the issue class from title and body."""
        if not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or unavailable."
            )

        try:
            result = service.predict(title=request.title, body=request.body)
            return result
        except ValueError as e:
            # Handle empty combined text after preprocessing
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during prediction."
            )

    @app.post("/similar", response_model=SimilarResponse)
    async def similar(request: SimilarRequest):
        """Retrieve similar issues based on title and body."""
        if not retrieval_service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retrieval artifact is not loaded or unavailable."
            )

        try:
            results = retrieval_service.search(
                title=request.title,
                body=request.body,
                top_k=request.top_k,
                label_filter=request.label_filter,
                exclude_issue_id=request.exclude_issue_id
            )
            
            # The count from metadata
            indexed_count = retrieval_service.artifact_metadata.get("indexed_issue_count", 0)
            
            return {
                "results": results,
                "retrieval_method": "tfidf_cosine",
                "indexed_issue_count": indexed_count
            }
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during retrieval."
            )

    return app

# The default exported application for Uvicorn
app = create_app()
