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
)
from issue_intelligence.api.service import InferenceService

logger = logging.getLogger(__name__)

# Default model path relative to project root
DEFAULT_MODEL_PATH = Path("models/classical/final_linear_svc.joblib")


def create_app(model_path: Path | None = None) -> FastAPI:
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

    service = InferenceService(resolved_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        service.load_model()
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
        if service.is_ready:
            return {"status": "ok", "model_loaded": True}
        return {"status": "not_ready", "model_loaded": False}

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

    return app

# The default exported application for Uvicorn
app = create_app()
