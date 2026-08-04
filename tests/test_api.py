"""API tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from issue_intelligence.api.app import create_app


@pytest.fixture
def dummy_pipeline_path():
    """Create a real, fitted sklearn pipeline for testing."""
    X = ["fix the bug", "update the docs", "add new feature"]
    y = ["Bug", "Documentation", "Enhancement"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC(random_state=42))
    ])
    pipeline.fit(X, y)

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "dummy.joblib"
        joblib.dump(pipeline, model_path)
        yield model_path


def test_health_model_missing():
    """Test health endpoint when model does not exist."""
    app = create_app(
        model_path=Path("non_existent.joblib"),
        retrieval_path=Path("non_existent.joblib"),
    )
    # We must explicitly call startup for tests if not using TestClient context manager
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "not_ready",
            "model_loaded": False,
            "retrieval_loaded": False,
        }


def test_predict_model_missing():
    """Test predict returns 503 when model is missing."""
    app = create_app(model_path=Path("non_existent.joblib"))
    with TestClient(app) as client:
        response = client.post("/predict", json={"title": "bug", "body": "fix"})
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()


def test_model_info_missing():
    """Test model-info when model is missing."""
    app = create_app(model_path=Path("non_existent.joblib"))
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert not data["model_loaded"]
        assert data["expected_fields"] == ["title", "body"]
        assert data["labels"] is None


def test_health_model_loaded(dummy_pipeline_path):
    """Test health endpoint with loaded model."""
    app = create_app(
        model_path=dummy_pipeline_path,
        retrieval_path=Path("non_existent.joblib"),
    )
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "model_loaded": True,
            "retrieval_loaded": False,
        }


def test_model_info_loaded(dummy_pipeline_path):
    """Test model-info with loaded model."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"]
        assert data["expected_fields"] == ["title", "body"]
        assert data["labels"] == ["Bug", "Documentation", "Enhancement"]
        assert data["vocabulary_size"] > 0
        assert data["model_type"] == "TF-IDF + LinearSVC"


def test_predict_success(dummy_pipeline_path):
    """Test successful prediction with both fields."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"title": "fix the bug", "body": "stack trace"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_label" in data
        assert data["predicted_label"] in ["Bug", "Documentation", "Enhancement"]
        assert "decision_scores" in data
        assert "Bug" in data["decision_scores"]
        assert "decision_margin" in data
        assert isinstance(data["decision_margin"], float)
        assert data["model_name"] == "LinearSVC"


def test_predict_title_only(dummy_pipeline_path):
    """Test successful prediction with title only."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        response = client.post("/predict", json={"title": "fix the bug"})
        assert response.status_code == 200
        assert "predicted_label" in response.json()


def test_predict_body_only(dummy_pipeline_path):
    """Test successful prediction with body only."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        response = client.post("/predict", json={"body": "fix the bug stack trace"})
        assert response.status_code == 200
        assert "predicted_label" in response.json()


def test_predict_empty_validation(dummy_pipeline_path):
    """Test validation fails for empty payload."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        # Both omitted (None)
        response = client.post("/predict", json={})
        assert response.status_code == 422

        # Both blank
        response = client.post("/predict", json={"title": "   ", "body": "\n  \n"})
        assert response.status_code == 422
        assert "cannot both be blank" in response.json()["detail"][0]["msg"]


def test_predict_unicode_and_multiline(dummy_pipeline_path):
    """Test prediction with unicode and multiline code."""
    app = create_app(model_path=dummy_pipeline_path)
    with TestClient(app) as client:
        body = """
Here is a code snippet:
```python
def foo():
    return "🚀"
```
It crashes.
"""
        response = client.post(
            "/predict", json={"title": "Crash with unicode 🚀", "body": body}
        )
        assert response.status_code == 200
