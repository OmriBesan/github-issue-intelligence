import pytest
from fastapi.testclient import TestClient

from issue_intelligence.api.app import create_app

@pytest.fixture
def client_missing_retrieval():
    # Pass dummy paths that do not exist
    app = create_app(
        model_path=None, 
        retrieval_path=None
    )
    with TestClient(app) as client:
        yield client

def test_similar_503_when_unavailable(client_missing_retrieval):
    response = client_missing_retrieval.post("/similar", json={"title": "bug"})
    assert response.status_code == 503
    assert "Retrieval artifact is not loaded or unavailable" in response.json()["detail"]

from unittest.mock import patch

def dummy_load_model(self):
    self.is_ready = True

def dummy_load_artifact(self):
    self.is_ready = False

def test_predict_works_when_retrieval_unavailable():
    app = create_app()
    with patch("issue_intelligence.api.service.InferenceService.load_model", new=dummy_load_model), \
         patch("issue_intelligence.api.service.InferenceService.predict", return_value={"predicted_label": "Bug", "decision_scores": {"Bug": 1.0, "Documentation": -1.0, "Enhancement": -1.0}, "decision_margin": 2.0, "model_name": "LinearSVC"}), \
         patch("issue_intelligence.retrieval.service.RetrievalService.load_artifact", new=dummy_load_artifact):
        
        with TestClient(app) as client:
            # predict should work
            response = client.post("/predict", json={"title": "test"})
            assert response.status_code == 200
            assert response.json()["predicted_label"] == "Bug"
            
            # similar should fail
            response = client.post("/similar", json={"title": "test"})
            assert response.status_code == 503
def test_similar_validation_empty():
    app = create_app()
    with TestClient(app) as client:
        # even if not ready, validation runs first usually, but let's check
        response = client.post("/similar", json={"title": "", "body": "   "})
        assert response.status_code == 422

def test_similar_validation_top_k():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/similar", json={"title": "bug", "top_k": 20})
        assert response.status_code == 422
        
        response = client.post("/similar", json={"title": "bug", "top_k": 0})
        assert response.status_code == 422
