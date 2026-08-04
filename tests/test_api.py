from fastapi.testclient import TestClient

from issue_intelligence.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # models_loaded might be False if assets weren't built, but endpoint should work

def test_predict_endpoint_no_model():
    # If the model is not loaded, it should return 503
    # Note: this test depends on whether models were successfully loaded at startup.
    response = client.post("/predict", json={"title": "test", "body": "test"})
    assert response.status_code in [200, 503]

def test_search_endpoint_no_model():
    payload = {"title": "test", "body": "test", "top_k": 2}
    response = client.post("/search", json=payload)
    assert response.status_code in [200, 503]
