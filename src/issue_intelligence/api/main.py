import contextlib
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from issue_intelligence.models.retrieval import IssueRetriever


# Data structures
class PredictRequest(BaseModel):
    title: str
    body: str

class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

class SearchRequest(BaseModel):
    title: str
    body: str
    top_k: int = 3

class SearchResult(BaseModel):
    text: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]

# Global variables for models
classifier = None
retriever = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, retriever

    model_dir = Path("data/models")
    clf_path = model_dir / "classifier.pkl"
    ret_path = model_dir / "retriever.pkl"

    if not clf_path.exists() or not ret_path.exists():
        print("Warning: Model assets not found. Run scripts/build_api_assets.py first.")
    else:
        # Load classifier
        classifier = joblib.load(clf_path)

        # Load retriever
        retriever_data = joblib.load(ret_path)
        retriever = IssueRetriever(model_name="all-MiniLM-L6-v2")
        retriever.corpus_texts = retriever_data["corpus_texts"]
        retriever.corpus_embeddings = retriever_data["corpus_embeddings"]

        print("Models loaded successfully.")

    yield
    # Cleanup if needed

app = FastAPI(title="GitHub Issue Intelligence API", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": classifier is not None and retriever is not None
    }


@app.post("/predict", response_model=PredictResponse)
def predict_issue(request: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")

    combined_text = f"{request.title} {request.body}"
    prediction = classifier.predict([combined_text])[0]

    # Get probabilities
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba([combined_text])[0]
        classes = classifier.classes_
        prob_dict = {str(c): float(p) for c, p in zip(classes, probs, strict=False)}
    else:
        prob_dict = {}

    return PredictResponse(prediction=prediction, probabilities=prob_dict)


@app.post("/search", response_model=SearchResponse)
def search_similar_issues(request: SearchRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded.")

    combined_text = f"{request.title} {request.body}"
    results = retriever.search(combined_text, top_k=request.top_k)

    formatted_results = [
        SearchResult(text=res["text"], score=res["score"])
        for res in results
    ]
    return SearchResponse(results=formatted_results)
