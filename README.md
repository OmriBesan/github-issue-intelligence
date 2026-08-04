# GitHub Issue Intelligence 🧠🐛

> An AI/ML system that analyses GitHub issues from real open-source repositories,
> classifies them by type (Bug, Enhancement, Documentation), and retrieves semantically similar historical issues.

![GitHub Issue Intelligence UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)
![Model](https://img.shields.io/badge/Model-HuggingFace-F9AB00?style=flat-square&logo=huggingface)

---

## Project Overview

GitHub repositories accumulate thousands of issues. Manually triaging each
issue costs maintainer time and slows down project management.

This project is a complete end-to-end Machine Learning system that automates GitHub issue triage. 
It was built from scratch in Python and covers every step of the Data Science lifecycle:

1. **Data Collection & Preprocessing**: Downloaded 12,190 raw issues from `scikit-learn`, cleaned them, and established a solid label taxonomy.
2. **Classical ML (Baselines)**: Trained `Logistic Regression` and `SVC` models using `TF-IDF` vectorization, establishing strong, fast baselines.
3. **Advanced ML (Deep Learning)**: Fine-tuned a Transformer model (`DistilBERT`) using `PyTorch` and `Hugging Face` for state-of-the-art text classification.
4. **Semantic Search (RAG)**: Implemented duplicate detection using `sentence-transformers` and `cosine_similarity` to find issues with similar semantic meaning, not just overlapping keywords.
5. **Production API**: Wrapped the models in a robust REST API using `FastAPI`.
6. **Frontend UI**: Built an interactive web application using `Streamlit` allowing users to test the models in real-time.

---

## Architecture

* **Backend**: `FastAPI` serving predictions and semantic search endpoints.
* **Frontend**: `Streamlit` application.
* **Machine Learning**: `scikit-learn` (Baselines), `transformers` (Fine-tuning), `sentence-transformers` (Embeddings).

## How to Run the Project Locally

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Models
Before running the API, you need to generate the pre-trained assets (the `.pkl` files):
```bash
python scripts/build_api_assets.py
```

### 3. Run the API (Backend)
In one terminal, start the FastAPI server:
```bash
uvicorn src.issue_intelligence.api.main:app --reload
```
You can view the interactive API documentation (Swagger) at: http://localhost:8000/docs

### 4. Run the UI (Frontend)
In a second terminal, start the Streamlit frontend:
```bash
streamlit run src/issue_intelligence/ui/app.py
```
The browser will automatically open at `http://localhost:8501`. Type an issue title and description, and click "Analyze"!

---

## Project Structure

- `data/` - Raw, processed, and model assets (`.jsonl`, `.pkl`).
- `docs/` - Project documentation, planning, and architectural decisions.
- `notebooks/` - Jupyter notebooks documenting exploratory data analysis (EDA), model training, evaluation, and semantic search.
- `scripts/` - Automation scripts for downloading data, generating dummy data, and building API assets.
- `src/issue_intelligence/` - Core Python package containing models, API, and UI code.
- `tests/` - Comprehensive `pytest` suite ensuring code correctness and data integrity (195 tests).

## Next Steps (MLOps)
See `docs/deployment_plan.md` for a complete architectural design on how to deploy this project to production using Docker and GitHub Webhooks/Actions.
