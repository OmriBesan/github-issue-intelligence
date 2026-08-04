# GitHub Issue Intelligence

**Repository:** https://github.com/OmriBesan/github-issue-intelligence

A machine-learning system for classifying **scikit-learn GitHub issues** as:

- **Bug**
- **Documentation**
- **Enhancement**

The project also retrieves lexically similar historical issues using **TF-IDF cosine similarity** and provides a local demonstration through **FastAPI** and **Streamlit**.

> The current trained model is specific to scikit-learn. The architecture can be adapted to other repositories, but reliable use elsewhere requires repository-specific labelled data, retraining, and evaluation.

---

## Main Result

The final model is a **TF-IDF + LinearSVC** pipeline evaluated on a held-out temporal test set containing 856 newer scikit-learn issues from 2024–2026.

| Metric | Result |
|---|---:|
| Macro F1 | **0.9300** |
| Accuracy | **0.9369** |
| Weighted F1 | **0.9368** |

The selected LinearSVC model also outperformed the BERT-Tiny benchmark while being faster, smaller, and easier to run on CPU.

---

## What the System Does

Given a GitHub issue title and body, the system can:

1. Predict whether the issue is a Bug, Documentation issue, or Enhancement.
2. Display the raw LinearSVC decision scores and decision margin.
3. Retrieve lexically similar historical scikit-learn issues.
4. Present the results through a local Streamlit interface.

Decision scores are **not probabilities** and are not displayed as confidence percentages.

Retrieved issues are ranked by **TF-IDF cosine similarity**. Similar issues are not necessarily duplicates.

---

## Project Pipeline

```text
GitHub issue collection
        ↓
Dataset audit and label mapping
        ↓
Preprocessing and deterministic splitting
        ↓
Baselines and classical ML models
        ↓
Robustness experiments and temporal tuning
        ↓
BERT-Tiny comparison
        ↓
Final held-out temporal evaluation
        ↓
FastAPI inference service
        ↓
TF-IDF similar-issue retrieval
        ↓
Streamlit demonstration interface
```

---

## Dataset

The project uses public issues from the `scikit-learn/scikit-learn` GitHub repository.

| Dataset stage | Records |
|---|---:|
| Raw collected issues | 12,190 |
| Final labelled modelling dataset | 5,710 |
| Temporal train split | 3,996 |
| Temporal validation split | 858 |
| Held-out temporal test split | 856 |

Final class distribution:

| Class | Count |
|---|---:|
| Bug | 2,274 |
| Enhancement | 2,039 |
| Documentation | 1,397 |

The held-out test set was isolated until the final evaluation.

---

## Model Comparison

### Temporal validation

| Model | Macro F1 |
|---|---:|
| Stratified-random baseline | 0.3376 |
| Logistic Regression | 0.9051 |
| LinearSVC | 0.9060 |
| SGDClassifier | 0.9077 |
| Tuned LinearSVC | **0.9087** |
| BERT-Tiny | 0.8550 |

### Selected final configuration

- TF-IDF lowercasing
- unigrams and bigrams
- `min_df=5`
- `max_df=0.95`
- sublinear term frequency
- LinearSVC with `C=0.3`
- balanced class weights

The tuned training-only vocabulary contained 20,473 terms.  
After refitting on train + validation, the final saved model contained 27,129 terms.

---

## Robustness Checks

The project includes:

- title-only and body-only experiments,
- target-label word masking,
- prefix-free evaluation,
- performance by year,
- performance by text length,
- exact-duplicate checks,
- near-duplicate checks,
- decision-margin analysis,
- influential-term review,
- misclassification review.

No exact train-validation duplicates were found, and no near-duplicate pairs were found at cosine thresholds of 0.90, 0.95, or 0.99.

---

## Local Architecture

```text
Streamlit UI
    ↓ HTTP
FastAPI service
    ├── /health
    ├── /model-info
    ├── /predict
    └── /similar
          ↓
TF-IDF + LinearSVC classifier
TF-IDF cosine retrieval index
```

The Streamlit interface communicates only with FastAPI. It does not load datasets or model artifacts directly.

---

## Run the Local Demonstration

### 1. Start the FastAPI backend

```powershell
.venv\Scripts\python -m uvicorn issue_intelligence.api.app:app --reload
```

### 2. Start Streamlit in a second terminal

```powershell
.venv\Scripts\python -m streamlit run src/issue_intelligence/ui/app.py
```

The frontend uses:

```text
http://127.0.0.1:8000
```

by default. A different backend can be configured with the `ISSUE_API_URL` environment variable.

---

## Example Input

**Title**

```text
RandomForestClassifier raises an error when fitting sparse input
```

**Body**

```text
Calling fit with a sparse matrix produces an unexpected ValueError.
```

The verified local smoke test predicted **Bug** and returned five lexically similar historical issues.

---

## API Summary

### `GET /health`

Reports classifier and retrieval availability.

### `GET /model-info`

Reports the loaded model type, labels, and vocabulary information.

### `POST /predict`

Example request:

```json
{
  "title": "RandomForestClassifier raises an error when fitting sparse input",
  "body": "Calling fit with a sparse matrix produces an unexpected ValueError."
}
```

### `POST /similar`

Example request:

```json
{
  "title": "RandomForestClassifier raises an error when fitting sparse input",
  "body": "Calling fit with a sparse matrix produces an unexpected ValueError.",
  "top_k": 5
}
```

---

## Quality Checks

Final repository verification:

| Check | Result |
|---|---|
| Automated tests | **341 passed** |
| Ruff | **All checks passed** |
| Dependency check | **No broken requirements** |
| Stage 4B metrics | **Unchanged after final evaluation** |
| Generated model and retrieval artifacts | **Gitignored** |

---

## Limitations

- The model was trained and evaluated only on scikit-learn issues.
- GitHub labels may contain human inconsistency and historical noise.
- Issues without a reliable target type were excluded.
- LinearSVC decision scores are uncalibrated.
- TF-IDF retrieval measures lexical similarity, not full semantic equivalence.
- Similar retrieved issues are not confirmed duplicates.
- The transformer benchmark used BERT-Tiny under CPU and sequence-length constraints.
- The FastAPI service and Streamlit interface are local demonstration components, not production deployment.

---

## Project Report

The complete English project report is available in:

[`REPORT.md`](REPORT.md)

It describes the data journey, modelling decisions, experiments, findings, system architecture, limitations, and use of AI tools.

---

## AI Assistance

AI coding assistants were used for planning, implementation support, debugging, documentation, and review.

The project team remained responsible for:

- defining the task and labels,
- protecting the held-out test set,
- reviewing generated code,
- verifying reported metrics,
- correcting unsupported claims,
- running tests and linting,
- understanding the final implementation.

See [`docs/gpt_usage.md`](docs/gpt_usage.md) for the detailed AI-usage record.

---

## Authors

- Omri Besan
- Leon Pasternak
