# Project Plan — GitHub Issue Intelligence

This document describes the planned stages of the project.
Stages are executed one at a time and are subject to revision
based on findings from the dataset audit in Stage 1.

---

## Stage 0 — Project Initialisation ✅

**Goal:** Create a clean, reproducible project skeleton.

Tasks:
- Scaffold directory structure
- Create virtual environment (Python 3.14)
- Install initial dependencies
- Write initial documentation
- Run basic verification tests

**Deliverables:**
- Working project structure
- `requirements.txt`
- Passing `pytest` and clean `ruff` output

---

## Stage 1 — Dataset Collection and Audit ✅

**Goal:** Download GitHub issues from one or more open-source repositories
and understand the data before committing to any label scheme or model.

Planned repositories (to be confirmed after audit):
1. scikit-learn — likely primary dataset
2. pandas — possible secondary dataset
3. vLLM — possible advanced AI-domain dataset

Audit checks:
- Total number of issues
- Percentage with useful labels
- Label frequencies and class imbalance
- Missing titles or descriptions
- Issue text lengths
- Issues with multiple labels
- Label changes over time
- Possible data leakage
- Pull requests mistakenly returned by the Issues API
- Temporal distribution of issues

**Deliverables:**
- `scripts/collect_issues.py` — downloads raw issues via GitHub API
- `data/raw/` — raw downloaded JSON files (not committed)
- `notebooks/01_data_audit.ipynb` — exploratory audit notebook
- Updated `docs/decisions.md` — final label scheme decision

---

## Stage 2 — Preprocessing and Baselines (In Progress)

**Goal:** Establish the cleaned dataset and performance floor before applying any real ML model.

**Stage 2A:** Cleaned Modelling Dataset Preparation ✅ (Complete)
**Stage 2B:** Data Splitting (Next)

Models:
- Majority-class baseline (always predicts the most frequent class)
- `DummyClassifier` from scikit-learn (stratified random prediction)
- Possibly a simple rule-based baseline using keyword matching

**Why this matters:** A model that cannot beat a majority-class baseline is not
useful. Baselines give us a concrete reference point for all future experiments.

**Deliverables:**
- `src/issue_intelligence/baselines.py`
- Baseline evaluation results recorded in `reports/`

---

## Stage 3 — Classical NLP Models (TF-IDF)

**Goal:** Apply classical text classification with feature engineering.

Feature engineering:
- TF-IDF on issue title + description

Models:
- Logistic Regression
- Linear SVM (`LinearSVC`)
- SGD Classifier
- Possibly Multinomial Naive Bayes

**Why TF-IDF first:** TF-IDF is interpretable, fast to train, and often
surprisingly competitive with more complex approaches on short text. It is also
directly covered in the course material.

**Deliverables:**
- `src/issue_intelligence/features.py` — TF-IDF pipeline
- `src/issue_intelligence/models/classical.py` — model training and prediction
- `notebooks/02_classical_models.ipynb` — training and comparison notebook

---

## Stage 4 — Proper Evaluation

**Goal:** Rigorously evaluate all models and understand where they fail.

Evaluation methods:
- Macro F1 (treats all classes equally — important with imbalanced data)
- Weighted F1
- Per-class precision and recall
- Confusion matrix
- Temporal train/test split (training on older issues, testing on newer)
- Leakage analysis (ensuring no future information leaks into training)
- Error analysis (examining misclassified issues manually)

**Course theory connection:**
This stage connects to generalisation, overfitting, and model selection
from the course curriculum.

**Deliverables:**
- `src/issue_intelligence/evaluation.py`
- `reports/stage4_evaluation.md`
- `notebooks/03_evaluation.ipynb`

---

## Stage 5 — Transformer-based Classification

**Goal:** Apply a pre-trained language model to improve classification quality.

Approach:
- Fine-tune or use a sentence transformer (e.g. `distilbert-base-uncased`)
- Compare against TF-IDF baselines from Stage 3

**Course theory connection:**
Model capacity, regularisation, and the bias-variance tradeoff.

**Deliverables:**
- `src/issue_intelligence/models/transformer.py`
- Updated evaluation report

---

## Stage 6 — Semantic Issue Retrieval

**Goal:** Given a new issue, retrieve the most semantically similar historical issues.

Approach:
- Compute issue embeddings using a sentence transformer
- Store embeddings in a vector index (e.g. FAISS or a simple cosine similarity search)
- Query index at inference time

**Deliverables:**
- `src/issue_intelligence/retrieval.py`
- `notebooks/04_retrieval.ipynb`

---

## Stage 7 — Duplicate Issue Detection

**Goal:** Detect when a newly submitted issue is likely a duplicate of an
existing open issue.

Approach:
- Use embedding similarity from Stage 6
- Threshold tuning using precision-recall curves

**Deliverables:**
- `src/issue_intelligence/duplicates.py`

---

## Stage 8 — FastAPI Backend

**Goal:** Expose the trained models through a REST API.

Endpoints (planned):
- `POST /classify` — predict issue type from title + description
- `GET /similar` — retrieve similar historical issues
- `GET /health` — health check

**Deliverables:**
- `src/issue_intelligence/api/` — FastAPI application
- `tests/test_api.py`
- API documentation (auto-generated by FastAPI)

---

## Stage 9 — Final Report and GitHub Presentation

**Goal:** Document the full project for the course submission and as a
portfolio piece.

Report sections:
- Problem statement and motivation
- Dataset description and audit findings
- Model comparison table
- Evaluation results with theory connections
- Limitations and future work
- AI tool usage log

**Course theory connections to include:**
- Generalisation and overfitting
- Model capacity
- Margin-based learning (SVM)
- Regularisation (L2 in Logistic Regression, etc.)
- Model selection
- Brief mention of PAC learning, VC Dimension, Rademacher Complexity
  where naturally applicable to the results

**Deliverables:**
- `reports/final_report.md` (or PDF)
- Polished `README.md`
- Cleaned `docs/gpt_usage.md`
