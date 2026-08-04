# GitHub Issue Intelligence

> An AI/ML system that analyses GitHub issues from real open-source repositories,
> classifies them by type, and retrieves semantically similar historical issues.

---

## Project Status

**Current Status: Stage 5A Complete — FastAPI Inference Service Built**

Stages 0, 1A-1D, 2A-2C, 3A, 3B, 3C, 4A, 4B, and 5A are ✅ Complete.
BERT-Tiny (4.4M params) Temporal Macro F1: **0.8550** — 5.4 pp below tuned LinearSVC (0.9087).
LinearSVC advances to final test-set evaluation. Transformer does not.
Next: Stage 5B — Optional UI or Semantic Retrieval pipeline.
**Note:** Evaluated once on a held-out temporal test set (2024-2026). No post-test tuning occurred.

---

## Motivation

GitHub repositories accumulate thousands of issues. Manually triaging each
issue — deciding whether it is a bug, a feature request, or a documentation
problem — costs maintainer time and slows down project management.

This project explores whether ML models trained on historical, human-labelled
issues can automate or assist with that triage process reliably enough to be
useful in practice.

---

## Planned Phases

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Project initialisation — structure, environment, dependencies | ✅ Done |
| 1 | Dataset collection and audit (GitHub Issues API) | ⬜ Next |
| 2 | Simple baselines (majority-class, dummy classifier) | ⬜ Planned |
| 3 | Classical NLP models (TF-IDF + Logistic Regression, SVM, SGD) | ⬜ Planned |
| 4 | Proper evaluation (macro F1, confusion matrix, temporal split) | ⬜ Planned |
| 5 | Transformer-based text classification | ⬜ Planned |
| 6 | Semantic issue retrieval using embeddings | ⬜ Planned |
| 7 | Duplicate issue detection | ⬜ Planned |
| 8 | FastAPI backend and optional UI | ⬜ Planned |
| 9 | Final report and GitHub presentation | ⬜ Planned |

> **Note:** The table above may change after the dataset audit in Stage 1.
> Label classes, repository selection, and modelling approach will all be
> decided based on what the data actually looks like.

---

## Repository Structure

```
github-issue-intelligence/
├── .agents/rules/          # Project development rules for AI coding assistants
├── data/
│   ├── raw/                # Downloaded issues (not committed)
│   ├── interim/            # Cleaned / partially processed data
│   └── processed/          # Final model-ready datasets
├── docs/                   # Project documentation and decision log
├── notebooks/              # Jupyter notebooks for exploration and reporting
├── reports/figures/        # Generated charts and figures
├── scripts/                # One-off helper scripts (data download, etc.)
├── src/issue_intelligence/ # Reusable Python package (all core logic lives here)
├── tests/                  # pytest test suite
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.12 or later (this project uses Python 3.14)
- `git`

### 2. Clone the repository

```bash
git clone <repository-url>
cd github-issue-intelligence
```

### 3. Create and activate the virtual environment

**Windows (PowerShell):**
```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GitHub Personal Access Token
```

### 6. Verify the setup

```bash
pytest tests/ -v
ruff check src/ tests/
```

---

## API Usage

A FastAPI service is available to run inferences against the final tuned LinearSVC model.

### Starting the Server
```powershell
.venv\Scripts\python -m uvicorn issue_intelligence.api.app:app --reload
```

Optionally set the model path via environment variable:
```powershell
$env:ISSUE_MODEL_PATH="models/classical/final_linear_svc.joblib"
```
If the model file is missing, the API will start but will report not-ready and return 503s for predictions. You must run Stage 3A/3C scripts to build a model, or provide a saved pipeline.

### Endpoints

- `GET /health` : Returns readiness status.
- `GET /model-info` : Returns model labels, expected inputs, and configuration.
- `POST /predict` : Submits an issue title and body.

**Example Request:**
```json
{
  "title": "RandomForestClassifier raises an error when fitting sparse input",
  "body": "Calling fit with a sparse matrix produces an unexpected ValueError."
}
```

**Example Response:**
```json
{
  "predicted_label": "Bug",
  "decision_scores": {
    "Bug": 0.5218,
    "Documentation": -0.8881,
    "Enhancement": -0.6790
  },
  "decision_margin": 1.2008,
  "model_name": "LinearSVC"
}
```

> **Note:** The `decision_scores` are raw LinearSVC decision scores. They are NOT probabilities or calibrated confidence metrics. `decision_margin` represents the gap between the top prediction and the runner-up.

- `POST /similar` : Retrieves similar historical issues using lexical TF-IDF cosine similarity.

**Example Request:**
```json
{
  "title": "RandomForestClassifier raises an error when fitting sparse input",
  "body": "Calling fit with a sparse matrix produces an unexpected ValueError.",
  "top_k": 5
}
```

**Example Response:**
```json
{
  "results": [
    {
      "issue_id": 14613,
      "title": "EllipticEnvelope does not work with a sparse matrix",
      "target_label": "Documentation",
      "created_at": "2019-08-09T13:17:27Z",
      "url": "https://github.com/scikit-learn/scikit-learn/issues/14613",
      "similarity_score": 0.21400409717343596
    }
  ],
  "retrieval_method": "tfidf_cosine",
  "indexed_issue_count": 4854
}
```

> **Note:** Similarity scores are purely lexical distance metrics, not probabilities or proof of duplicate issues.

## Configuration

All secrets and environment-specific settings are stored in `.env` (not
committed). See `.env.example` for the full list of variables.

---

## Documentation

| File | Purpose |
|------|---------|
| `docs/project_plan.md` | Detailed stage-by-stage plan |
| `docs/decisions.md` | Architecture and design decision log |
| `docs/gpt_usage.md` | Record of AI tool assistance |
| `docs/handoff.md` | Handoff guide for resuming work |

---

## Data Collection

Issues are collected from GitHub using the `scripts/collect_issues.py` script.
Pull requests are excluded automatically.

```powershell
# Collect up to 500 issues from scikit-learn (open + closed)
.venv\Scripts\python scripts\collect_issues.py `
    --owner scikit-learn `
    --repo  scikit-learn `
    --state all `
    --max-issues 500 `
    --output data\raw\scikit-learn_issues_sample.json
```

Output files (not committed — see `.gitignore`):
- `data/raw/scikit-learn_issues_sample.json` — collected issue records
- `data/raw/scikit-learn_issues_sample_metadata.json` — collection statistics

Requires `GITHUB_TOKEN` to be set in `.env`.

---

## Contributing

This is a university course project with two contributors. Changes are made
in small, reviewable increments. See `docs/handoff.md` for context before
starting any new work session.
