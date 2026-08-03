# GitHub Issue Intelligence

> An AI/ML system that analyses GitHub issues from real open-source repositories,
> classifies them by type, and retrieves semantically similar historical issues.

---

## Project Status

**Current Status: Stage 2C Complete, Ready for NLP Baselines**

Stages 0, 1A-1D, 2A-2C are ✅ Complete.
The simple baselines (Majority Class and Stratified Random) have been evaluated.
The performance floor is established (Temporal Macro F1: ~0.34).
Next task is Stage 3: Classical NLP Models (TF-IDF).

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
