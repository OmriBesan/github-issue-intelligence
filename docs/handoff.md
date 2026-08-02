# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stages:** Stages 0, 1A, 1B, 1C, 1D, and 2A
**Date completed:** 2026-08-02

The cleaned dataset has been successfully prepared in Stage 2A.
The next step is Stage 2B: creating the temporal and stratified splits.

---

## Project Purpose
An AI/ML system that analyses GitHub issues from real open-source repositories,
classifies them by type, and retrieves semantically similar historical issues.

---

## Final Label Mapping
The final three-class target scheme for the dataset is:
```python
LABEL_SCHEME = {
    "Bug":           ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement":   ["Enhancement", "New Feature", "RFC"],
}
```
**Build / CI is explicitly excluded** from the type-prediction target as it is a component modifier.

---

## Dataset Results

- **12,190** collected unique issues
- **5,710** usable processed issues

**Processed Class Distribution:**
- Bug: 2,274
- Enhancement: 2,039
- Documentation: 1,397

---

## Environment Setup Commands

### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
source .venv/bin/activate
```

---

## How to Verify Current State

**Run Tests (Expected: 171 tests passing)**
```bash
pytest tests/ -v
```

**Run Linter (Expected: no issues reported)**
```bash
ruff check src/ scripts/ tests/
```

---

## Commands to Recreate Datasets

**Note: `.env` and generated datasets are NOT committed to version control.**

### 1. Collect and Combine Raw Datasets
*Requires `GITHUB_TOKEN` in `.env`.*
```powershell
# Create sample
.venv\Scripts\python scripts\collect_issues.py --owner scikit-learn --repo scikit-learn --state all --max-issues 500 --output data\raw\scikit-learn_issues_sample.json

# Create history
.venv\Scripts\python scripts\collect_issues.py --owner scikit-learn --repo scikit-learn --state all --sort created --direction asc --max-issues 5000 --output data\raw\scikit-learn_issues_history.json

# Create additional gap fill
.venv\Scripts\python scripts\collect_issues.py --owner scikit-learn --repo scikit-learn --state all --sort created --direction asc --since 2018-05-29T00:00:00Z --max-issues 12000 --output data\raw\scikit-learn_issues_additional.json

# Combine
.venv\Scripts\python scripts\combine_datasets.py --inputs data\raw\scikit-learn_issues_history.json data\raw\scikit-learn_issues_additional.json data\raw\scikit-learn_issues_sample.json --output data\raw\scikit-learn_issues_combined.json --min-year 2011
```

### 2. Generate Cleaned Modelling Dataset (Stage 2A)
```powershell
.venv\Scripts\python scripts\prepare_dataset.py --input data\raw\scikit-learn_issues_combined.json --out-dir data\processed --sample 3 --seed 42
```
This produces `data/processed/scikit-learn_issues_model.jsonl` (and `.csv`).

---

## Stage 2B Work Status

There is currently **no partial Stage 2B work** committed. Stage 2A was confirmed fully completed.

---

## Exact Next Task

**Stage 2B — temporal and secondary stratified train/validation/test splits.**

> [!WARNING]
> Do NOT train models before Stage 2B is completely implemented and reviewed.

Tell the assistant:
> "Begin Stage 2B only: create reproducible train, validation, and test splits.
> Use a temporal split as the primary evaluation strategy (~70% train / 15% validation / 15% test).
> Also create a secondary stratified random split for comparison only.
> Read the project constraints and rules carefully."
