# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stages:** Stages 0, 1A-1D, 2A-2C, 3A, 3B, 3C, 4A, 4B
**Date completed:** 2026-08-04

Best classical candidate: LinearSVC (C=0.3, balanced, bigrams, min_df=5).
Temporal val Macro F1: 0.9087. Vocabulary: 20,473 terms (70% smaller than untuned).
Stage 4B (Final Evaluation) complete. Test Macro F1: 0.9300. BERT-Tiny (0.8550) did not beat LinearSVC (0.9087) so only LinearSVC advances.

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

### 3. Generate Splits (Stage 2B)
```powershell
.venv\Scripts\python scripts\create_splits.py --input data\processed\scikit-learn_issues_model.jsonl --out-dir data\processed\splits
```

### 4. Run Baselines (Stage 2C)
```powershell
.venv\Scripts\python scripts\run_baselines.py --splits-dir data\processed\splits --results-dir reports\results --figures-dir reports\figures\baselines
```

---

## Exact Next Task

**Stage 5 — (Upcoming Feature / API).**

Tell the assistant to proceed to the next stage.
