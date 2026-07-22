# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stage:** Stage 0 — Project Initialisation
**Date completed:** 2026-07-22

The project skeleton has been created. The virtual environment is set up,
initial dependencies are installed, and the basic verification tests pass.

No data has been collected. No models have been trained.

---

## How to Activate the Environment

### Windows (PowerShell)

```powershell
# From the project root directory:
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this first (once):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS / Linux

```bash
source .venv/bin/activate
```

You should see `(.venv)` prepended to your terminal prompt.

---

## How to Run Tests

```bash
# From the project root, with the venv active:
pytest tests/ -v
```

Expected output:

```
tests/test_project_setup.py::test_package_importable PASSED
tests/test_project_setup.py::test_package_version PASSED
2 passed in ...s
```

---

## How to Run the Linter

```bash
ruff check src/ tests/
```

Expected output: no issues reported.

---

## Environment Details

| Item | Value |
|------|-------|
| Python version | 3.14.0 |
| Virtual environment | `.venv/` (not committed) |
| Dependencies file | `requirements.txt` |
| Package location | `src/issue_intelligence/` |

---

## Next Planned Task — Stage 1: Dataset Collection and Audit

The next task is to collect GitHub issues from the candidate repositories
and perform an audit to finalise the label scheme.

Before starting Stage 1:
1. Obtain a GitHub Personal Access Token (classic, with `public_repo` read scope).
2. Copy `.env.example` to `.env` and add your token.
3. Ask the AI assistant to begin Stage 1 data collection.

Stage 1 will produce:
- A script to download issues via the GitHub Issues API
- Raw data files saved to `data/raw/` (not committed)
- A Jupyter notebook (`notebooks/01_data_audit.ipynb`) with the audit
- An updated `docs/decisions.md` with the final label scheme decision

---

## Open Questions

1. **Which repositories to include?**
   scikit-learn is the primary candidate, but we should audit pandas and vLLM
   before deciding whether to combine them or use them separately.

2. **Which labels to keep?**
   This will be answered by the audit in Stage 1.

3. **Multi-label vs. single-label formulation?**
   Some issues have multiple labels. We will decide after seeing the data.

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/issue_intelligence/__init__.py` | Package root; defines `__version__` |
| `src/issue_intelligence/data/__init__.py` | Data subpackage (empty for now) |
| `tests/test_project_setup.py` | Basic setup verification tests |
| `docs/project_plan.md` | Full staged project plan |
| `docs/decisions.md` | Decision log (read before making architectural choices) |
| `docs/gpt_usage.md` | AI usage log (update after each AI-assisted stage) |
| `.env.example` | Template for secrets — copy to `.env` and fill in |
| `requirements.txt` | Python dependencies |
