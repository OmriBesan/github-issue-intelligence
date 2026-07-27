# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stage:** Stage 1B — Dataset Audit
**Date completed:** 2026-07-27

Stages 0, 1A, and 1B are all complete.
- A sample of 500 scikit-learn issues has been collected (`data/raw/`).
- A full audit notebook has been created and executed.
- Six figures have been generated and saved to `reports/figures/`.
- Provisional label scheme and next-step recommendations are documented.

---

## How to Activate the Environment

### Windows (PowerShell)

```powershell
# From the project root directory:
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## How to Run Tests

```bash
pytest tests/ -v
```

Expected: **50 passed** (34 audit tests + 16 collector tests).

## How to Run the Linter

```bash
ruff check src/ scripts/ tests/
```

Expected: no issues reported.

---

## How to Re-run the Audit Script

```powershell
.venv\Scripts\python scripts\audit_dataset.py `
    --input   data\raw\scikit-learn_issues_sample.json `
    --metadata data\raw\scikit-learn_issues_sample_metadata.json `
    --figures-dir reports\figures
```

This regenerates all six figures and prints a text summary.

## How to Re-execute the Notebook

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --inplace --execute `
    notebooks\01_data_audit.ipynb `
    --ExecutePreprocessor.kernel_name=github-issue-intelligence `
    --ExecutePreprocessor.timeout=300
```

To view the notebook interactively:

```powershell
.venv\Scripts\jupyter notebook notebooks\01_data_audit.ipynb
```

---

## Stage 1B Audit Results (2026-07-27)

### Overview

| Metric | Value |
|--------|-------|
| Total issues | 500 |
| Unique label names | 59 |
| With at least one label | 462 (92.4%) |
| Without labels | 38 (7.6%) |
| With multiple labels | 228 (45.6%) |
| Missing/empty body | 1 (0.2%) |
| Missing/empty title | 0 |
| Open issues | 167 (33.4%) |
| Closed issues | 333 (66.6%) |

### Text Length (combined title + body)

| Stat | Value |
|------|-------|
| Min | 13 chars |
| Median | 1,766 chars |
| Mean | 2,497 chars |
| 95th percentile | 6,631 chars |
| Max | 56,563 chars |

### Leakage Check

| Check | Count |
|-------|-------|
| Titles with category prefix ([BUG], ENH:, …) | 120 (24.0%) |
| Titles containing a type label name literally | 79 (15.8%) |

### Provisional Type Subset (5 labels, single-type issues only)

| Metric | Value |
|--------|-------|
| Usable issues | 326 (65.2% of total) |
| Excluded — no type label | 161 |
| Excluded — multiple type labels | 13 |

| Class | Count | % of subset |
|-------|-------|-------------|
| Bug | 156 | 47.9% |
| Documentation | 64 | 19.6% |
| New Feature | 55 | 16.9% |
| RFC | 26 | 8.0% |
| Build / CI | 25 | 7.7% |

---

## Files Generated in Stage 1B

| File | Description | Committed? |
|------|-------------|-----------|
| `src/issue_intelligence/data/audit.py` | Reusable audit functions | Yes |
| `scripts/audit_dataset.py` | CLI audit script | Yes |
| `tests/test_audit.py` | 34 unit tests | Yes |
| `notebooks/01_data_audit.ipynb` | Executed audit notebook | Yes |
| `reports/figures/top_label_frequencies.png` | Top-30 labels bar chart | Yes |
| `reports/figures/labels_per_issue_distribution.png` | Labels-per-issue histogram | Yes |
| `reports/figures/type_label_distribution.png` | Type class bar chart | Yes |
| `reports/figures/creation_dates_by_month.png` | Monthly issue counts | Yes |
| `reports/figures/text_length_distribution.png` | Text length histograms | Yes |
| `reports/figures/label_cooccurrence_heatmap.png` | Co-occurrence heatmap | Yes |

Data files (gitignored, must be present locally):
- `data/raw/scikit-learn_issues_sample.json`
- `data/raw/scikit-learn_issues_sample_metadata.json`

---

## Key Audit Findings

1. **Class imbalance is severe** — Bug dominates at 47.9%.
   Build / CI has only 25 examples (7.7%).
   A model cannot be reliably trained on 25 examples of one class.

2. **24% leakage risk** — Many issue titles contain category prefixes.
   Prefixes must be stripped in preprocessing and leakage measured explicitly.

3. **Two distinct label roles** — Type labels and component labels must be
   separated.  We should design two classification tasks.

4. **Sample covers only 10 months** — Recency bias is a real concern.
   The full history of scikit-learn goes back to 2010.

5. **326 usable single-type issues** — This is enough for a prototype model
   but not for a reliable, balanced classifier.

---

## Next Planned Task — Stage 1C: Larger Collection

Before preprocessing or modelling, collect a larger and more historically
representative sample.

Tell the assistant:
> **"Begin Stage 1C: re-collect scikit-learn issues to obtain at least
> 3,000 issues spanning the full repository history."**

Criteria for the new collection:
- At least 100 single-type examples in each of the five classes.
- Issues spanning at least 3 years of history.
- Same collector and output format as Stage 1A.

---

## Git History

| Commit | Message |
|--------|---------|
| `545c2bf` | feat: add GitHub issue collection pipeline (Stage 1A) |
| `cffa512` | chore: initialize project structure |

(Stage 1B commit will appear here after the commit is made.)

---

## Environment Details

| Item | Value |
|------|-------|
| Python version | 3.14.0 |
| Virtual environment | `.venv/` (not committed) |
| Jupyter kernel | `github-issue-intelligence` |
| Dependencies file | `requirements.txt` |
| Lock file | `requirements-lock.txt` |
| Git repository | Local only (no remote) |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/issue_intelligence/data/github_client.py` | GitHub API collector |
| `src/issue_intelligence/data/audit.py` | Reusable audit functions |
| `scripts/collect_issues.py` | Data collection CLI |
| `scripts/audit_dataset.py` | Audit CLI (generates figures + summary) |
| `notebooks/01_data_audit.ipynb` | Full audit notebook (executed) |
| `tests/test_github_client.py` | Collector unit tests (16 tests) |
| `tests/test_audit.py` | Audit unit tests (34 tests) |
| `reports/figures/` | All generated audit figures |
| `docs/project_plan.md` | Full staged project plan |
| `docs/decisions.md` | Decision log — decisions 001–011 |
| `docs/gpt_usage.md` | AI usage log |
| `.env.example` | Secrets template — copy to `.env` |
