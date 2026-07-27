# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stage:** Stage 1C — Historical Collection and Label Review
**Date completed:** 2026-07-27

Stages 0, 1A, 1B, and 1C are all complete.

- A 500-issue recent sample (2025–2026) exists in `data/raw/scikit-learn_issues_sample.json`.
- A 5,000-issue historical sample (2010–2018) exists in `data/raw/scikit-learn_issues_history.json`.
- A full historical audit has been run on the 5,000-issue dataset.
- Eight figures are saved to `reports/figures/history/`.
- A three-scheme label comparison has been performed.
- Five new design decisions (012–016) are recorded.

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

Expected: **69 passed** (45 audit tests + 22 collector tests + 2 setup tests).

## How to Run the Linter

```bash
ruff check src/ scripts/ tests/
```

Expected: no issues reported.

---

## How to Re-run the Historical Audit

```powershell
.venv\Scripts\python scripts\audit_dataset.py `
    --input    data\raw\scikit-learn_issues_history.json `
    --metadata data\raw\scikit-learn_issues_history_metadata.json `
    --figures-dir reports\figures\history
```

This regenerates all eight figures and prints the full text summary including
the three-scheme label comparison.

## How to Re-run the Recent Sample Audit

```powershell
.venv\Scripts\python scripts\audit_dataset.py `
    --input    data\raw\scikit-learn_issues_sample.json `
    --metadata data\raw\scikit-learn_issues_sample_metadata.json `
    --figures-dir reports\figures
```

---

## Collection Commands

### Recent sample (Stage 1A — newest 500 issues)

```powershell
.venv\Scripts\python scripts\collect_issues.py `
    --owner scikit-learn --repo scikit-learn `
    --state all --max-issues 500 `
    --output data\raw\scikit-learn_issues_sample.json
```

### Historical sample (Stage 1C — oldest 5000 issues)

```powershell
.venv\Scripts\python scripts\collect_issues.py `
    --owner scikit-learn --repo scikit-learn `
    --state all --sort created --direction asc --max-issues 5000 `
    --output data\raw\scikit-learn_issues_history.json
```

---

## Stage 1C Collection Results (2026-07-27)

| Metric | Value |
|--------|-------|
| API items inspected | 11,087 |
| Pull requests excluded | 6,087 (54.9%) |
| Duplicates skipped | 0 |
| Regular issues saved | 5,000 |
| Rate limit remaining | 4,889 |
| Earliest issue date | 2010-08-31 |
| Latest issue date | 2018-05-28 |
| Years covered | 9 (2010–2018) |

### Issues per Year

| Year | Count |
|------|-------|
| 2010 | 13 |
| 2011 | 174 |
| 2012 | 470 |
| 2013 | 519 |
| 2014 | 508 |
| 2015 | 926 |
| 2016 | 936 |
| 2017 | 1,080 |
| 2018 | 374 (partial — through May only) |

> [!NOTE]
> The 2018 data is partial. Coverage ends at May 2018 because the 5,000-issue
> limit was reached. The gap between 2018 and the recent sample (Sep 2025)
> represents ~7 years of uncollected data.

---

## Stage 1C Audit Results

| Metric | Value |
|--------|-------|
| Total issues | 5,000 |
| With labels | 2,504 (50.1%) |
| Without labels | 2,496 (49.9%) |
| Multi-label | 1,429 (28.6%) |
| Missing body | 40 (0.8%) |
| Unique label names | 69 |
| Leakage — prefix titles | 266 (5.3%) |
| Leakage — label in title | 370 (7.4%) |

> [!NOTE]
> The 49.9% unlabelled rate is much higher than the recent sample (7.6%).
> This reflects scikit-learn's older issue-tracking practices — labelling
> was not consistently applied before approx. 2015.

---

## Critical Label Finding: "Enhancement" Was Missed

The **fourth most frequent label** in the historical dataset is **"Enhancement"
(456 occurrences)**.  This label was **not** included in the provisional 5-class
scheme, which only listed "New Feature" (210 occurrences).

The label scheme evolved over time:
- 2010–2017: "Enhancement" was the primary feature-request label
- 2017–present: "New Feature" (and "RFC") replaced "Enhancement"

**Any future label scheme must include "Enhancement" in the feature class.**

---

## Label Scheme Comparison

Three schemes were evaluated on the 5,000-issue historical dataset:

| Scheme | Classes | Usable Issues | Min Class | Imbalance Ratio | Verdict |
|--------|---------|--------------|-----------|-----------------|---------|
| A (5-class original) | Bug, Documentation, New Feature, RFC, Build/CI | 1,524 | 3 (RFC) | 252:1 | **Rejected** |
| B (4-class, RFC merged) | Bug, Documentation, Enhancement, Build/CI | 1,524 | 59 (Build/CI) | 12.8:1 | Borderline |
| C (3-class core) | Bug, Documentation, Enhancement | 1,525 | 269 (Enhancement) | 2.8:1 | **Recommended** |

> [!IMPORTANT]
> None of these schemes include the "Enhancement" label from older issues.
> Adding it to Scheme C would increase the Enhancement count from 269 to ~725,
> reducing the imbalance ratio to approximately 1.5:1.

### Recommended Label Mapping for Stage 2

```python
LABEL_SCHEME = {
    "Bug":           ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement":   ["Enhancement", "New Feature", "RFC", "Build / CI"],
}
```

---

## Files Generated in Stage 1C

### Code Changes (committed)

| File | Change |
|------|--------|
| `src/issue_intelligence/data/github_client.py` | Added sort, direction, deduplication |
| `src/issue_intelligence/data/audit.py` | Added 3 new analysis functions |
| `scripts/collect_issues.py` | Added --sort, --direction CLI args |
| `scripts/audit_dataset.py` | Added 2 new charts + scheme comparison |
| `tests/test_github_client.py` | Added 7 new tests (total: 22) |
| `tests/test_audit.py` | Added 15 new tests (total: 45) |
| `docs/decisions.md` | Added Decisions 012–016 |
| `docs/gpt_usage.md` | Added Entry 004 |
| `docs/handoff.md` | Rewritten for Stage 1C |

### Data Files (gitignored — must be present locally)

| File | Size |
|------|------|
| `data/raw/scikit-learn_issues_history.json` | ~13 MB |
| `data/raw/scikit-learn_issues_history_metadata.json` | ~400 bytes |
| `data/raw/scikit-learn_issues_sample.json` | ~1.4 MB |
| `data/raw/scikit-learn_issues_sample_metadata.json` | ~325 bytes |

### Figures (gitignored — regenerate with audit script)

Historical figures are in `reports/figures/history/`:

| File | Content |
|------|---------|
| `top_label_frequencies.png` | Top-30 labels bar chart |
| `labels_per_issue_distribution.png` | Label count histogram |
| `type_label_distribution.png` | 5-class provisional type counts |
| `creation_dates_by_month.png` | Monthly issue timeline |
| `text_length_distribution.png` | Title/body/combined length distributions |
| `label_cooccurrence_heatmap.png` | Top-20 label co-occurrence matrix |
| `issues_per_year.png` | **NEW** — Issues created per calendar year |
| `type_labels_by_year.png` | **NEW** — Stacked type-label chart by year |

---

## Decisions Summary

| Decision | Summary |
|----------|---------|
| 012 | Collect oldest-first using direction=asc |
| 013 | "Enhancement" is a separate, important historical label |
| 014 | RFC retired — only 3 historical examples |
| 015 | Build/CI borderline — needs review after combined dataset |
| 016 | Scheme C (3-class) recommended for Stage 2 |

---

## Next Planned Task — Stage 2: Preprocessing

Stage 1 (Data Collection and Audit) is now complete.

The next task is **Stage 2: Text Preprocessing**.

Tell the assistant:
> **"Begin Stage 2: preprocess the historical issue dataset for machine learning.
> Use the recommended 3-class label scheme (Bug, Documentation, Enhancement)
> with the label mapping from Decision 016."**

Key requirements for Stage 2:
1. Strip known category prefixes from titles (e.g. "[BUG]", "ENH:", "RFC:").
2. Truncate combined text to a sensible maximum length.
3. Apply the Scheme C label mapping, including "Enhancement" as a historical label.
4. Build a train/validation/test split with stratification.
5. Save the preprocessed dataset as a separate file (do not overwrite raw data).

---

## Git History

| Commit | Message |
|--------|---------|
| `515fba3` | feat: add scikit-learn dataset audit (Stage 1B) |
| `545c2bf` | feat: add GitHub issue collection pipeline (Stage 1A) |
| `cffa512` | chore: initialize project structure |

(Stage 1C commit will appear here after commit is made.)

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
