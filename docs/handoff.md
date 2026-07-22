# Handoff Guide — GitHub Issue Intelligence

This document is written so that either project partner can pick up the
work at any point and know exactly where to start.

---

## Current State

**Completed stage:** Stage 1A — GitHub Issue Collection Pipeline
**Date completed:** 2026-07-22

Stage 0 (project initialisation) and Stage 1A (data collection) are both
complete. A sample of 500 scikit-learn issues has been collected and saved
locally. The raw data files are not committed (gitignored).

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
pytest tests/ -v
```

Expected output:

```
tests/test_github_client.py::TestParseNextPageUrl::test_returns_next_url_when_present  PASSED
tests/test_github_client.py::TestParseNextPageUrl::test_returns_none_when_no_next      PASSED
tests/test_github_client.py::TestParseNextPageUrl::test_returns_none_for_empty_header  PASSED
tests/test_github_client.py::TestExtractIssueFields::test_labels_simplified_to_name_list PASSED
tests/test_github_client.py::TestExtractIssueFields::test_user_type_extracted          PASSED
tests/test_github_client.py::TestExtractIssueFields::test_user_type_none_when_missing  PASSED
tests/test_github_client.py::TestExtractIssueFields::test_pull_request_field_not_in_output PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_pull_requests_excluded     PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_regular_issues_retained    PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_pagination_stops_at_max    PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_missing_token_raises_error PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_http_error_raises_github_api_error PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_saved_json_contains_expected_records PASSED
tests/test_github_client.py::TestGitHubIssueCollector::test_metadata_file_saved        PASSED
tests/test_project_setup.py::test_package_importable                                   PASSED
tests/test_project_setup.py::test_package_version                                      PASSED
16 passed
```

---

## How to Run the Linter

```bash
ruff check src/ scripts/ tests/
```

Expected output: no issues reported.

---

## How to Re-run Data Collection

If you need to refresh the raw data:

```powershell
.venv\Scripts\python scripts\collect_issues.py `
    --owner scikit-learn `
    --repo  scikit-learn `
    --state all `
    --max-issues 500 `
    --output data\raw\scikit-learn_issues_sample.json
```

Requires `GITHUB_TOKEN` in `.env`.

---

## Output File Locations

| File | Description | Committed? |
|------|-------------|-----------|
| `data/raw/scikit-learn_issues_sample.json` | 500 collected issues | No (gitignored) |
| `data/raw/scikit-learn_issues_sample_metadata.json` | Collection statistics | No (gitignored) |

---

## Stage 1A Collection Results (2026-07-22)

| Metric | Value |
|--------|-------|
| API items inspected | 2322 |
| Pull requests excluded | 1822 (78.5%) |
| Regular issues saved | 500 |
| Issues with labels | 462 (92.4%) |
| Issues without labels | 38 (7.6%) |
| Missing/empty body | 1 (0.2%) |
| Earliest issue date | 2025-09-10 |
| Latest issue date | 2026-07-22 |
| Rate limit remaining | 4976 |

---

## Known Limitations

1. **500 issues is a small sample.** scikit-learn has thousands of issues.
   We collected only the most recent 500 (state=all, newest first). The audit
   will reveal whether this is sufficient for modelling.

2. **Date range is limited.** All 500 issues are from approximately the last
   10 months. This may not represent the full label history.

3. **Label distribution is preliminary.** The top labels (Bug, Needs Triage,
   Documentation, New Feature) are promising, but the final label scheme will
   be decided after the Stage 1B audit.

4. **Windows terminal encoding.** The `✅` emoji was replaced with `[OK]` in
   the CLI script because Windows PowerShell defaults to cp1252 encoding.

---

## Environment Details

| Item | Value |
|------|-------|
| Python version | 3.14.0 |
| Virtual environment | `.venv/` (not committed) |
| Dependencies file | `requirements.txt` |
| Lock file | `requirements-lock.txt` |
| Package location | `src/issue_intelligence/` |
| Git repository | Initialised locally (no remote) |
| Stage 0 commit | `cffa512` — "chore: initialize project structure" |

---

## Next Planned Task — Stage 1B: Dataset Audit

Stage 1B creates the audit notebook that analyses the 500 collected issues.

Before starting Stage 1B:
1. Confirm the raw data files exist:
   - `data/raw/scikit-learn_issues_sample.json`
   - `data/raw/scikit-learn_issues_sample_metadata.json`
2. Ask the AI assistant: **"Begin Stage 1B: create the dataset audit notebook."**

Stage 1B will produce:
- `notebooks/01_data_audit.ipynb` — full exploratory analysis
- Updated `docs/decisions.md` — final label scheme decision
- Charts in `reports/figures/`

---

## Open Questions

1. **Is 500 issues enough?** The audit may show that some label classes have
   too few examples for reliable modelling. We may need to collect more.

2. **Which labels to keep?** Labels like "Needs Triage", "Array API",
   "Callbacks", and "Sprint" are not issue type categories. The audit must
   decide which labels map to meaningful prediction classes.

3. **Multi-label vs. single-label?** Some issues may have multiple type
   labels simultaneously. We will decide after the audit.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/issue_intelligence/data/github_client.py` | Reusable GitHub API client |
| `scripts/collect_issues.py` | CLI wrapper for data collection |
| `tests/test_github_client.py` | Unit tests (no real network calls) |
| `tests/test_project_setup.py` | Package import verification |
| `docs/project_plan.md` | Full staged project plan |
| `docs/decisions.md` | Decision log (read before making design choices) |
| `docs/gpt_usage.md` | AI usage log (update after each AI-assisted stage) |
| `.env.example` | Template for secrets — copy to `.env` and fill in |
| `requirements.txt` | Human-readable dependencies |
| `requirements-lock.txt` | Exact installed versions (pip freeze output) |
