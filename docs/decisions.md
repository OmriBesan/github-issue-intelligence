# Decision Log — GitHub Issue Intelligence

This file records key architectural and design decisions made during the
project. Each entry explains what was decided, why, and any alternatives
that were considered.

---

## Decision 001 — Audit repositories before selecting the final dataset

**Date:** 2026-07-22
**Stage:** 0 (initialisation)
**Status:** Accepted

### Context

We identified three candidate repositories for our dataset:
1. **scikit-learn** — a mature, well-labelled ML library with thousands of issues
2. **pandas** — a popular data manipulation library with a large issue history
3. **vLLM** — a newer AI-focused project with domain-specific issue types

Before committing to any repository, label scheme, or problem formulation, we
need to understand what the data actually looks like.

### Decision

We will collect issues from the candidate repositories first and perform a
thorough audit before deciding:
- Which repository (or combination) to use
- Which labels to keep, merge, or discard
- How to handle class imbalance
- Whether we can reliably predict component or only category

### Rationale

- Label quality varies enormously across repositories and over time.
- Some repositories use labels inconsistently; others have too few labelled issues.
- Class imbalance can make evaluation misleading if we do not account for it upfront.
- Choosing the dataset without auditing would risk building a system on
  unreliable or poorly-distributed data.

### Alternatives considered

- **Commit to scikit-learn immediately:** Simpler to start, but risks discovering
  late that the labels are noisy or imbalanced in unexpected ways.
- **Use a pre-built dataset from Kaggle or HuggingFace:** Faster, but defeats the
  purpose of demonstrating real-world data collection skills for the course.

### Consequences

- Stage 1 must be completed before any model training begins.
- The label set will only be finalised after the audit.
- The `README.md` includes a note that the design may change after the audit.

---

## Decision 002 — Use Python 3.14 for the virtual environment

**Date:** 2026-07-22
**Stage:** 0 (initialisation)
**Status:** Accepted

### Context

The project specification requested Python 3.12. However, only Python 3.14.0
is installed on the development machine.

### Decision

Use Python 3.14.0 for the virtual environment. All planned dependencies
(requests, pandas, numpy, matplotlib, scikit-learn, jupyter, pytest, ruff)
have releases compatible with Python 3.14.

### Consequences

- If a collaborator installs Python 3.12, the environment should still work
  for all current dependencies. We will document the actual Python version
  in `docs/handoff.md`.
- If any dependency is found to be incompatible with Python 3.14, we will
  record the fix in this decision log.

---

## Decision 003 — Separate data collection from the audit notebook

**Date:** 2026-07-22
**Stage:** 1A (data collection)
**Status:** Accepted

### Context

We could have combined data collection and the audit into a single notebook.

### Decision

Collection lives in a standalone Python module (`github_client.py`) and a
CLI script (`collect_issues.py`). The audit notebook (Stage 1B) will import
the already-collected JSON files.

### Rationale

- Collection makes real network requests; notebooks are not the right place
  for these.
- Separating them means collection can be re-run independently if the raw
  data needs refreshing, without re-running all audit cells.
- It keeps reusable logic in `src/`, which is testable, importable, and
  version-controlled independently of the notebook.

---

## Decision 004 — Exclude pull requests at collection time

**Date:** 2026-07-22
**Stage:** 1A (data collection)
**Status:** Accepted

### Context

The GitHub Issues API returns both issues and pull requests in the same
endpoint. In our scikit-learn sample of 2322 API items, 1822 (78.5%) were
pull requests.

### Decision

Any API item that contains a `"pull_request"` key is immediately discarded
during collection. It is never written to disk.

### Rationale

- Pull requests are not GitHub issues in the semantic sense. Including them
  would corrupt the classification task.
- Filtering at collection time avoids polluting the raw JSON with records
  that would need to be filtered out in every downstream step.

---

## Decision 005 — Preserve raw data before any preprocessing

**Date:** 2026-07-22
**Stage:** 1A (data collection)
**Status:** Accepted

### Decision

The raw JSON from the API is saved to `data/raw/` exactly as received
(with only field selection and label simplification applied). No text
cleaning, tokenisation, or normalisation is applied at collection time.

### Rationale

- Preprocessing decisions (lowercasing, stopword removal, etc.) should be
  made after examining the data in the audit stage.
- Preserving raw text allows us to experiment with different preprocessing
  strategies without re-downloading from the API.
- The `data/raw/` directory is gitignored so large files are never committed.

---

## Decision 006 — Collect 500 issues as the first sample

**Date:** 2026-07-22
**Stage:** 1A (data collection)
**Status:** Accepted

### Context

scikit-learn has thousands of issues. We need a manageable first sample for
the audit without exhausting the GitHub API rate limit.

### Decision

500 regular issues (state=all, most recent first) is our initial sample.

### Rationale

- 500 issues is enough to observe label distributions, identify class
  imbalance, and assess data quality.
- It fits comfortably within one GitHub API rate-limit window (5000
  requests/hour for authenticated users).
- The collector can be re-run with a larger `--max-issues` value later if
  the audit shows we need more data.

- 2322 API items inspected to collect 500 regular issues.
- 1822 PRs excluded (78.5% of API items were PRs).
- 4976 rate-limit requests remaining after collection.

---

## Decision 007 — Keep reusable audit logic in `src/`, not in the notebook

**Date:** 2026-07-27
**Stage:** 1B (dataset audit)
**Status:** Accepted

### Decision

All analysis functions (label frequency, co-occurrence, leakage detection,
provisional subset construction, etc.) are implemented in
`src/issue_intelligence/data/audit.py`.

The notebook (`notebooks/01_data_audit.ipynb`) and CLI script
(`scripts/audit_dataset.py`) call these functions rather than reimplementing
the logic inline.

### Rationale

- Reusable module code is unit-testable; notebook cell code is not.
- If the dataset changes, only one code path needs updating.
- The notebook stays focused on explanation and visualisation.

---

## Decision 008 — Use five provisional type labels for Stage 1B

**Date:** 2026-07-27
**Stage:** 1B (dataset audit)
**Status:** Provisional — subject to revision after broader collection

### Context

The audit revealed 59 unique label names across 500 issues.  Most labels
serve workflow, component, or lifecycle roles and are not suitable as
classification targets.

### Decision

The five provisional issue-type labels are:
- **Bug**
- **Documentation**
- **New Feature**
- **RFC**
- **Build / CI**

### Rationale

- These five labels describe the *kind* of problem, not the triage state
  or the affected component.
- Together they cover 326 of the 500 issues (65.2%) when restricted to
  single-type issues.
- They appear consistently in the top 10 most frequent labels.

### Known limitations

- The sample is only 500 recent issues (10-month window).  The label
  distribution may not reflect the full repository history.
- Bug is heavily over-represented (47.9% of the provisional subset).
  Build / CI has only 25 examples (7.7%).
- The label scheme must be re-evaluated after collecting 3,000+ issues.

---

## Decision 0023 — Exclude 'Build / CI' from Target Prediction
**Date:** 2026-07-28
**Context:** 'Build / CI' co-occurs highly with 'Bug' and 'Enhancement'. It is a component modifier rather than a core issue type.
**Decision:** Drop 'Build / CI' completely from the classification target, resulting in a clean 3-class target: Bug, Documentation, Enhancement.

## Decision 024 — Temporal Cutoffs for Primary Evaluation Split
**Date:** 2026-08-03
**Context:** We need a 70/15/15 train/val/test split that preserves historical order without leaking issues from the same calendar day across sets.
**Decision:** We slice the dataset strictly by `created_at` timestamp.
- **Train (70.0%):** 2010-10-19 to 2022-10-12
- **Validation (15.0%):** 2022-10-13 to 2024-05-28
- **Test (15.0%):** 2024-05-29 to 2026-07-27
We also enforced a rule to push forward any exact-day overlap so that issues filed on the boundary day remain together in the earlier split.

## Decision 025 — Macro F1 for Model Evaluation
**Date:** 2026-08-03
**Context:** The class distribution is imbalanced (Bug ~40%, Enhancement ~36%, Documentation ~24%). We need a single metric to rank models that doesn't artificially reward ignoring the minority class.
**Decision:** We will use **Macro F1** as the primary evaluation metric because it computes F1 for each class independently and averages them equally. This forces the model to perform well across *all* classes, not just the majority class.


---

## Decision 009 — Separate issue-type from component prediction

**Date:** 2026-07-27
**Stage:** 1B (dataset audit)
**Status:** Accepted (plan for Stage 2+)

### Context

The audit identified two clearly distinct label roles in scikit-learn:
- **Type labels** (Bug, Documentation, …) — what kind of problem.
- **Component labels** (Array API, Callbacks, …) — where the problem is.

### Decision

Design two separate classification tasks rather than one combined task:
1. **Task 1 (priority):** Predict issue type (5-class classification).
2. **Task 2 (optional):** Predict affected component.

### Rationale

- Mixing type and component labels in a single classifier would require
  multi-label output, which adds significant complexity for a first model.
- The component label vocabulary is larger and changes as new sub-projects
  emerge (e.g. Array API, Callbacks, free-threading in the current sample).

---

## Decision 010 — Acknowledge and measure leakage risk before modelling

**Date:** 2026-07-27
**Stage:** 1B (dataset audit)
**Status:** Accepted

### Finding

The audit found that 120 out of 500 issue titles (24%) contain a category
prefix such as `[BUG]`, `ENH:`, or `[RFC]`.  An additional 79 titles contain
a type label name literally.

### Decision

1. Strip known prefixes from titles in the preprocessing step.
2. Evaluate the final model on both the full dataset and the prefix-free
   subset to quantify the leakage effect.
3. Document this risk in the model card.

---

## Decision 011 — Collect at least 3,000 historical issues before modelling

**Date:** 2026-07-27
**Stage:** 1B (dataset audit)
**Status:** Accepted (recommended next step)

### Context

The audit found that the smallest provisional class (Build / CI) has only
25 examples.  This is insufficient for reliable training.

### Decision

Before Stage 2 (preprocessing and modelling), re-run the collector with
`--max-issues 3000` or higher, targeting older issues (earliest first) to
reduce recency bias.

### Success criteria

- Each of the five type classes must have at least 100 single-type examples.
- The collection should span at least 3 years of repository history.

---

## Decision 012 — Collect historical issues oldest-first using sort=created direction=asc

**Date:** 2026-07-27
**Stage:** 1C (historical collection)
**Status:** Accepted

### Context

The Stage 1B sample (500 issues) covered only the 10 most-recent months
of scikit-learn history.  The provisional subset had severe class imbalance
(6.2:1 ratio) and the smallest class (Build / CI) had only 25 examples.

### Decision

Re-collect using `sort=created&direction=asc&state=all&max_issues=5000`.
This fetches the oldest 5,000 regular issues first, maximising historical
coverage.

### Result

- 5,000 regular issues collected
- 11,087 API items inspected
- 6,087 pull requests excluded
- 0 duplicates skipped
- Date range: 2010-08-31 to 2018-05-28 (9 calendar years, 7.7 years)
- Rate limit remaining: 4,889

---

## Decision 013 — "Enhancement" label is a distinct historical target, not equivalent to "New Feature"

**Date:** 2026-07-27
**Stage:** 1C (label review)
**Status:** Accepted — important correction to provisional scheme

### Context

In the historical dataset (2010–2018), the top label by frequency is "Bug"
(837) but the fourth-most frequent label is "Enhancement" (456).  The
provisional scheme included "New Feature" (210 occurrences) but NOT
"Enhancement".  The two labels co-exist: some issues have only "Enhancement",
others have only "New Feature", and some have both.

### Finding

The scikit-learn label scheme changed over time:
- 2010–approx.2017: "Enhancement" was the primary feature-request label
- 2017–present: "New Feature" and "RFC" have largely replaced "Enhancement"

If "Enhancement" is excluded from the target scheme, the feature-request
class covers only 206 out of approximately 660+ eligible issues.

### Decision

Any final label scheme must include "Enhancement" in the feature/enhancement
class.  An updated provisional scheme for Stage 2 will use:
  - Bug (or similar)
  - Documentation
  - Enhancement (combining "Enhancement" + "New Feature" + optionally "RFC")
  - Build / CI (if kept as separate class)

---

## Decision 014 — RFC label is unusable as an independent class

**Date:** 2026-07-27
**Stage:** 1C (label review)
**Status:** Accepted

### Context

The provisional five-class scheme included "RFC" as a separate class.

### Finding

In the historical dataset of 5,000 issues:
- RFC: only **3** single-type examples (essentially zero)

In the Stage 1B sample of 500 recent issues:
- RFC: 26 single-type examples — still below the 100-example threshold

Conclusion: "RFC" has never been consistently applied as a primary label in
scikit-learn's history.  It appears as a secondary label alongside "New
Feature" or "Enhancement" rather than as a standalone type.

### Decision

RFC will **not** be retained as an independent class.  It will be merged into
the Enhancement/New Feature class in all future label schemes.

---

## Decision 015 — Build / CI is borderline; final inclusion depends on merged dataset

**Date:** 2026-07-27
**Stage:** 1C (label review)
**Status:** Under review — revisit after Stage 2 data collection

### Context

In the historical dataset (5,000 issues):
- Build / CI: **59** single-type examples (below the 100-example threshold)

In the recent sample (Stage 1B):
- Build / CI: **25** single-type examples

Combined across both samples without deduplication: approx. 84 examples.
The threshold is 100.

### Finding

Build / CI is also arguably a component label (which part of the system is
affected) rather than an issue-type label (what kind of issue is it).
A "Build / CI" issue is typically also a "Bug" or "Enhancement" — it
describes the target system, not the nature of the problem.

### Decision

Build / CI will be held under review.  If the combined dataset (Stage 1D or
beyond) yields ≥ 100 single-type examples, it will be retained.  Otherwise
it will be merged into Bug or dropped.  This decision is deferred to Stage 2.

---

## Decision 016 — Recommended three label schemes for Stage 2 evaluation

**Date:** 2026-07-27
**Stage:** 1C (label review)
**Status:** Accepted — to be finalised in Stage 2

### Context

The Stage 1C audit evaluated three label schemes against the 5,000-issue
historical dataset.

### Scheme A — Original 5-class (Bug, Documentation, New Feature, RFC, Build/CI)
- Usable issues: 1,524
- Imbalance ratio: **252:1** (Bug=757, RFC=3) — completely unacceptable
- Verdict: **rejected**

### Scheme B — 4-class (Bug, Documentation, Enhancement, Build/CI)
where Enhancement = New Feature + RFC
- Usable issues: 1,524
- Imbalance ratio: **12.8:1** (Bug=757, Build/CI=59)
- Verdict: acceptable only if Build/CI reaches ≥ 100 examples in the full
  dataset; otherwise to be merged.
- Note: does NOT yet include the "Enhancement" label from older issues.

### Scheme C — 3-class core (Bug, Documentation, Enhancement)
where Enhancement = New Feature + RFC + Build/CI
- Usable issues: 1,525
- Imbalance ratio: **2.8:1** (Bug=757, Enhancement=269)
- Verdict: **recommended as baseline** — well-balanced, robust, and
  semantically clear.
- Note: still does NOT include "Enhancement" (old label) — adding it will
  further improve the Enhancement count to ~725, reducing the ratio.

### Decision

Scheme C (3-class) is the recommended starting point for Stage 2.  The
exact label mappings will be:
  - Bug: ["Bug"]
  - Documentation: ["Documentation"]
  - Enhancement: ["Enhancement", "New Feature", "RFC", "Build / CI"]

This scheme is the only one that achieves both:
1. ≥ 100 examples per class
2. Imbalance ratio < 3:1

---

## Decision 017 — GitHub API pagination limit requires using `since` filter instead of `start_page`

**Date:** 2026-07-27
**Stage:** 1D (historical coverage completion)
**Status:** Accepted

### Context

Stage 1C planned to resume collection using `start_page=112` (the page where
the historical collection ended at 11,087 API items).  On execution, the
GitHub API returned HTTP 422:

> "Pagination with the page parameter is not supported for large datasets,
> please use cursor based pagination (after/before)"

This is a known GitHub API restriction for repositories with large issue counts.

### Decision

Use the `since` query parameter instead of `start_page` for large repositories.
The `since` parameter filters issues by `updated_at >= since`, which effectively
skips old closed issues and returns the 2018-onwards dataset we need.

The `since` parameter was added to the collector and to the CLI:

```
--since 2018-05-29T00:00:00Z
```

Deduplication by issue ID ensures that any pre-2018 issues that happen to have
been updated after the `since` date are safely handled if we run the combiner.

---

## Decision 018 — `since` filter may include pre-cutoff issues updated after the cutoff date

**Date:** 2026-07-27
**Stage:** 1D
**Status:** Accepted — trade-off acknowledged

### Context

Using `since=2018-05-29T00:00:00Z` returns all issues with `updated_at >=
2018-05-29`. This includes:

1. Old issues (2010-2017 creation) that were updated after 2018 (e.g., still
   open, had new comments, or were linked to PRs).
2. Issues created after 2018-05-29 (the ones we want).

### Decision

Group 1 issues are already present in the historical collection. The combiner's
deduplication by GitHub issue ID will remove them.  No action needed.

Group 2 issues are exactly the gap we need to fill.

This approach is preferred over any cursor-based strategy because it requires
no changes to the core pagination architecture and produces correct results via
deduplication.

---

## Decision 019 — Combine three raw files into one deduplicated dataset

**Date:** 2026-07-27
**Stage:** 1D
**Status:** Accepted

### Files combined

1. `data/raw/scikit-learn_issues_history.json` — 5,000 issues (2010-2018)
2. `data/raw/scikit-learn_issues_additional.json` — ~8,000-12,000 issues
   (primarily 2018-2025, some pre-2018 overlap handled by deduplication)
3. `data/raw/scikit-learn_issues_sample.json` — 500 issues (2025-2026)

### Decision

A dedicated `combine_datasets.py` script:
1. Loads files in the order above
2. Deduplicates by GitHub issue ID (first occurrence wins)
3. Sorts by created_at ascending
4. Saves the combined issues and metadata to:
   - `data/raw/scikit-learn_issues_combined.json`
   - `data/raw/scikit-learn_issues_combined_metadata.json`

---

## Decision 020 — Build / CI excluded from type target in final candidate scheme

**Date:** 2026-07-27
**Stage:** 1D (label mapping review)
**Status:** Accepted

### Context

Decision 015 held Build / CI under review.  The Stage 1D label review will
provide automated co-occurrence statistics and a 40-issue human-review sample
to inform the final recommendation.

### Candidate scheme under evaluation

```python
{
    "Bug":           ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement":   ["Enhancement", "New Feature", "RFC"],
}
```

Build / CI is excluded from this candidate scheme because:
1. It is semantically a component identifier (CI infrastructure), not an issue
   type.  A "Build / CI" issue is almost always also a Bug or Enhancement.
2. It has historically low counts relative to the other classes.
3. Including it produces a large class imbalance.

The label review confirmed this decision. Build / CI is permanently excluded
from the issue-type prediction target.

---

## Decision 021 — Final label mapping adopted for the modelling dataset

**Date:** 2026-07-28
**Stage:** 2A — Dataset Preparation
**Status:** Accepted and implemented

### Decision

The following label mapping is the definitive three-class target for all
subsequent modelling work:

```python
LABEL_SCHEME = {
    "Bug":           ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement":   ["Enhancement", "New Feature", "RFC"],
}
```

**Evidence supporting the mapping:**
1. Historical label evolution analysis (Stage 1B-1D) confirmed that
   "Enhancement" (pre-2019) and "New Feature" (post-2019) describe the
   same issue type.
2. "RFC" (Request for Comments) is the formal process for proposing new
   features and belongs in the same Enhancement class.
3. The three raw labels share very low pairwise overlap (5 issues for
   Enhancement + New Feature, 2 for Enhancement + RFC, 12 for New Feature +
   RFC out of 5,710 usable issues).
4. The candidate scheme achieves a 1.63:1 imbalance ratio (max/min class)
   which is well within the < 3:1 target.

**Applied to the combined dataset:**

| Class | Count | % |
|-------|-------|---|
| Bug | 2,274 | 39.8% |
| Enhancement | 2,039 | 35.7% |
| Documentation | 1,397 | 24.5% |
| Total usable | 5,710 | |

---

## Decision 022 — Category prefix removal is the only title transformation

**Date:** 2026-07-28
**Stage:** 2A
**Status:** Accepted

### Context

The Stage 1D leakage analysis found that 9.3% of usable issue titles contain
a category prefix (e.g. `[BUG]`, `BUG:`, `ENH:`, `RFC:`, `DOC:`).  These
prefixes directly reveal the target class and would cause information leakage
into models that use raw title text.

### Decision

Remove known category prefixes from the BEGINNING of issue titles only.

Prefixes removed (case-insensitive, colon or bracket form):
- BUG: / [BUG]
- DOC: / DOCS: / [DOC] / [DOCS]
- ENH: / ENHANCEMENT: / [ENH] / [ENHANCEMENT]
- FEATURE: / [FEATURE]
- NEW FEATURE: / [NEW FEATURE]
- RFC: / [RFC]
- FIX: / [FIX]
- MAINT: / MAINTENANCE: / [MAINT]
- TST: / [TST]

The same words appearing in the MIDDLE of a title are NOT removed.

No other title transformations are applied at this stage (no lowercasing,
no punctuation removal, no stopword removal, no stemming).

Actual prefix removals on the combined dataset: **171 titles** (3.0% of 5,710).

### Justification

The apparent discrepancy between the Stage 1D leakage estimate (9.3% using
the audit.py detect_leakage pattern) and the Stage 2A result (3.0%) is
explained by:
1. The audit.py leakage check uses a broader pattern (matches bare words at
   the start, e.g. "BUG ") whereas the preprocessing only removes well-formed
   prefixes (colon or bracket forms like "BUG:" or "[BUG]").
2. The audit checked the broader provisional usable subset; Stage 2A applies
   to the final 5,710 record subset only.

The conservative approach (colon/bracket forms only) is preferred because
removing bare words would incorrectly strip legitimate phrases like
"Bug tracker: is the title" or "RFC compliance".

---

## Decision 023 — Processed data format: CSV + JSONL + metadata JSON

**Date:** 2026-07-28
**Stage:** 2A
**Status:** Accepted

### Decision

The processed dataset is saved in two machine-readable formats:

1. **CSV** (`scikit-learn_issues_model.csv`) — for pandas, Excel inspection,
   and table-based tools. Lists of labels are JSON-encoded in the
   `original_labels` column.

2. **JSONL** (`scikit-learn_issues_model.jsonl`) — one record per line,
   ideal for streaming and transformer fine-tuning pipelines.

3. **Metadata JSON** (`scikit-learn_issues_model_metadata.json`) — provenance,
   exclusion counts, class statistics, SHA-256 hashes of the CSV and JSONL.

Both CSV and JSONL contain identical records in identical order, verified by
the equivalence check in `scripts/prepare_dataset.py`.

Processed data is gitignored (large, reproducible from the combined raw file).

---

*Decisions 001–023 recorded as of Stage 2A.*
