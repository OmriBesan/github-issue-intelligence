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

*Decisions 001–016 recorded as of Stage 1C.*
