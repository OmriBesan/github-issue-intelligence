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
# #   D e c i s i o n   0 2 6   - -   C l a s s   W e i g h t i n g   f o r   C l a s s i c a l   M o d e l s 
 * * D a t e : * *   2 0 2 6 - 0 8 - 0 4 
 * * C o n t e x t : * *   T h e   t a r g e t   c l a s s e s   a r e   i m b a l a n c e d   ( B u g   4 0 % % ,   D o c u m e n t a t i o n   2 5 % % ,   E n h a n c e m e n t   3 6 % % ) .   W i t h o u t   c o r r e c t i o n   m o d e l s   m a y   u n d e r - p r e d i c t   t h e   m i n o r i t y   c l a s s . 
 * * D e c i s i o n : * *   U s e   c l a s s _ w e i g h t = ' b a l a n c e d '   f o r   L R ,   L i n e a r S V C ,   a n d   S G D C l a s s i f i e r .   M u l t i n o m i a l N B   d o e s   n o t   s u p p o r t   c l a s s _ w e i g h t .   B a l a n c e d   w e i g h t i n g   i m p r o v e d   D o c u m e n t a t i o n   r e c a l l   v s   a n   u n w e i g h t e d   r u n . 
 
 # #   D e c i s i o n   0 2 7   - -   S G D   l o g _ l o s s   a s   L i n e a r   T e x t   L o s s 
 * * D a t e : * *   2 0 2 6 - 0 8 - 0 4 
 * * C o n t e x t : * *   S G D C l a s s i f i e r   s u p p o r t s   m u l t i p l e   l o s s   f u n c t i o n s .   l o g _ l o s s   g i v e s   c a l i b r a t e d   p r o b a b i l i t i e s   a n d   b e h a v e s   l i k e   o n l i n e   L o g i s t i c   R e g r e s s i o n   f o r   t e x t . 
 * * D e c i s i o n : * *   U s e   S G D C l a s s i f i e r ( l o s s = ' l o g _ l o s s ' )   s o   a l l   t h r e e   l i n e a r   m o d e l s   a r e   d i r e c t l y   c o m p a r a b l e   i n   t h e i r   o p t i m i s a t i o n   o b j e c t i v e s . 
 
 

## Decision 028 -- 0.90+ Macro F1 is Credible After Robustness Checks
**Date:** 2026-08-04
**Context:** A Temporal Macro F1 of 0.9077 is unusually high and required validation before being trusted.
**Evidence:**
- Zero exact or near-duplicate train/val pairs at any cosine threshold (0.90, 0.95, 0.99).
- Max p99 train-val cosine similarity is only 0.65 — well below any near-duplicate threshold.
- Label-word masking (bug, docs, enhancement, feature, RFC) dropped Macro F1 by only 0.003.
- Performance is consistent across all three validation years: 0.8871 (2022), 0.9194 (2023), 0.8942 (2024).
- Body-only features score 0.9029 vs full combined 0.9077 — issue bodies carry genuine semantic information.
- Prefix-free subset (records without category prefixes) scores 0.9051 — nearly identical to full set.
**Decision:** The 0.90+ Macro F1 result is credible. No data leakage or shortcut learning was detected.
The model learns substantive textual patterns, not label shortcuts.

## Decision 029 -- Why Temporal Performance Exceeds Random Performance
**Date:** 2026-08-04
**Context:** Temporal Macro F1 (SGD 0.9077) exceeds Random Macro F1 (SGD 0.8745).
Training set sizes are essentially equal (3996 vs 3995). The original explanation that the temporal
training set was larger was incorrect.
**Correct explanation:** The temporal training set covers 2010-2022, a period with more mature and
consistently-labelled scikit-learn issues. The random training set mixes issues from all years,
including the most recent noisier issues, making the learned representation slightly less consistent.
The temporal validation set (2022-2024) is a coherent temporal slice, while the random validation
set is a mixed-year sample. This means temporal evaluation is a stricter, more realistic assessment
of how the model will perform on future issues.


## Decision 030 -- Best Classical Candidate: LinearSVC (C=0.3, min_df=5, bigrams)
**Date:** 2026-08-04
**Context:** Stage 3C targeted tuning of SGDClassifier and LinearSVC using 3-fold expanding-window
temporal cross-validation on the training split only.
**Inner fold results:**
- Best SGD: hinge, alpha=0.001, balanced, ngram=(1,2), min_df=5 → mean F1=0.8199, std=0.0646
- Best LinearSVC: C=0.3, balanced, ngram=(1,2), min_df=5 → mean F1=0.8219, std=0.0667
**External validation:**
- SGD temporal val: 0.9055 (delta=-0.0022 vs untuned 0.9077)
- LinearSVC temporal val: 0.9087 (delta=+0.0027 vs untuned 0.9060)
**Decision:** LinearSVC with C=0.3, balanced weighting, bigrams (1,2), min_df=5 is named the
best classical candidate. Vocabulary shrinks from 67,838 to 20,473 terms (70% reduction) with
no meaningful loss in accuracy.
This is NOT the final project model. It will be compared with future transformer models.

## Decision 031 -- Tuning Improvement is Not Practically Meaningful
**Date:** 2026-08-04
**Context:** The best LinearSVC configuration improved temporal Macro F1 by +0.0027.
The best SGD configuration degraded by -0.0022.
**Decision:** The Stage 3A untuned models were already very well-configured. The tuning search
confirmed that the original settings (balanced weighting, bigrams, min_df=2) were nearly optimal.
The only meaningfully actionable finding is that min_df=5 shrinks the vocabulary by 70% with
negligible performance cost, making the model faster to fit and inspect.

## Decision 032 -- Inner Fold F1 (~0.82) Substantially Lower Than External Val (~0.91)
**Date:** 2026-08-04
**Context:** Inner expanding-window folds (training 1000-2999 records, validating on the next
period) produce mean Macro F1 ~0.82, while external validation (training on full 3996 records)
produces ~0.91.
**Explanation:** (1) Fold 1 trains on only 1000 records — insufficient for full TF-IDF coverage.
(2) The inner folds are harder: fold 1 validates on 2015-2018 data with only pre-2015 training.
(3) The full training set has 3× more data (3996 vs ~1000-2999). The gap is expected and not
evidence of leakage; it shows the model benefits from more training data.


## Decision 033 -- Transformer Checkpoint: BERT-Tiny (google/bert_uncased_L-2_H-128_A-2)
**Date:** 2026-08-04
**Context:** Stage 4A requires a transformer-based issue classifier. The preferred checkpoint
is distilbert-base-uncased (66M params, 6 layers). A timing estimate on the project's CPU-only
hardware (Windows 11, no GPU, ~3.8 GB free RAM) showed distilbert requires ~14.5s/step at
batch=16, seq_len=256, giving ~60 min/epoch and ~180 min for 3 epochs — unreasonable.
**Decision:** Use google/bert_uncased_L-2_H-128_A-2 (BERT-Tiny, 4.4M params, 2 layers, hidden=128).
This is the official Google BERT-Tiny checkpoint. It uses standard BERT WordPiece tokenization
(no sentencepiece required). Measured timing: 0.26s/step, ~3.3 min for 3 epochs.
**Expected tradeoff:** BERT-Tiny has 15× fewer parameters and shallower attention than DistilBERT.
Classification F1 is typically 5–15 pp lower on downstream tasks. The goal is not to match
DistilBERT but to determine whether any BERT-family transformer beats the tuned LinearSVC.

## Decision 034 -- BERT-Tiny Does Not Beat Tuned LinearSVC
**Date:** 2026-08-04
**Context:** Stage 4A training result (3 epochs, lr=5e-5, class-weighted loss, max_length=256).
BERT-Tiny temporal val Macro F1: 0.8550
LinearSVC (tuned) temporal val Macro F1: 0.9087
Delta: -0.0537 (BERT-Tiny is 5.4 pp WORSE).
**Decision:** BERT-Tiny does not justify its extra complexity on this dataset.
The tuned LinearSVC remains the stronger model and the preferred candidate for final test evaluation.
This result is not surprising: (1) GitHub issue text is technical but follows predictable patterns
that TF-IDF captures well; (2) BERT-Tiny's 2-layer architecture captures minimal contextual
information; (3) 49.9% of training examples are truncated at 256 tokens, removing information
that TF-IDF retains; (4) The LinearSVC was already tuned; the transformer was not.

## Decision 035 -- Token Truncation: 49.9% at max_length=256
**Date:** 2026-08-04
**Context:** Mean token length is 491.5; median is 256.0; p99 is 3594; max is 25878.
Nearly half of all training examples are truncated, losing tail content (code, stack traces,
comments). This disadvantages transformers relative to TF-IDF which uses the full combined_text.
**Implication:** If a larger transformer (DistilBERT, BERT-base) were used on a GPU, max_length
should be increased to at least 512, or chunked/hierarchical encoding used for very long issues.

## Decision 036 -- LinearSVC Advances to Final Test Evaluation; BERT-Tiny Does Not
**Date:** 2026-08-04
**Context:** The two candidates are tuned LinearSVC (0.9087) and BERT-Tiny (0.8550).
**Decision:** Only LinearSVC advances to final held-out test set evaluation.
BERT-Tiny performed too poorly to warrant using the held-out test set on it.
A stronger transformer (DistilBERT on GPU, or fine-tuned BERT-base) could be revisited in
a future stage if compute becomes available.

## Stage 4B - Final Test Set Evaluation
- **Decision:** The locked LinearSVC configuration was evaluated exactly once on the held-out temporal test set.
- **Rationale:** Strict separation of test data to prevent information leak. 
- **Result:** The model achieved a Macro F1 of 0.9300 and Accuracy of 0.9369. The test set was accessed exactly once and no hyperparameter modifications were made after observing the results.

## Stage 5A - FastAPI Inference Service
- **Decision:** API returns raw LinearSVC decision scores and margins.
- **Rationale:** SVM decision boundaries (from `decision_function`) produce raw decision scores. Applying Platt scaling or softmax post-hoc is not natively calibrated for this pipeline. Reporting raw LinearSVC decision scores prevents misleading the user with pseudo-probabilities.
- **Decision:** Missing model raises HTTP 503 on `POST /predict` but allows the app to start (HTTP 200 on `/health` with `status: not_ready`).
- **Rationale:** Kubernetes or container orchestrators prefer services to start and report readiness clearly, rather than crashing in an infinite restart loop if the model volume is delayed.

### Stage 5B: Similar-Issue Retrieval
- **Retrieval Corpus Composition**: The retrieval artifact is built strictly from the train and validation sets (4,854 records). The held-out temporal test set (856 records) remains completely excluded by construction to protect the integrity of final scientific evaluation.
- **TF-IDF Cosine Method**: We reused the exact TF-IDF vectorizer fitted on the training split to construct a lexical index of the retrieval corpus. Exact cosine similarity (dot product of L2-normalized sparse vectors) provides a fast and robust lexical retrieval method without introducing heavy external dependencies (like vector databases or embeddings).
- **Not Duplicate Detection**: The retrieved results are provided purely based on lexical similarity (cosine scores). These scores are not probabilities, not confidence scores, and do not make quantitative claims about issues being duplicates.
- **Robust API Loading**: The API loads the retrieval artifact via `ISSUE_RETRIEVAL_PATH` during application startup. If the artifact is missing, `/similar` cleanly returns a 503 while preserving full uptime for `/predict`.
