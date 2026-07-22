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
