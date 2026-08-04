# AI Tool Usage Log — GitHub Issue Intelligence

This file records how AI tools (GPT-4, Gemini, Antigravity coding assistant,
etc.) were used during the project, as required by the course lecturer.

The intent is transparency: to show which parts of the project were assisted
by AI and how the output was reviewed, adapted, and integrated.

---

## Entry 001 — Project Initialisation (Stage 0)

**Date:** 2026-07-22
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 0 — Project Initialisation

### What the AI assisted with

1. **Brainstorming and refining the project idea**
   - We described the general concept (analysing GitHub issues with ML) and
     worked with the AI to sharpen the problem statement, choose candidate
     repositories, and define a useful output (issue category, component,
     similar issues, duplicate detection).

2. **Planning the project architecture**
   - The AI suggested a staged workflow (data → baselines → classical NLP →
     transformers → retrieval → API) and explained the rationale for each stage.
   - We reviewed, challenged, and adapted the plan to match the course
     requirements and our own understanding.

3. **Defining the gradual workflow rules**
   - We iterated with the AI on a set of project development rules (no
     overengineering, small reviewable stages, no secrets in code, etc.)
     to ensure the project remains understandable to both team members.

4. **Initialising the repository structure**
   - The AI generated the initial file tree, all boilerplate files
     (`.gitignore`, `.env.example`, `requirements.txt`, `README.md`, docs),
     and the minimal test suite.
   - We reviewed every file before approving it.
   - The AI executed the virtual environment setup and dependency installation
     and reported the results.

### What we did ourselves

- Defined the overall project goal and course constraints.
- Chose the three candidate repositories (scikit-learn, pandas, vLLM).
- Reviewed and approved every generated file.
- Made the decision to audit the data before committing to a label scheme.

### How we verified the AI output

- Read every generated file before approving.
- Ran `pytest` and `ruff` to confirm the scaffolding works.
- Reviewed the plan in `docs/project_plan.md` against the course requirements.

---

## Entry 002 — Stage 1A: GitHub Issue Collection Pipeline

**Date:** 2026-07-22
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 1A — Data Collection

### What the AI assisted with

1. **Designing the GitHub API client module** (`github_client.py`)
   - Suggested using `requests.Session` for connection reuse.
   - Designed the PR filtering logic (checking for the `"pull_request"` key).
   - Designed the Link-header pagination parser.
   - Structured the metadata dictionary to include only safe, non-sensitive fields.

2. **Writing the CLI script** (`collect_issues.py`)
   - Generated the `argparse` interface with `--help` text.
   - Wrote the statistics printer (label frequencies, date range, missing bodies).

3. **Writing the unit test suite** (`test_github_client.py`)
   - Designed the `FakeResponse` class to avoid any real network calls.
   - Wrote all 7 test cases covering PR exclusion, pagination, token safety,
     HTTP errors, and file output verification.

4. **Debugging and fixing a real bug**
   - The AI discovered and fixed a `KeyError: 'page'` bug that appeared when
     following GitHub Link-header pagination (params dict was cleared but the
     debug log still referenced it). Fixed by introducing a separate `page_number`
     counter.

5. **Fixing a Windows encoding issue**
   - The `✅` emoji in print statements caused a `UnicodeEncodeError` on Windows
     cp1252 terminals. The AI replaced them with plain ASCII `[OK]` markers.

### What we did ourselves

- Approved the design of the collector before any code was written.
- Reviewed every generated file.
- Ran the actual collection and inspected the real output.

### How we verified the AI output

- All 16 pytest tests pass.
- `ruff check` reports no issues.
- The real collection produced 500 valid issues with 0 PR entries and no token
  data in any output file, confirmed by a programmatic JSON inspection script.

---

## Entry 003 — Stage 1B: Dataset Audit

**Date:** 2026-07-27
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 1B — Dataset Audit

### What the AI assisted with

1. **Designing the audit module** (`audit.py`)
   - Proposed the full set of analysis functions (12 functions).
   - Designed `build_provisional_subset()` with the single-type-label rule.
   - Designed `detect_leakage()` with both prefix and literal-label checks.
   - Designed `categorize_labels_provisionally()` with four roles.

2. **Writing the CLI audit script** (`audit_dataset.py`)
   - Generated all six matplotlib figure-generation functions.
   - Generated the text summary printer.
   - Used `plt.switch_backend("Agg")` to avoid display issues on Windows.

3. **Writing the unit test suite** (`test_audit.py`)
   - 34 new tests covering all 10 required scenarios.
   - All tests use synthetic issue records, not the real raw data.

4. **Writing and executing the Jupyter notebook** (`01_data_audit.ipynb`)
   - Structured 33 cells across 12 sections.
   - The notebook calls `audit.py` functions and focuses on explanation.
   - Executed successfully: all 19 code cells ran, all figures saved.

5. **Documenting decisions** (`docs/decisions.md`)
   - Decisions 007–011 covering the audit architecture, provisional labels,
     task separation, leakage risk, and next collection recommendation.

6. **Debugging issues**
   - Fixed `MissingIDFieldWarning`: added `uuid4`-based `id` fields to all cells.
   - Fixed `FileNotFoundError` from nbconvert's output path resolution by
     switching to `--inplace` execution.

### What we did ourselves

- Reviewed and approved the design of the audit module before coding.
- Inspected the notebook output cells after execution.
- Reviewed the six generated figures for correctness and readability.

### How we verified the AI output

- 50 pytest tests pass (34 new + 16 from Stage 1A).
- `ruff check` reports no issues.
- All 19 notebook code cells executed and have non-empty outputs.
- 6 figure PNG files saved to `reports/figures/`.
- The audit script run produced a complete text summary from the real data.

---

## Entry 004 — Stage 1C: Historical Collection and Label Review

**Date:** 2026-07-27
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 1C — Historical Collection and Label Review

### What the AI assisted with

1. **Extending `github_client.py`**
   - Added `sort` and `direction` parameters to `collect()`.
   - Added ID-based deduplication (`seen_ids` set) with `duplicates_skipped` counter.
   - Added `earliest_created_at` and `latest_created_at` to metadata.
   - Backwards compatible — defaults unchanged for existing Stage 1A usage.

2. **Extending `collect_issues.py`**
   - Added `--sort` and `--direction` CLI arguments.
   - Updated summary output to show sort/direction and duplicates skipped.
   - Added historical collection example to the script docstring.

3. **Extending `audit.py`**
   - Added `compute_yearly_distribution()` — counts issues per year.
   - Added `compute_label_distribution_by_year()` — type-label counts by year
     (single-type rule applies).
   - Added `build_label_scheme_stats()` — computes class counts for any custom
     label scheme, including imbalance ratio and min class count.

4. **Extending `audit_dataset.py`**
   - Added `make_yearly_chart()` — bar chart of issues per year.
   - Added `make_label_by_year_chart()` — stacked bar chart of type labels by year.
   - Updated `print_summary()` to include date coverage and label scheme comparison.
   - Three schemes (A, B, C) are evaluated automatically during each audit run.

5. **Writing new unit tests**
   - 19 new tests in `test_github_client.py` (sort/direction params, deduplication,
     date coverage, output path independence).
   - 15 new tests in `test_audit.py` (yearly distribution, label by year,
     label scheme stats).

6. **Running the real historical collection**
   - Executed: `sort=created&direction=asc&max_issues=5000`
   - Result: 5,000 issues, 2010-2018, 0 duplicates.

7. **Running the historical audit**
   - Generated 8 figures in `reports/figures/history/`.
   - Produced three-scheme comparison in the audit output.

8. **Documenting five new decisions** (012–016)
   - Historical collection method
   - "Enhancement" label discovery
   - RFC retirement
   - Build/CI status
   - Recommended Scheme C (3-class)

### What we did ourselves

- Reviewed the staging of all code changes.
- Interpreted the label scheme comparison output.
- Made the final architectural decision to recommend Scheme C.

### How we verified the AI output

- 69 pytest tests pass (total: 34 + 22 + 13 across three test files).
- `ruff check` reports no issues.
- 5,000 issues collected and verified in the JSON output.
- 8 figure PNG files saved to `reports/figures/history/`.
- Token never appears in any output file (confirmed by metadata inspection).
- `earliest_created_at` = `2010-08-31T07:38:16Z` (9 years of history).

---

## Entry 005 — Stage 1D: Historical Coverage Completion and Label Mapping Review

**Date:** 2026-07-28
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 1D — Historical Coverage and Label Validation

### What the AI assisted with

1. **Discovering the GitHub API pagination limit**
   - Attempting to resume collection with `start_page=112` returned HTTP 422.
   - The AI identified this as a known GitHub limitation for large repositories
     and proposed using the `since` query parameter instead.

2. **Extending the collection pipeline**
   - Added `since` parameter to `GitHubIssueCollector.collect()`.
   - Added `--since` CLI argument to `collect_issues.py`.
   - Added `start_page` parameter (retained for small repositories).
   - Collected 8,380 additional issues (2011–2026-07-27) using
     `--since 2018-05-29T00:00:00Z`.

3. **Creating the combine module and script**
   - New `src/issue_intelligence/data/combine.py` with `load_and_combine`,
     `find_yearly_gaps`, `sample_issues_by_label`, `build_cooccurrence_stats`,
     and `save_combined`.
   - New `scripts/combine_datasets.py` CLI that merges, deduplicates, and
     reports yearly coverage.
   - Combined all three raw files into 12,190 unique issues covering
     2010–2026 with no gaps from 2011 onward.

4. **Generating the label review sample**
   - New `scripts/label_review_sample.py` samples up to 40 issues per label
     (Enhancement, New Feature, RFC, Build / CI) with a fixed seed.
   - Saves `reports/label_mapping_review.csv` (160 rows) for human review.
   - Prints co-occurrence statistics, pairwise overlap, frequency by year, and
     the full candidate scheme evaluation.

5. **Writing tests for all new functionality**
   - New `tests/test_combine.py` with 36 tests covering combination,
     deduplication, yearly gap detection, deterministic sampling,
     the 3-class scheme, Build/CI exclusion, and co-occurrence stats.
   - Two new tests in `test_github_client.py` for `start_page` and `since`.

### What the student reviewed and verified

- Confirmed `reports/label_mapping_review.csv` was saved with 160 rows.
- Confirmed `data/raw/scikit-learn_issues_combined.json` has 12,190 unique issues.
- Inspected yearly distribution: no gaps 2011–2026.
- Reviewed candidate scheme counts: Bug 2,274 / Enhancement 2,039 / Documentation 1,397.
- Reviewed Build / CI co-occurrence: 72/365 issues (19.7%) co-occur with Bug,
  confirming it is primarily a modifier, not a type class.
- Ran `pytest tests/ -v` (103 passed) and `ruff check` (clean).
- Reviewed git diff before committing.

### What was NOT AI-generated

- The decision to exclude Build / CI from the type target (human semantic judgement).
- Verification that 2019 is the correct transition year from "Enhancement" to "New Feature".
- The final interpretation of the co-occurrence statistics.

---

*Future entries will be added at the end of each stage.*

---

## Entry 006 — Stage 2A: Cleaned Modelling Dataset Preparation

**Date:** 2026-07-28
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** 2A — Dataset Preparation

### What the AI assisted with

1. **Designing and implementing `preprocessing.py`**
   - Label mapping (Bug, Documentation, Enhancement) with reverse lookup dict.
   - Conservative text cleaning: prefix removal, whitespace normalization.
   - `build_dataset()` pipeline with ordered exclusion rules (dedup → no target
     → multi-target → empty text).
   - Deterministic sorting by (created_at, issue_id).
   - I/O helpers: `save_csv()`, `save_jsonl()`, `build_metadata()` with SHA-256
     hashes.

2. **Implementing `test_preprocessing.py`**
   - 68 unit tests covering all 17 required test categories.
   - Tests for each label class, each prefix form (colon and bracket),
     case-insensitivity, position-sensitivity, CSV/JSONL equivalence, metadata
     correctness, ordering, and edge cases.
   - All synthetic records — no external file I/O in tests.

3. **Implementing `scripts/prepare_dataset.py`**
   - CLI with `--input`, `--out-dir`, `--prefix`, `--sample`, `--seed`.
   - Full summary printout with exclusion counts, class distribution, imbalance
     ratio, prefix removal statistics, and date range.
   - CSV/JSONL equivalence verification section.
   - Random example inspector per class.
   - Prefix-removal example display.
   - Windows cp1252 compatibility via `sys.stdout.reconfigure(encoding="utf-8")`.

### What the student reviewed and verified

- Ran `pytest tests/ -v` — all 171 tests pass.
- Ran `ruff check` — clean.
- Ran `prepare_dataset.py` on the real 12,190-issue combined file.
- Inspected random examples from all three classes and verified targets.
- Verified 8 prefix-removal examples showing `raw_title` vs `clean_title`.
- Confirmed CSV/JSONL equivalence check: 5,710 rows, same order, same targets.
- Confirmed the data files are gitignored (not committed).

### What was NOT AI-generated

- The decision to use colon/bracket forms only (conservative, avoids
  removing legitimate title words like "Bug report").
- The justification for discrepancy between Stage 1D leakage estimate (9.3%)
  and Stage 2A actual removals (3.0%), explained in Decision 022.
- Review of the prefix-removal examples to confirm no meaningful words were
  accidentally stripped.

---

## Entry 007 — Handoff Preparation (Post-Interruption)

**Date:** 2026-08-02
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 2A verification and partner handoff

### What the AI assisted with
- Inspected the current repository state after an unexpected quota interruption.
- Verified that Stage 2A was completely committed in commit `ac45b77`.
- Verified that 171 tests passed and `ruff check` was clean.
- Verified there was no partial Stage 2B work (the working tree was clean).
- Rewrote `docs/handoff.md` to clearly reflect that Stage 2A was complete and Stage 2B is the next step.
- Updated `README.md` and `docs/project_plan.md` to indicate Stage 2A completion.

---

## Entry 008 — Data Splitting (Stage 2B)

**Date:** 2026-08-03
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 2B

### What the AI assisted with
- Created `src/issue_intelligence/data/splitting.py` to handle both temporal and stratified random splitting.
- Implemented `build_temporal_split` with boundary adjustments to ensure issues from the same calendar date are never split across train/validation/test sets.
- Created robust test suite in `tests/test_splitting.py`.
- Formatted and linted code automatically (Ruff).
- Implemented `scripts/create_splits.py` CLI utility.

### What the student reviewed and verified
- Reviewed the temporal cutoff adjustment logic to ensure no data leakage across dates.
- Verified that all 11 unit tests in `test_splitting.py` pass.
- Examined the final CLI output which confirms exactly 70.0% train, 15.0% validation, and 15.0% test distribution for both splits.

---

## Entry 009 — Simple Baselines (Stage 2C)

**Date:** 2026-08-03
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 2C

### What the AI assisted with
- Created `src/issue_intelligence/evaluation/metrics.py` to calculate accuracy, macro F1, weighted F1, and class-wise precision/recall, as well as confusion matrices.
- Implemented `MajorityClassBaseline` and `StratifiedRandomBaseline` wrapper models.
- Wrote robust tests for metrics and baselines.
- Created `scripts/run_baselines.py` to load dataset splits, generate CSV/JSON results, and plot confusion matrices.
- Updated documentation status to reflect baseline completion.

### What the student reviewed and verified
- Reviewed metric calculations to ensure correct weighting and zero division handling.
- Verified the reason for the Macro F1 difference between temporal and random splits: the majority class naturally changes depending on the evaluation window.
- Verified all 192 tests pass successfully and Ruff finds zero errors.

 
 - - - 
 
 
 
 # #   E n t r y   0 1 0   - -   C l a s s i c a l   T F - I D F   M o d e l s   ( S t a g e   3 A ) 
 
 
 
 * * D a t e : * *   2 0 2 6 - 0 8 - 0 4 
 
 * * T o o l : * *   G o o g l e   A n t i g r a v i t y   ( A I   c o d i n g   a s s i s t a n t   p o w e r e d   b y   G e m i n i ) 
 
 * * S t a g e : * *   S t a g e   3 A 
 
 
 
 # # #   W h a t   t h e   A I   a s s i s t e d   w i t h 
 
 -   C r e a t e d   s r c / i s s u e _ i n t e l l i g e n c e / m o d e l s / c l a s s i c a l . p y   w i t h   L R ,   L i n e a r S V C ,   S G D ,   M u l t i n o m i a l N B   p i p e l i n e s . 
 
 -   C r e a t e d   s r c / i s s u e _ i n t e l l i g e n c e / m o d e l s / m o d e l _ r e g i s t r y . p y . 
 
 -   C r e a t e d   s c r i p t s / r u n _ c l a s s i c a l _ m o d e l s . p y   w i t h   t i m i n g ,   v o c a b   s t a t s ,   t o p - f e a t u r e   e x t r a c t i o n . 
 
 -   C r e a t e d   t e s t s / t e s t _ c l a s s i c a l _ m o d e l s . p y   ( 1 7   t e s t s ) .   A l l   2 0 9   p a s s . 
 
 
 
 # # #   W h a t   t h e   s t u d e n t   r e v i e w e d   a n d   v e r i f i e d 
 
 -   V e r i f i e d   t h a t   T F - I D F   i s   o n l y   f i t t e d   o n   t r a i n i n g   d a t a . 
 
 -   C o n f i r m e d   S G D C l a s s i f i e r   a c h i e v e s   b e s t   t e m p o r a l   M a c r o   F 1   ( 0 . 9 0 7 7 ) . 
 
 -   C o n f i r m e d   t e m p o r a l   s c o r e s   e x c e e d   r a n d o m   s c o r e s ,   e v i d e n c i n g   d i s t r i b u t i o n   s h i f t . 
 
 

---

## Entry 011 -- Robustness Validation (Stage 3B)

**Date:** 2026-08-04
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 3B

### What the AI assisted with
- Created `src/issue_intelligence/evaluation/robustness.py` with 10 experiment utilities.
- Created `scripts/run_robustness_checks.py` running all experiments in one pass.
- Created `tests/test_robustness.py` (28 tests). All 237 project tests pass.
- Generated 8 figures: variant comparison, year performance, length buckets, margin distributions, near-dup histogram.
- Generated 5 output files: robustness_summary.csv, robustness_details.json, misclassified_examples.csv, near_duplicate_review.csv, influential_term_review.csv.
- Corrected the earlier wrong explanation for temporal vs random performance gap.

### What the student reviewed and verified
- Confirmed zero near-duplicates at cosine thresholds 0.90, 0.95, and 0.99.
- Verified label-word masking only reduces Macro F1 by 0.003 (SGD: 0.9077 to 0.9051).
- Confirmed performance is consistent across years 2022-2024 (no cliff or collapse).
- Reviewed misclassified examples CSV and influential term CSV for suspicious patterns.


---

## Entry 012 -- Hyperparameter Tuning (Stage 3C)

**Date:** 2026-08-04
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 3C

### What the AI assisted with
- Created `src/issue_intelligence/evaluation/temporal_cv.py` with expanding-window fold generator.
- Created `src/issue_intelligence/models/tuning.py` with candidate definitions and selection logic.
- Created `scripts/tune_classical_models.py` running full Phase 1 + Phase 2 search.
- Created `tests/test_temporal_cv.py` (8 tests) and `tests/test_tuning.py` (11 tests). All 258 pass.
- Generated 4 output files: fold_results.csv, all_candidates.csv, best.json, validation_comparison.csv.
- Generated 6 figures in reports/figures/tuning/.

### What the student reviewed and verified
- Confirmed inner fold boundaries are strictly chronological (2010→2015, 2015→2018, 2018→2020, 2020→2022).
- Verified that LinearSVC temporal val improved by +0.0027 and noted this is not practically meaningful.
- Confirmed min_df=5 reduces vocabulary from 67,838 to 20,473 terms (70%) with negligible F1 loss.
- Confirmed test split was never accessed.


---

## Entry 013 -- Transformer Classifier (Stage 4A)

**Date:** 2026-08-04
**Tool:** Google Antigravity (AI coding assistant powered by Gemini)
**Stage:** Stage 4A

### What the AI assisted with
- Created `src/issue_intelligence/data/transformer_dataset.py` with dataset utilities,
  label encoding, truncation statistics.
- Created `src/issue_intelligence/models/transformer.py` with model loading, class weight
  computation, parameter counting.
- Created `scripts/train_transformer.py` — full training loop with WeightedTrainer, epoch
  history tracking, 6 figures.
- Created `scripts/evaluate_transformer.py` — standalone evaluation.
- Created `tests/test_transformer_dataset.py` (15 tests) and `tests/test_transformer_pipeline.py`
  (11 tests). All 280 tests pass.
- Performed hardware audit: CPU-only, no GPU, 3.8 GB free RAM.
- Timed DistilBERT at 14.5s/step (~180 min for 3 epochs) and BERT-Tiny at 0.26s/step
  (~3.3 min for 3 epochs). Selected BERT-Tiny based on hardware constraint.
- Diagnosed and fixed transformers 5.x API changes: evaluation_strategy → eval_strategy,
  no_cuda → use_cpu, tokenizer → processing_class in Trainer.

### What the student reviewed and verified
- Confirmed BERT-Tiny Macro F1 = 0.8550, substantially below LinearSVC 0.9087 (delta=-0.054).
- Confirmed 49.9% truncation rate is a structural disadvantage for transformers.
- Confirmed LinearSVC is the correct model to advance to final test evaluation.

### Stage 4B: Final Test Set Evaluation
- Prompted GPT to implement the final evaluation script exactly as specified using the locked LinearSVC configuration.
- GPT fixed dataset paths and classification_report parameter mismatches based on the synthetic test suite.

### Stage 5A: FastAPI Inference Service
- Prompted GPT to build `src/issue_intelligence/api` with endpoints for health, model info, and prediction.
- GPT successfully implemented Pydantic validation (requiring at least one non-blank field) and extracted correct SVM `decision_function` raw margins without defaulting to softmax probabilities.
- Used `TestClient` and a synthetic pipeline to rigorously test the API layer without relying on the true test set.
