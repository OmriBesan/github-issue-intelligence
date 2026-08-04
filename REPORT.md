# GitHub Issue Intelligence

**Theoretical Data Science — Python Final Project**  
**Authors:** Omri Besan and Leon Pasternak  
**Repository:** https://github.com/OmriBesan/github-issue-intelligence  
**Date:** August 2026

---

## Abstract

GitHub repositories can accumulate thousands of issues describing bugs, documentation problems, feature requests, and other maintenance tasks. Reading, categorizing, and searching these issues requires repeated manual effort from maintainers. This project presents **GitHub Issue Intelligence**, an end-to-end machine-learning system developed and evaluated on the scikit-learn repository. The current trained model classifies scikit-learn GitHub issues into three issue types—**Bug**, **Documentation**, and **Enhancement**—and retrieves lexically similar historical issues. The software architecture can be adapted to other repositories, but reliable use elsewhere would require repository-specific labelled data, retraining, and evaluation.

The project includes a reproducible GitHub data-collection pipeline, dataset auditing and preprocessing, temporal and stratified splitting, baseline models, classical text classifiers, robustness experiments, hyperparameter tuning, a compact transformer benchmark, a final held-out temporal evaluation, a FastAPI inference service, TF-IDF similar-issue retrieval, and a local Streamlit demonstration interface.

The final classifier is a TF-IDF and LinearSVC pipeline trained on the combined temporal training and validation data. On a held-out set of 856 newer issues from 2024–2026, it achieved **0.9300 Macro F1**, **0.9369 accuracy**, and **0.9368 weighted F1**. It also outperformed a BERT-Tiny benchmark while being substantially faster, smaller, and simpler to run on CPU. The final repository contains 341 passing automated tests and a fully documented local demonstration workflow.

---

## 1. Introduction

GitHub Issues are commonly used to report software bugs, request new features, suggest improvements, and identify documentation problems. In large open-source projects, maintainers must repeatedly read new reports, assign labels, decide which team should handle them, and search for related discussions.

The goal of this project was to build a transparent and reproducible system that assists with the first stage of issue triage. Given the title and body of a new scikit-learn issue, the system performs two tasks:

1. Predict whether the issue is a **Bug**, **Documentation** issue, or **Enhancement**.
2. Retrieve historical issues with similar wording and technical content using TF-IDF cosine similarity.

The current model is trained and evaluated specifically on scikit-learn issues. The API can technically accept text from another repository, but the reported performance does not apply outside scikit-learn. Adapting the system to another repository would require collecting labelled issues, defining an appropriate label mapping, retraining the model, rebuilding the retrieval index, and evaluating the new version.

The project was designed as more than a single notebook experiment. It combines data engineering, classical machine learning, neural NLP comparison, temporal evaluation, robustness analysis, API development, testing, and a local user interface.

The project does **not** attempt to replace maintainers or automatically make final labeling decisions. It provides model suggestions and related examples that can support human review.

---

## 2. Problem Definition

The supervised classification task is defined as:

- **Input text (X):** issue title + issue body
- **Target label (y):** Bug, Documentation, or Enhancement


The target classes were derived from scikit-learn's historical GitHub labels.

### 2.1 Label mapping

| Final class | Raw labels |
|---|---|
| Bug | Bug |
| Documentation | Documentation |
| Enhancement | Enhancement, New Feature, RFC |

The three raw improvement-oriented labels were merged into **Enhancement** because they represent the same broad issue type across different periods of the repository's history.

Issues were excluded when they had no reliable target label or when they mapped to more than one target class. Workflow labels, component labels, and administrative labels were not treated as issue types.

### 2.2 Main evaluation metric

**Macro F1** was selected as the primary metric because it gives equal weight to all three classes. This was important because Documentation was smaller than the other two classes. Accuracy and weighted F1 were reported as secondary metrics.

---

## 3. Dataset Collection and Audit

The data was collected from the public scikit-learn GitHub repository through a reproducible Python collection pipeline.

### 3.1 Raw dataset

| Property | Value |
|---|---:|
| Raw issues collected | 12,190 |
| Earliest issue | 2010-10-19 |
| Latest issue | 2026-07-27 |
| Repository | scikit-learn/scikit-learn |

The initial audit examined issue volume, date coverage, label frequency, label overlap, text quality, and target-label coverage.

### 3.2 Final modelling dataset

After label mapping and exclusion rules, the final modelling dataset contained **5,710** usable records.

| Class | Count | Percentage |
|---|---:|---:|
| Bug | 2,274 | 39.8% |
| Enhancement | 2,039 | 35.7% |
| Documentation | 1,397 | 24.5% |
| **Total** | **5,710** | **100%** |

The class imbalance ratio was approximately 1.63:1, which was manageable but still justified the use of Macro F1 and balanced class weights.

### 3.3 Exclusions

| Exclusion reason | Count |
|---|---:|
| No target class | 6,323 |
| Multiple target classes | 157 |
| Duplicate issue ID | 0 |
| Empty combined text | 0 |

More than half of the raw issues were excluded because many scikit-learn issues use component or workflow labels without a clear issue-type label. Keeping only clearly labelled examples reduced dataset size but produced a more reliable supervised-learning target.

---

## 4. Preprocessing

The preprocessing pipeline was implemented as reusable source code rather than notebook-only logic.

The main steps were:

- Preserve the raw title and body for traceability.
- Build a cleaned title and combined title-body text.
- Remove conservative label-like title prefixes such as `BUG:`, `DOC:`, `[RFC]`, and `[ENH]`.
- Preserve technical information such as code blocks, stack traces, multiline text, and error messages.
- Normalize missing title or body fields to empty strings.
- Reject records whose combined text is empty.
- Sort records deterministically by creation time and issue ID.

A conservative prefix-removal strategy was used to avoid deleting legitimate words that happened to appear at the beginning of a title. Only 171 usable records, approximately 3%, had a prefix removed.

The processed CSV and JSONL outputs were checked for equal row counts, matching issue order, and matching target values.

---

## 5. Data Splitting Strategy

Two split strategies were created.

### 5.1 Temporal split

The temporal split was the primary strategy because it more closely represents real deployment: models learn from older issues and are evaluated on newer ones.

| Split | Records | Date range |
|---|---:|---|
| Train | 3,996 | 2010-10-19 to 2022-10-12 |
| Validation | 858 | 2022-10-13 to 2024-05-28 |
| Test | 856 | 2024-05-29 to 2026-07-27 |

The held-out test set remained isolated until the final evaluation.

### 5.2 Stratified random split

A deterministic stratified 70/15/15 random split was also generated for comparison. It preserved global class proportions but was treated as secondary because it mixes issues from different years.

Temporal validation was not assumed to be automatically harder. In this dataset, the tuned LinearSVC scored higher on temporal validation than on random validation. Temporal evaluation remained primary because it better measures prediction on future issues.

---

## 6. Baselines

Two simple baselines were implemented:

1. **Majority-class baseline**
2. **Stratified-random baseline**

### 6.1 Temporal validation baseline results

| Baseline | Macro F1 | Accuracy |
|---|---:|---:|
| Majority class | 0.1719 | 0.3473 |
| Stratified random | 0.3376 | 0.3695 |

The stratified-random Macro F1 of **0.3376** was used as the minimum performance floor for later models.

---

## 7. Classical Machine-Learning Models

Four classical text classifiers were compared using TF-IDF features:

- Logistic Regression
- LinearSVC
- SGDClassifier
- Multinomial Naive Bayes

The initial TF-IDF configuration used lowercasing, unigrams and bigrams, sublinear term frequency, `min_df=2`, and `max_df=0.95`.

### 7.1 Initial temporal validation results

| Model | Macro F1 | Accuracy |
|---|---:|---:|
| SGDClassifier | **0.9077** | **0.9161** |
| LinearSVC | 0.9060 | 0.9138 |
| Logistic Regression | 0.9051 | 0.9138 |
| Multinomial Naive Bayes | 0.8405 | 0.8706 |

The three linear models performed similarly and substantially outperformed the baselines. Multinomial Naive Bayes was weaker, especially on the less frequent Documentation class.

---

## 8. Robustness and Error Analysis

The strong validation score was investigated through multiple checks rather than accepted at face value.

The experiments included:

- title-only classification,
- body-only classification,
- masking target-label words,
- evaluation on prefix-free examples,
- performance by year,
- performance by text-length bucket,
- exact-duplicate detection,
- near-duplicate detection,
- decision-margin analysis,
- review of misclassified examples,
- review of influential terms.

### 8.1 Selected robustness results

| Experiment | SGD Macro F1 | LinearSVC Macro F1 |
|---|---:|---:|
| Combined text | 0.9077 | 0.9060 |
| Title only | 0.7615 | 0.7502 |
| Body only | 0.9029 | 0.8963 |
| Label words masked | 0.9051 | 0.9016 |
| Prefix-free subset | 0.9051 | 0.9032 |

Masking obvious label words caused only a very small decrease, which provides evidence that the model was not relying mainly on explicit words such as “bug” or “documentation.”

No exact train-validation duplicates were found. No near-duplicate pairs were found at cosine-similarity thresholds of 0.90, 0.95, or 0.99. These checks reduce concern that repeated issue text artificially inflated validation performance.

Performance decreased for very long issues, which is a useful limitation. Long bodies may contain irrelevant conversation, logs, or mixed technical context that weakens the signal.

---

## 9. Hyperparameter Tuning

Hyperparameters were tuned using expanding-window temporal cross-validation inside the training split. This avoided selecting parameters using the external validation or test sets.

The search compared SGDClassifier and LinearSVC configurations, including:

- regularization strength,
- loss functions,
- class weighting,
- TF-IDF `min_df`,
- unigram versus bigram features,
- feature limits.

### 9.1 Selected model

The selected model was:

- **Classifier:** LinearSVC
- **C:** 0.3
- **Class weight:** balanced
- **TF-IDF n-grams:** (1, 2)
- **min_df:** 5
- **max_df:** 0.95
- **sublinear_tf:** True
- **max_features:** None

On the temporal training split alone, this configuration produced a vocabulary of **20,473** terms.

### 9.2 Tuned validation results

| Model | Temporal Macro F1 |
|---|---:|
| Tuned SGDClassifier | 0.9055 |
| Tuned LinearSVC | **0.9087** |

The improvement over the untuned models was small. The main practical benefit of tuning was reducing vocabulary size while preserving performance.

---

## 10. Transformer Benchmark

A compact transformer benchmark was added to test whether a neural language model would outperform the classical approach.

The model was **BERT-Tiny** (`google/bert_uncased_L-2_H-128_A-2`), selected because the project was trained on a CPU-only Windows machine with limited memory.

### 10.1 Configuration

- 2 transformer layers
- hidden size 128
- 3 epochs
- batch size 16
- learning rate `5e-5`
- maximum sequence length 256
- class-weighted cross-entropy

Approximately 49.9% of training examples were truncated at 256 tokens. This was a disadvantage compared with TF-IDF, which used the full combined text.

### 10.2 Validation comparison

| Metric | LinearSVC | BERT-Tiny |
|---|---:|---:|
| Macro F1 | **0.9087** | 0.8550 |
| Training time | 1.94 s | 411.9 s |
| Inference time | 0.39 s | 4.867 s |
| Model size | <1 MB during tuning | 18.3 MB |

BERT-Tiny was approximately 5.4 Macro-F1 points lower and much slower. Therefore, LinearSVC advanced to the final held-out test evaluation.

This result does not imply that transformers are generally inferior. The comparison was limited by a very small checkpoint, CPU constraints, token truncation, and limited tuning. It does show that a well-designed classical model was a better choice for this dataset and hardware environment.

---

## 11. Final Held-Out Temporal Evaluation

After model selection was complete, the tuned LinearSVC pipeline was refitted on the combined temporal training and validation sets. The held-out temporal test set was then loaded for the first and only final evaluation.

The final fitted TF-IDF vocabulary contained **27,129** terms because the vectorizer was refitted on the larger train-plus-validation corpus.

### 11.1 Final results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bug | 0.9488 | 0.9701 | 0.9593 |
| Documentation | 0.8981 | 0.8981 | 0.8981 |
| Enhancement | 0.9500 | 0.9157 | 0.9325 |
| **Macro average** |  |  | **0.9300** |

Additional metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9369 |
| Weighted F1 | 0.9368 |
| Test records | 856 |
| Training time on train+validation | 3.37 s |
| Test inference time | 0.54 s |
| Saved pipeline size | 1.25 MB |

The test Macro F1 was higher than the temporal validation result. This provides additional evidence that the selected LinearSVC pipeline generalizes well to newer scikit-learn issues. It does not prove that all future issue distributions will behave similarly.

---

## 12. Final System Architecture

The final project contains three application layers.

### 12.1 Classification service

A FastAPI service loads the final saved pipeline and exposes:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /similar`

The prediction response contains:

- predicted label,
- raw LinearSVC decision scores,
- decision margin,
- model name.

The decision scores are **not probabilities** and are not converted into percentages. The predicted class is the class with the highest score. The decision margin is the difference between the highest and second-highest scores.

Model and retrieval loading are independent. If the retrieval artifact is unavailable, classification can continue to work.

### 12.2 Similar-issue retrieval

The retrieval component uses the same final TF-IDF vectorizer and exact cosine similarity.

The retrieval corpus contains only the temporal training and validation issues:

| Property | Value |
|---|---:|
| Indexed issues | 4,854 |
| Bug | 1,873 |
| Enhancement | 1,790 |
| Documentation | 1,191 |
| Matrix shape | 4,854 × 27,129 |
| Date range | 2010-10-19 to 2024-05-28 |

The held-out temporal test set was not included in the retrieval index.

Returned values are cosine-similarity scores. They are not probabilities, confidence values, or proof that two issues are duplicates. The feature should be understood as lexical similar-issue retrieval.

### 12.3 Streamlit demonstration interface

A local Streamlit interface communicates with the FastAPI service through HTTP. It does not load model artifacts or datasets directly.

The user can:

- enter an issue title and body,
- request a prediction,
- view raw class decision scores,
- view the decision margin,
- retrieve between 1 and 10 similar historical issues,
- optionally filter retrieved issues by class.

Classification and retrieval failures are handled independently, and backend errors are shown as user-friendly messages rather than raw tracebacks.

---

## 13. Software Quality and Reproducibility

The project was developed as a Python package with reusable modules, scripts, tests, documentation, and tracked design decisions.

Final verification results:

| Check | Result |
|---|---|
| Automated tests | 341 passed |
| Ruff | All checks passed |
| `pip check` | No broken requirements |
| Git working tree | Clean at final verification |
| Generated model artifacts | Ignored and reproducible |
| Processed datasets | Ignored and reproducible |
| Secrets and local environment files | Not committed |

The repository documents commands for:

- collecting issues,
- preparing the modelling dataset,
- creating splits,
- running baselines,
- training and tuning models,
- evaluating the transformer,
- generating the final model,
- building the retrieval index,
- starting FastAPI,
- starting Streamlit.

---

## 14. Limitations

The system has several important limitations.

### 14.1 Repository-specific training

The classifier was trained only on scikit-learn issues. Label definitions, issue templates, vocabulary, and contributor behavior may differ in another repository. Cross-repository performance was not evaluated.

### 14.2 Label noise

GitHub labels are assigned by humans and may be inconsistent across maintainers and time periods. The target classes are useful but are not perfect ground truth.

### 14.3 Excluded issues

Issues without a clear type label were excluded. The model therefore learns from the most reliably labelled subset rather than all repository activity.

### 14.4 Uncalibrated decision scores

LinearSVC decision values are not calibrated probabilities. A larger margin often indicates stronger separation, but it should not be interpreted as an exact probability of correctness.

### 14.5 Lexical retrieval

TF-IDF similarity is based mainly on shared terms and phrases. It can miss conceptually related issues that use different vocabulary. It can also retrieve issues that share words but are not truly related.

### 14.6 Transformer benchmark constraints

The transformer experiment used BERT-Tiny, CPU training, a 256-token limit, and limited tuning. It was a practical benchmark rather than a comprehensive evaluation of transformer architectures.

### 14.7 No confirmed duplicate detection

The project retrieves similar issues but does not classify issues as confirmed duplicates. Reliable duplicate detection would require labelled duplicate relationships and a separate evaluation protocol.

---

## 15. Use of AI Tools

AI coding assistants were used throughout the project for planning, implementation suggestions, documentation, debugging, and review.

The project team remained responsible for:

- selecting the final task and dataset,
- approving the label mapping,
- protecting the held-out test set,
- reviewing generated code,
- checking reported metrics,
- identifying incorrect claims,
- rejecting unverified partner-branch work,
- running tests and linting,
- correcting documentation,
- understanding the final implementation.

AI-generated changes were accepted only after repository inspection and automated verification. The repository includes a dedicated GPT-usage log describing how AI assistance was used and corrected.

---

## 16. Conclusion

GitHub Issue Intelligence demonstrates that a classical text-classification pipeline can provide strong and efficient issue triage on a real open-source dataset.

The final TF-IDF and LinearSVC classifier achieved **0.9300 Macro F1** on a held-out temporal test set of newer scikit-learn issues. It outperformed the BERT-Tiny benchmark while requiring less training time, less storage, and simpler CPU inference.

The project also extends beyond offline classification. The selected model is exposed through a tested FastAPI service, connected to TF-IDF similar-issue retrieval, and demonstrated through a local Streamlit interface.

The main lesson is that model complexity should be justified by evidence. For this dataset, careful label design, temporal evaluation, leakage checks, robust classical features, and disciplined validation mattered more than choosing a larger neural model.

---

## Appendix A: Main Results Summary

| Stage | Main result |
|---|---|
| Raw collection | 12,190 scikit-learn issues |
| Final modelling data | 5,710 issues |
| Baseline floor | 0.3376 Macro F1 |
| Best initial classical model | SGDClassifier, 0.9077 validation Macro F1 |
| Selected tuned model | LinearSVC, 0.9087 validation Macro F1 |
| BERT-Tiny benchmark | 0.8550 validation Macro F1 |
| Final held-out test | 0.9300 Macro F1 |
| Final accuracy | 0.9369 |
| Final weighted F1 | 0.9368 |
| Retrieval corpus | 4,854 historical issues |
| Automated tests | 341 passing |

## Appendix B: Local Demonstration Commands

Start the FastAPI backend:

```powershell
.venv\Scripts\python -m uvicorn issue_intelligence.api.app:app --reload
```

Start the Streamlit interface in a second terminal:

```powershell
.venv\Scripts\python -m streamlit run src/issue_intelligence/ui/app.py
```
