# Stage 4: Proper Evaluation Report

## Overview
As part of Stage 4, we conducted a rigorous evaluation of the classical NLP models, specifically focusing on Logistic Regression with TF-IDF features.
The purpose of this stage is to understand the model's limitations, investigate data leakage, and analyze failure modes before transitioning to more computationally expensive Transformer models.

## Evaluation Methods
We evaluated the models using two different validation setups:
1. **Stratified Split**: A standard randomized split preserving class distribution.
2. **Temporal Split**: A time-based split (train on older issues, test on newer issues) to simulate real-world conditions.

We computed Macro F1 scores, generated Confusion Matrices, and performed Error Analysis on misclassified instances.

## Findings
- **Temporal vs Stratified**: Performance often drops on temporal splits compared to stratified ones because software distributions drift over time (e.g., new components, different feature requests).
- **Confusion Matrix**: Misclassifications predominantly occur between `Enhancement` and `Bug`. `Documentation` issues are typically well-separated.
- **Error Analysis**: Many misclassified issues contain ambiguous language (e.g., words like "fix" used in feature requests, or "add" used in bug reports). This indicates that simple TF-IDF keyword overlap is insufficient to capture the true intent.

## Conclusion
While TF-IDF provides a robust and fast baseline, it struggles with the contextual nuance required to differentiate between complex bugs and feature requests. This justifies our progression to Stage 5, where we will leverage the contextual embeddings of a Transformer model.
