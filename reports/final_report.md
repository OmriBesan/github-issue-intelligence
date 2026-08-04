# Final Report — GitHub Issue Intelligence

## 1. Problem Statement and Motivation
Open-source maintainers face a significant triage burden. Large repositories like `scikit-learn` receive thousands of issues that must be manually categorized into Bugs, Enhancements, or Documentation requests. Furthermore, users often submit duplicate issues, wasting maintainer time and splitting discussions. 

Our goal was to automate this process by building a Machine Learning system capable of:
1. Categorizing incoming issues.
2. Detecting semantically similar issues to flag potential duplicates.

## 2. Dataset Description
We collected 12,190 historical issues from the `scikit-learn` repository via the GitHub API. 
After rigorous auditing, filtering pull requests, and standardizing labels, we established a high-quality dataset of 5,710 labeled issues mapping to three distinct classes:
- **Bug** (39.8%)
- **Enhancement** (35.7%)
- **Documentation** (24.5%)

## 3. Model Progression and Evaluation

We established a baseline using Classical NLP techniques before advancing to Deep Learning models.

### Classical Baselines (TF-IDF)
We trained Logistic Regression, Linear SVC, and Naive Bayes models on TF-IDF features.
- **Results**: Logistic Regression achieved a strong baseline performance, demonstrating that simple linear separators can effectively partition the TF-IDF vector space for short text.
- **Limitations**: TF-IDF relies on exact word overlap and cannot capture deep semantic nuance (e.g., distinguishing between "I need to fix this" (Enhancement) and "The code is broken, fix it" (Bug)).

### Transformer Fine-Tuning (DistilBERT)
To address the limitations of TF-IDF, we fine-tuned `distilbert-base-uncased`.
- **Results**: The Transformer model significantly outperformed the classical baselines, particularly on the ambiguous boundary between Bugs and Enhancements.

## 4. Theoretical Connections (Course Curriculum)

### Generalization, Overfitting, and Regularization
In evaluating the transition from Logistic Regression to DistilBERT, we must consider the theoretical bounds of generalization. 

The **VC Dimension** and **Rademacher Complexity** of transformer neural networks are vastly higher than those of linear models (like our TF-IDF Logistic Regression). This means Transformers possess a significantly greater capacity to "shatter" the training set (i.e., memorize the noise), drastically increasing the risk of overfitting. 

According to **PAC (Probably Approximately Correct) Learning** theory, achieving tight generalization bounds with such a high-capacity model requires either an exponentially larger dataset or the application of strong inductive biases and regularization. Because our dataset is relatively small (~5,700 samples), we heavily relied on regularization techniques during Transformer fine-tuning (e.g., Dropout, Weight Decay, and Early Stopping) to constrain the model's effective hypothesis space and ensure empirical risk minimization translated to true generalization on the test set.

### Margin-Based Learning
Our classical `LinearSVC` baseline successfully maximized the decision margin between classes in the high-dimensional TF-IDF space, illustrating the robustness of max-margin classifiers in sparse text data.

## 5. Semantic Search and Duplicate Detection
Instead of using lexical matching, we implemented **Retrieval-Augmented Generation (RAG)** principles. We vectorized historical issues using a pre-trained `sentence-transformer` and used Cosine Similarity to surface the most semantically related historical issues, effectively identifying duplicates regardless of keyword overlap.

## 6. Deployment and Architecture
We exposed the models using a `FastAPI` REST backend and built an interactive `Streamlit` UI. The architecture is designed to be containerized with Docker and directly integrated into GitHub via Webhooks.

## 7. Limitations and Future Work
- **Temporal Drift**: Software domains evolve. A model trained on 2018 issues may struggle with 2026 vocabulary (e.g., "Array API"). Continuous learning loops are necessary.
- **Context Window**: Extremely long stack traces may exceed DistilBERT's 512-token limit. We mitigated this by truncating, but advanced hierarchical models could be explored.
