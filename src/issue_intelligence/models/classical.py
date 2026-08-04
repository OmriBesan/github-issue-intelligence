"""Classical NLP models with TF-IDF features."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Build the standard TF-IDF vectorizer configuration for classical models."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=None,
    )


def build_logistic_regression() -> Pipeline:
    """Logistic Regression Pipeline."""
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])


def build_linear_svc() -> Pipeline:
    """Linear Support Vector Machine Pipeline."""
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LinearSVC(class_weight="balanced", random_state=42, max_iter=1000, dual=False)),
    ])


def build_sgd_classifier() -> Pipeline:
    """SGD Classifier Pipeline."""
    # Using modified_huber as a linear text loss, or log_loss
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, max_iter=1000)),
    ])


def build_multinomial_nb() -> Pipeline:
    """Multinomial Naive Bayes Pipeline."""
    # MultinomialNB doesn't support class_weight natively
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", MultinomialNB()),
    ])


def get_top_features(pipeline: Pipeline, class_labels: list[str], top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Extract top positive features per class for linear models."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    
    # Check if classifier has coef_
    if not hasattr(clf, "coef_"):
        return {}
        
    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_
    
    # Check shape to ensure it matches class labels
    if coef.shape[0] != len(class_labels):
        return {}
        
    # Ensure clf classes match requested labels order
    if hasattr(clf, "classes_"):
        actual_classes = list(clf.classes_)
    else:
        actual_classes = class_labels

    top_features = {}
    for i, label in enumerate(class_labels):
        try:
            # Find the actual index of the label in the classifier
            clf_index = actual_classes.index(label)
            class_coef = coef[clf_index]
            top_indices = np.argsort(class_coef)[-top_n:][::-1]
            top_features[label] = [(str(feature_names[j]), float(class_coef[j])) for j in top_indices]
        except ValueError:
            pass
            
    return top_features
