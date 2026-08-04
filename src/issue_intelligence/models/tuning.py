"""Candidate configurations and selection logic for hyperparameter tuning."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# TF-IDF base configuration (Stage 3A default)
# ---------------------------------------------------------------------------

BASE_TFIDF = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": None,
}


def _build_tfidf(**overrides: Any) -> TfidfVectorizer:
    params = {**BASE_TFIDF, **overrides}
    return TfidfVectorizer(**params)


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    model_type: str  # "SGDClassifier" or "LinearSVC"
    phase: int       # 1 or 2
    config_id: str
    clf_params: dict[str, Any]
    tfidf_params: dict[str, Any] = field(default_factory=dict)

    def build_pipeline(self) -> Pipeline:
        tfidf = _build_tfidf(**self.tfidf_params)
        if self.model_type == "SGDClassifier":
            clf = SGDClassifier(
                random_state=42, max_iter=1000, **self.clf_params
            )
        elif self.model_type == "LinearSVC":
            clf = LinearSVC(
                random_state=42, max_iter=2000, dual=False, **self.clf_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return Pipeline([("tfidf", tfidf), ("clf", clf)])

    @property
    def label(self) -> str:
        tfidf_label = (
            f"ngram={self.tfidf_params.get('ngram_range', BASE_TFIDF['ngram_range'])}_"
            f"mindf={self.tfidf_params.get('min_df', BASE_TFIDF['min_df'])}_"
            f"maxf={self.tfidf_params.get('max_features', 'None')}"
        ) if self.tfidf_params else "base_tfidf"
        return f"{self.config_id} | {tfidf_label}"


def _sgd_phase1() -> list[Candidate]:
    """Phase 1: vary SGD classifier settings with base TF-IDF."""
    candidates = []
    for loss in ["log_loss", "modified_huber", "hinge"]:
        for alpha in [1e-5, 1e-4, 1e-3]:
            cid = f"SGD_p1_{loss}_a{alpha}_balanced"
            candidates.append(Candidate(
                model_type="SGDClassifier",
                phase=1,
                config_id=cid,
                clf_params={"loss": loss, "alpha": alpha, "class_weight": "balanced"},
            ))
    # Also try unweighted with best-expected alpha
    for alpha in [1e-4, 1e-3]:
        cid = f"SGD_p1_log_a{alpha}_unweighted"
        candidates.append(Candidate(
            model_type="SGDClassifier",
            phase=1,
            config_id=cid,
            clf_params={"loss": "log_loss", "alpha": alpha, "class_weight": None},
        ))
    return candidates


def _sgd_phase2(best_clf_params: dict[str, Any]) -> list[Candidate]:
    """Phase 2: vary TF-IDF settings around the best SGD classifier."""
    candidates = []
    tfidf_variants = [
        {"ngram_range": (1, 1), "min_df": 2,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": None},   # base
        {"ngram_range": (1, 2), "min_df": 3,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 5,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": 40000},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": 80000},
        {"ngram_range": (1, 3), "min_df": 2,  "max_features": None},
        {"ngram_range": (1, 3), "min_df": 3,  "max_features": 80000},
    ]
    for tv in tfidf_variants:
        ng = tv["ngram_range"]
        mf = tv["max_features"] or "None"
        cid = f"SGD_p2_ng{ng}_mindf{tv['min_df']}_maxf{mf}"
        candidates.append(Candidate(
            model_type="SGDClassifier",
            phase=2,
            config_id=cid,
            clf_params=best_clf_params,
            tfidf_params=tv,
        ))
    return candidates


def _lsvc_phase1() -> list[Candidate]:
    """Phase 1: vary LinearSVC C and class_weight with base TF-IDF."""
    candidates = []
    for C in [0.1, 0.3, 1.0, 3.0]:
        cid = f"LSVC_p1_C{C}_balanced"
        candidates.append(Candidate(
            model_type="LinearSVC",
            phase=1,
            config_id=cid,
            clf_params={"C": C, "class_weight": "balanced"},
        ))
    for C in [0.3, 1.0]:
        cid = f"LSVC_p1_C{C}_unweighted"
        candidates.append(Candidate(
            model_type="LinearSVC",
            phase=1,
            config_id=cid,
            clf_params={"C": C, "class_weight": None},
        ))
    return candidates


def _lsvc_phase2(best_clf_params: dict[str, Any]) -> list[Candidate]:
    """Phase 2: vary TF-IDF settings around the best LinearSVC classifier."""
    candidates = []
    tfidf_variants = [
        {"ngram_range": (1, 1), "min_df": 2,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": None},   # base
        {"ngram_range": (1, 2), "min_df": 3,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 5,  "max_features": None},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": 40000},
        {"ngram_range": (1, 2), "min_df": 2,  "max_features": 80000},
        {"ngram_range": (1, 3), "min_df": 2,  "max_features": None},
        {"ngram_range": (1, 3), "min_df": 3,  "max_features": 80000},
    ]
    for tv in tfidf_variants:
        ng = tv["ngram_range"]
        mf = tv["max_features"] or "None"
        cid = f"LSVC_p2_ng{ng}_mindf{tv['min_df']}_maxf{mf}"
        candidates.append(Candidate(
            model_type="LinearSVC",
            phase=2,
            config_id=cid,
            clf_params=best_clf_params,
            tfidf_params=tv,
        ))
    return candidates


def get_phase1_candidates(model_type: str) -> list[Candidate]:
    if model_type == "SGDClassifier":
        return _sgd_phase1()
    if model_type == "LinearSVC":
        return _lsvc_phase1()
    raise ValueError(f"Unknown model type: {model_type}")


def get_phase2_candidates(model_type: str, best_clf_params: dict[str, Any]) -> list[Candidate]:
    if model_type == "SGDClassifier":
        return _sgd_phase2(best_clf_params)
    if model_type == "LinearSVC":
        return _lsvc_phase2(best_clf_params)
    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    candidate: Candidate
    fold_index: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    training_time_s: float
    prediction_time_s: float
    vocabulary_size: int


@dataclass
class AggregatedResult:
    candidate: Candidate
    mean_macro_f1: float
    std_macro_f1: float
    mean_weighted_f1: float
    mean_accuracy: float
    mean_training_time_s: float
    mean_prediction_time_s: float
    mean_vocabulary_size: float
    fold_results: list[FoldResult]


def aggregate_fold_results(
    candidate: Candidate,
    fold_results: list[FoldResult],
) -> AggregatedResult:
    """Aggregate per-fold metrics into a single AggregatedResult."""
    macro_f1s = [fr.macro_f1 for fr in fold_results]
    return AggregatedResult(
        candidate=candidate,
        mean_macro_f1=statistics.mean(macro_f1s),
        std_macro_f1=statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0,
        mean_weighted_f1=statistics.mean(fr.weighted_f1 for fr in fold_results),
        mean_accuracy=statistics.mean(fr.accuracy for fr in fold_results),
        mean_training_time_s=statistics.mean(fr.training_time_s for fr in fold_results),
        mean_prediction_time_s=statistics.mean(fr.prediction_time_s for fr in fold_results),
        mean_vocabulary_size=statistics.mean(fr.vocabulary_size for fr in fold_results),
        fold_results=fold_results,
    )


def select_best(results: list[AggregatedResult]) -> AggregatedResult:
    """Select the best configuration using tie-breaking rules.

    Primary: highest mean Macro F1.
    Tie-breaking:
      1. Lower std_macro_f1 (prefer stable configs).
      2. Simpler TF-IDF (unigrams < bigrams; no max_features limit preferred).
      3. Faster training time.
    """
    if not results:
        raise ValueError("No results to select from.")

    def sort_key(r: AggregatedResult) -> tuple:
        # Complexity proxy: bigrams more complex than unigrams; phase 2 tfidf variants
        ng_range = r.candidate.tfidf_params.get("ngram_range", BASE_TFIDF["ngram_range"])
        tfidf_complexity = ng_range[1]  # 1 for unigrams, 2 for bigrams, 3 for trigrams
        return (
            -round(r.mean_macro_f1, 4),  # primary: higher is better (negated)
            round(r.std_macro_f1, 4),     # tiebreak 1: lower std is better
            tfidf_complexity,             # tiebreak 2: simpler TF-IDF
            r.mean_training_time_s,       # tiebreak 3: faster training
        )

    return sorted(results, key=sort_key)[0]
