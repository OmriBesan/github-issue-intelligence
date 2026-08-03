"""Tests for the tuning module."""

from __future__ import annotations

import pytest

from issue_intelligence.models.tuning import (
    Candidate,
    FoldResult,
    aggregate_fold_results,
    get_phase1_candidates,
    get_phase2_candidates,
    select_best,
)

# ---------------------------------------------------------------------------
# 8. Candidate configurations are generated as intended
# ---------------------------------------------------------------------------


def test_sgd_phase1_has_expected_configs() -> None:
    candidates = get_phase1_candidates("SGDClassifier")
    # Should have 3 losses × 3 alphas (balanced) + 2 unweighted = 11
    assert len(candidates) >= 9  # at least 9
    assert all(c.model_type == "SGDClassifier" for c in candidates)
    assert all(c.phase == 1 for c in candidates)


def test_lsvc_phase1_has_expected_configs() -> None:
    candidates = get_phase1_candidates("LinearSVC")
    assert len(candidates) >= 4
    assert all(c.model_type == "LinearSVC" for c in candidates)


def test_unknown_model_raises_error() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_phase1_candidates("RandomForest")


# ---------------------------------------------------------------------------
# 9. No Cartesian explosion
# ---------------------------------------------------------------------------


def test_phase2_does_not_explode() -> None:
    sgd_candidates = get_phase2_candidates(
        "SGDClassifier", {"loss": "log_loss", "alpha": 1e-4, "class_weight": "balanced"}
    )
    lsvc_candidates = get_phase2_candidates(
        "LinearSVC", {"C": 1.0, "class_weight": "balanced"}
    )
    assert len(sgd_candidates) <= 30
    assert len(lsvc_candidates) <= 30


# ---------------------------------------------------------------------------
# 10-11. Mean and standard deviation are calculated correctly
# ---------------------------------------------------------------------------


def _make_candidate(
    config_id: str = "test", model_type: str = "LinearSVC"
) -> Candidate:
    return Candidate(
        model_type=model_type,
        phase=1,
        config_id=config_id,
        clf_params={"C": 1.0, "class_weight": "balanced"},
    )


def _make_fold_result(
    candidate: Candidate, fold_index: int, macro_f1: float
) -> FoldResult:
    return FoldResult(
        candidate=candidate,
        fold_index=fold_index,
        macro_f1=macro_f1,
        weighted_f1=macro_f1,
        accuracy=macro_f1,
        training_time_s=1.0,
        prediction_time_s=0.1,
        vocabulary_size=1000,
    )


def test_mean_macro_f1_correct() -> None:
    c = _make_candidate()
    frs = [_make_fold_result(c, i, f) for i, f in enumerate([0.80, 0.90, 0.85])]
    agg = aggregate_fold_results(c, frs)
    assert abs(agg.mean_macro_f1 - 0.85) < 1e-6


def test_std_macro_f1_correct() -> None:
    import statistics

    c = _make_candidate()
    vals = [0.80, 0.90, 0.85]
    frs = [_make_fold_result(c, i, f) for i, f in enumerate(vals)]
    agg = aggregate_fold_results(c, frs)
    assert abs(agg.std_macro_f1 - statistics.stdev(vals)) < 1e-6


# ---------------------------------------------------------------------------
# 12. Tie-breaking prefers lower variance
# ---------------------------------------------------------------------------


def test_tiebreak_prefers_lower_variance() -> None:
    c_stable = _make_candidate("stable")
    c_noisy = _make_candidate("noisy")

    agg_stable = aggregate_fold_results(
        c_stable, [_make_fold_result(c_stable, i, 0.90) for i in range(3)]
    )
    agg_noisy = aggregate_fold_results(
        c_noisy,
        [
            _make_fold_result(c_noisy, 0, 0.95),
            _make_fold_result(c_noisy, 1, 0.80),
            _make_fold_result(c_noisy, 2, 0.95),
        ],
    )
    # stable has mean 0.90, noisy has mean ~0.90 but higher std
    # Round to 4dp both are 0.9000 → tiebreak should prefer stable
    best = select_best([agg_stable, agg_noisy])
    assert best.candidate.config_id == "stable"


# ---------------------------------------------------------------------------
# 13. TF-IDF is fitted inside each fold (no shared vocabulary)
# ---------------------------------------------------------------------------


def test_pipeline_build_produces_independent_pipelines() -> None:
    c = _make_candidate()
    p1 = c.build_pipeline()
    p2 = c.build_pipeline()
    # They must be different instances
    assert p1 is not p2
    assert p1.named_steps["tfidf"] is not p2.named_steps["tfidf"]


# ---------------------------------------------------------------------------
# 17. Best configuration reproduction is deterministic
# ---------------------------------------------------------------------------


def test_select_best_is_deterministic() -> None:
    c1 = _make_candidate("low")
    c2 = _make_candidate("high")
    agg1 = aggregate_fold_results(c1, [_make_fold_result(c1, 0, 0.80)])
    agg2 = aggregate_fold_results(c2, [_make_fold_result(c2, 0, 0.90)])

    best1 = select_best([agg1, agg2])
    best2 = select_best([agg1, agg2])
    assert best1.candidate.config_id == best2.candidate.config_id


# ---------------------------------------------------------------------------
# 18. Unknown or invalid parameters produce clear errors
# ---------------------------------------------------------------------------


def test_invalid_model_type_in_candidate_build() -> None:
    c = Candidate(
        model_type="InvalidModel",
        phase=1,
        config_id="bad",
        clf_params={"C": 1.0},
    )
    with pytest.raises(ValueError, match="Unknown"):
        c.build_pipeline()


def test_select_best_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="No results"):
        select_best([])
