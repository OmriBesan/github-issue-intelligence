import pytest
from sklearn.pipeline import Pipeline

from issue_intelligence.models.classical import build_classical_pipeline


def test_build_classical_pipeline_valid_models():
    models = ["logistic_regression", "linear_svc", "sgd", "naive_bayes"]
    for m in models:
        pipeline = build_classical_pipeline(model_type=m)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "tfidf"
        assert pipeline.steps[1][0] == "clf"


def test_build_classical_pipeline_invalid_model():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_classical_pipeline(model_type="invalid_model")


def test_pipeline_fit_predict():
    pipeline = build_classical_pipeline(model_type="logistic_regression")

    X_train = [
        "crash fix error bug",
        "documentation typo bug",
        "new feature enhancement bug",
        "another fail issue"
    ]
    y_train = ["Bug", "Documentation", "Enhancement", "Bug"]

    pipeline.fit(X_train, y_train)

    X_test = ["fix the bug", "add new feature"]
    preds = pipeline.predict(X_test)

    assert len(preds) == 2
    assert preds[0] in ["Bug", "Documentation", "Enhancement"]
