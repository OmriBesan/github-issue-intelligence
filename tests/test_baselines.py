import pytest

from issue_intelligence.baselines import MajorityClassBaseline, RuleBasedBaseline


def test_majority_class_baseline():
    model = MajorityClassBaseline()

    # Train data with mostly "Bug"
    X_train = ["issue 1", "issue 2", "issue 3"]
    y_train = ["Bug", "Enhancement", "Bug"]

    model.fit(X_train, y_train)
    assert model.majority_class_ == "Bug"

    # Predict should always return "Bug"
    X_test = ["issue 4", "issue 5"]
    preds = model.predict(X_test)
    assert preds == ["Bug", "Bug"]


def test_majority_class_baseline_empty_y():
    model = MajorityClassBaseline()
    with pytest.raises(ValueError):
        model.fit(["x"], [])


def test_majority_class_baseline_unfitted():
    model = MajorityClassBaseline()
    with pytest.raises(RuntimeError):
        model.predict(["test"])


def test_rule_based_baseline():
    model = RuleBasedBaseline(default_class="Bug")

    # fit does nothing
    model.fit(None, None)

    X_test = [
        "Please fix this crash immediately",  # Bug
        "We should add support for python 3.14",  # Enhancement
        "There is a typo in the readme file",  # Documentation
        "This is an ambiguous text with no keywords",  # Default (Bug)
        "Add a new feature but fix the bug first"  # Tie (Enhancement and Bug=2)
    ]

    preds = model.predict(X_test)

    # First: 'crash', 'fix' -> Bug
    # Second: 'add', 'support' -> Enhancement
    # Third: 'readme', 'typo' -> Documentation
    # Fourth: none -> Bug
    # Fifth: tie between Enhancement(2) and Bug(2)
    #   -> Enhancement (order of tie breaking: Doc > Enhancement > Bug)

    assert preds == ["Bug", "Enhancement", "Documentation", "Bug", "Enhancement"]
