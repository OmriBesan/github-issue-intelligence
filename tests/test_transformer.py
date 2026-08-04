from sklearn.preprocessing import LabelEncoder

from issue_intelligence.models.transformer import prepare_dataset


def test_prepare_dataset():
    X = ["text1", "text2"]
    y = ["Bug", "Enhancement"]

    le = LabelEncoder()
    ds, fitted_le = prepare_dataset(X, y, le)

    assert len(ds) == 2
    assert "text" in ds.column_names
    assert "label" in ds.column_names
    assert len(fitted_le.classes_) == 2
