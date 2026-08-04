from issue_intelligence.features import get_tfidf_vectorizer


def test_get_tfidf_vectorizer_defaults():
    vec = get_tfidf_vectorizer()
    assert vec.max_features == 10000
    assert vec.ngram_range == (1, 2)
    assert vec.min_df == 2
    assert vec.max_df == 0.95
    assert vec.stop_words == "english"


def test_get_tfidf_vectorizer_custom():
    vec = get_tfidf_vectorizer(max_features=500, ngram_range=(1, 1))
    assert vec.max_features == 500
    assert vec.ngram_range == (1, 1)


def test_vectorizer_fit_transform():
    vec = get_tfidf_vectorizer(min_df=1) # min_df=1 for small test
    X = ["this is a test document", "this document is another test"]
    X_vec = vec.fit_transform(X)

    # Should create a sparse matrix
    assert X_vec.shape[0] == 2
    # Vocabulary should contain words except stop words (e.g., 'this', 'is', 'a')
    assert len(vec.vocabulary_) > 0
