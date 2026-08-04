from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    """
    Creates and returns a configured TfidfVectorizer.

    Args:
        max_features: Maximum number of features (vocabulary size) to keep.
        ngram_range: The lower and upper boundary of the range of n-values for
            different n-grams to be extracted.
        min_df: Ignore terms that have a document frequency strictly lower than
            this threshold.
        max_df: Ignore terms that have a document frequency strictly higher than
            this threshold.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
    )
