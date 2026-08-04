from issue_intelligence.models.retrieval import IssueRetriever


def test_issue_retriever():
    retriever = IssueRetriever(model_name="all-MiniLM-L6-v2")

    corpus = [
        "The app crashes when I click the save button",
        "Please add a dark mode feature",
        "The application closes unexpectedly upon saving"
    ]

    retriever.embed_corpus(corpus)

    # Search for something similar to the crash
    query = "Whenever I try to save, the program dies"
    results = retriever.search(query, top_k=2)

    assert len(results) == 2
    # The first or second should ideally be the crash-related ones
    assert results[0]["score"] > 0.0
    assert "crash" in results[0]["text"] or "closes" in results[0]["text"]
