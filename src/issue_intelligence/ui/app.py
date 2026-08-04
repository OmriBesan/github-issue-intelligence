"""Streamlit frontend for GitHub Issue Intelligence.

Run from the repository root:
    .venv\\Scripts\\python -m streamlit run src/issue_intelligence/ui/app.py

The UI communicates exclusively with the FastAPI backend via HTTP.
It does not load any model artifact, dataset, or retrieval index directly.

Backend address defaults to http://127.0.0.1:8000 and can be overridden with:
    ISSUE_API_URL=http://...  .venv\\Scripts\\python -m streamlit run ...
"""

from __future__ import annotations

import streamlit as st

from issue_intelligence.ui.client import (
    PredictResult,
    SimilarResult,
    get_health,
    get_similar,
    predict,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GitHub Issue Intelligence",
    page_icon=None,
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("GitHub Issue Intelligence")
st.markdown(
    "Classify a GitHub issue as **Bug**, **Documentation**, or **Enhancement** "
    "and retrieve lexically similar historical scikit-learn issues."
)

# ---------------------------------------------------------------------------
# Backend status
# ---------------------------------------------------------------------------

with st.expander("Backend status", expanded=True):
    health = get_health()
    if not health.reachable:
        st.error(
            f"API backend is unreachable. {health.error or ''}\n\n"
            "Start the backend with:\n\n"
            "```\n"
            ".venv\\Scripts\\python -m uvicorn "
            "issue_intelligence.api.app:app --reload\n"
            "```"
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("API", "Reachable")
        col2.metric(
            "Classifier",
            "Loaded" if health.model_loaded else "Not loaded",
        )
        col3.metric(
            "Retrieval",
            "Loaded" if health.retrieval_loaded else "Not loaded",
        )
        if health.status != "ok":
            st.warning(
                "The classifier is not loaded. Predictions will return HTTP 503. "
                "Ensure the model artifact exists at "
                "`models/classical/final_linear_svc.joblib`."
            )
        if not health.retrieval_loaded:
            st.info(
                "The retrieval artifact is not loaded. Similar-issue results will "
                "be unavailable. The classifier still works independently."
            )

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.subheader("Issue input")

with st.form("issue_form"):
    title = st.text_input(
        "Title",
        placeholder=(
            "e.g. RandomForestClassifier raises an error "
            "when fitting sparse input"
        ),
    )
    body = st.text_area(
        "Body",
        placeholder=(
            "e.g. Calling fit with a sparse matrix produces an unexpected ValueError."
        ),
        height=140,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        top_k = st.slider(
            "Number of similar issues", min_value=1, max_value=10, value=5
        )
    with col_right:
        label_filter_choice = st.selectbox(
            "Label filter",
            options=["All", "Bug", "Documentation", "Enhancement"],
        )

    submitted = st.form_submit_button("Analyze issue", type="primary")

# ---------------------------------------------------------------------------
# Input validation and API calls
# ---------------------------------------------------------------------------

if submitted:
    title_stripped = (title or "").strip()
    body_stripped = (body or "").strip()

    if not title_stripped and not body_stripped:
        st.error("Please provide at least a title or a body before submitting.")
        st.stop()

    title_val = title_stripped or None
    body_val = body_stripped or None
    filter_val = label_filter_choice if label_filter_choice != "All" else None

    # Run both API calls; failures are independent
    pred_result: PredictResult = predict(title=title_val, body=body_val)
    sim_result: SimilarResult = get_similar(
        title=title_val,
        body=body_val,
        top_k=top_k,
        label_filter=filter_val,
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Classification result
    # -----------------------------------------------------------------------

    st.subheader("Classification result")

    if pred_result.ok:
        st.success(f"**Predicted label: {pred_result.predicted_label}**")

        col_a, col_b = st.columns(2)
        col_a.metric("Model", pred_result.model_name)
        col_b.metric("Decision margin", f"{pred_result.decision_margin:.4f}")

        st.markdown("**Raw LinearSVC decision scores**")
        st.caption(
            "These scores are raw LinearSVC decision function values. "
            "They are **not probabilities** and are "
            "**not calibrated confidence percentages**. "
            "The class with the highest score is the predicted class. "
            "The decision margin is the highest score minus the "
            "second-highest score."
        )

        # Display scores from the API response — labels from API, not hard-coded
        scores = pred_result.decision_scores
        if scores:
            # Sort descending by score value for readability
            sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            for label, score in sorted_scores:
                marker = " ← predicted" if label == pred_result.predicted_label else ""
                st.text(f"  {label}: {score:+.6f}{marker}")
    else:
        # Classification failed but retrieval may still work
        if pred_result.status_code == 503:
            st.warning(
                "Classifier is unavailable (HTTP 503). "
                "Ensure the model artifact is present and the backend has loaded it."
            )
        elif pred_result.status_code == 422:
            st.warning(f"Validation error: {pred_result.error}")
        else:
            st.error(
                f"Classification failed. {pred_result.error or ''} "
                f"(status {pred_result.status_code or 'N/A'})"
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Similar-issue results
    # -----------------------------------------------------------------------

    st.subheader("Similar historical issues")
    st.caption(
        "Ranked by **TF-IDF cosine similarity** to the query. "
        "Higher values indicate greater lexical similarity. "
        "Similarity is **not a probability**. "
        "Similar issues are **not necessarily duplicates**."
    )

    if sim_result.ok:
        if not sim_result.results:
            st.info(
                "No similar issues found. This usually means the query "
                "contains no terms that appear in the training vocabulary "
                "(out-of-vocabulary query)."
            )
        else:
            filter_label = (
                f"filtered to **{filter_val}**" if filter_val else "all labels"
            )
            st.caption(
                f"Showing top {len(sim_result.results)} results "
                f"({filter_label}, {sim_result.indexed_issue_count:,} issues indexed)."
            )
            for i, issue in enumerate(sim_result.results, start=1):
                date_str = (issue.created_at or "")[:10]  # YYYY-MM-DD
                st.markdown(
                    f"**{i}. [#{issue.issue_number} — {issue.title}]({issue.url})**  \n"
                    f"Label: `{issue.target_label}` | "
                    f"Created: {date_str} | "
                    f"Cosine similarity: `{issue.similarity_score:.4f}`"
                )
    else:
        if sim_result.status_code == 503:
            st.warning(
                "Similar-issue retrieval is unavailable (HTTP 503). "
                "The classification result above is still valid."
            )
        else:
            st.error(
                f"Retrieval failed. {sim_result.error or ''} "
                f"(status {sim_result.status_code or 'N/A'})"
            )

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Limitations"):
    st.markdown(
        """
- The classifier was trained on scikit-learn GitHub issues only.
  It may not generalise to issues from other repositories.
- Decision scores are uncalibrated. They cannot be interpreted as probabilities.
- Retrieval uses lexical TF-IDF similarity, not semantic embedding search.
  Issues with similar vocabulary are surfaced regardless of conceptual similarity.
- Retrieved issues are examples of similar wording or topics,
  not confirmed duplicates.
"""
    )
