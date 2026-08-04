
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GitHub Issue Intelligence",
    page_icon="🐛",
    layout="wide"
)

st.title("GitHub Issue Intelligence 🧠")
st.markdown("Automatically classify issues and detect duplicates using AI.")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of similar issues to fetch", 1, 5, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5)

st.subheader("Submit a New Issue")

with st.form("issue_form"):
    title = st.text_input(
        "Issue Title", placeholder="e.g. ValueError when calling fit()"
    )
    body = st.text_area(
        "Issue Body",
        placeholder="e.g. The application crashes with a ValueError...",
        height=150
    )
    submitted = st.form_submit_button("Analyze Issue")

if submitted:
    if not title or not body:
        st.warning("Please provide both a title and a body.")
    else:
        payload = {"title": title, "body": body}

        col1, col2 = st.columns(2)

        # 1. Classification
        with col1:
            st.markdown("### 🏷️ Classification")
            with st.spinner("Classifying..."):
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload)
                    res.raise_for_status()
                    data = res.json()

                    category = data.get("prediction", "Unknown")
                    st.success(f"**Predicted Category:** {category}")

                    probs = data.get("probabilities", {})
                    if probs:
                        st.markdown("**Probabilities:**")
                        for cat, prob in probs.items():
                            st.progress(prob, text=f"{cat}: {prob:.2%}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

        # 2. Duplicate Detection
        with col2:
            st.markdown("### 🔍 Potential Duplicates")
            with st.spinner("Searching for similar issues..."):
                try:
                    search_payload = {"title": title, "body": body, "top_k": top_k}
                    res = requests.post(f"{API_URL}/search", json=search_payload)
                    res.raise_for_status()
                    results = res.json().get("results", [])

                    found_dup = False
                    for idx, item in enumerate(results):
                        score = item["score"]
                        if score >= similarity_threshold:
                            found_dup = True
                            title_exp = f"Similar Issue #{idx+1} (Score: {score:.2f})"
                            with st.expander(title_exp, expanded=True):
                                st.write(item["text"])

                    if not found_dup:
                        st.info("No highly similar issues found.")

                except Exception as e:
                    st.error(f"Error connecting to API: {e}")
