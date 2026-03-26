from __future__ import annotations

import streamlit as st

from app.service import ModelVariant, get_summarizer, summarize_documents


@st.cache_resource
def preload_models() -> dict[str, object]:
    """Load both models once at app startup to speed up inference."""
    return {
        "v1": get_summarizer("v1"),
        "v2": get_summarizer("v2"),
    }


st.set_page_config(page_title="Multi-Document Summarizer")

st.title("Multi-Document Summarizer")
st.write("Summarize up to 10 text files.")

with st.spinner("Loading summarization models..."):
    preload_models()

model_variant: ModelVariant = st.radio(
    "Model Version",
    options=["v2", "v1"],
    index=0,
    horizontal=True,
    help="v2 uses context files selection. v1 uses direct file concatenation.",
)

uploaded_files = st.file_uploader(
    "Upload text files (*.txt)",
    type=["txt"],
    accept_multiple_files=True,
)

if uploaded_files and len(uploaded_files) > 10:
    st.error("Please upload a maximum of 10 files.")


def handle_summarize(files, variant: ModelVariant) -> None:
    if not files:
        st.warning("Upload at least 1 text file.")

        return

    if len(files) > 10:
        st.warning("Too many files. Maximum is 10.")

        return

    docs: list[str] = []

    for f in files:
        text = f.read().decode("utf-8", errors="ignore").strip()

        if text:
            docs.append(text)

    if not docs:
        st.error("All uploaded files are empty or unreadable.")

        return

    with st.spinner("Generating summary..."):
        try:
            summary = summarize_documents(docs, model_variant=variant)
        except Exception as e:
            st.error(f"Summarization failed: {e}")

            return

    st.subheader("Summary")

    st.write(summary)


if st.button("Summarize", type="primary"):
    handle_summarize(uploaded_files, model_variant)
