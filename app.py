import streamlit as st
import pandas as pd
import os
from src.data_loader import load_feedback_csv
from src.retrieval import Retriever
from src.summarizer import Summarizer
from src.evaluator import semantic_similarity_eval, llm_rubric_eval
from src.visualizer import bar_top_themes, pie_sentiment
from yaml import safe_load
import pathlib

# Load config
CFG_PATH = pathlib.Path(__file__).resolve().parents[0] / "config" / "settings.yaml"
cfg = safe_load(open(CFG_PATH))

# Page config & theme (light/soft)
st.set_page_config(page_title="LLM Insight Assistant", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .reportview-container {
        background: #ffffff;
    }
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💡 LLM-Powered Insight Assistant")
st.caption("Turn raw text feedback into concise, executive-ready insights")

# Sidebar: settings and model toggle
st.sidebar.header("Settings")
llm_backend = st.sidebar.selectbox("LLM Backend", options=["auto", "openai", "hf"], index=0)
emb_provider = st.sidebar.selectbox("Embedding Provider", options=["auto", "openai", "local"], index=0)
collection_name = st.sidebar.text_input("Chroma collection name", value=cfg["vector_store"]["collection_name"])
top_k = st.sidebar.slider("Retrieval top-k", 1, 20, 6)
show_eval = st.sidebar.checkbox("Show evaluation", value=True)

# Upload / sample data
uploaded = st.file_uploader("Upload feedback CSV (must have 'text' column)", type=["csv"])
use_sample = False
if not uploaded:
    st.info("No file uploaded — using sample dataset.")
    sample_path = os.path.join("data", "sample_feedback.csv")
    df = load_feedback_csv(sample_path)
    use_sample = True
else:
    df = load_feedback_csv(uploaded)

st.write(f"Dataset: {len(df)} rows")
if st.checkbox("Show raw data"):
    st.dataframe(df)

# Build embeddings & index
if st.button("Build / refresh index"):
    retriever = Retriever(embedding_provider=emb_provider, collection_name=collection_name)
    ids = [str(i) for i in df.index.tolist()]
    texts = df["text"].tolist()
    with st.spinner("Indexing documents into Chroma..."):
        retriever.index_texts(ids, texts)
    st.success("Index built.")

# Query input
query = st.text_input("Ask a question (e.g., 'Summarize complaints about pricing')", value="Summarize complaints about pricing")
if st.button("Run query"):
    retriever = Retriever(embedding_provider=emb_provider, collection_name=collection_name)
    docs, dists = retriever.query(query, top_k=top_k)
    st.subheader("Retrieved examples")
    for i, d in enumerate(docs[:5]):
        st.markdown(f"> {d}  \n---  ")

    # Summarization
    summarizer = Summarizer(backend=llm_backend)
    with st.spinner("Generating theme summaries..."):
        summary = summarizer.summarize(docs)
    st.subheader("📝 Theme Summary")
    st.markdown(summary)

    with st.spinner("Generating executive report..."):
        report = summarizer.executive_report(summary)
    st.subheader("📋 Executive Report")
    st.write(report)

    # Very small heuristic to extract sentiment counts (best-effort)
    # This is placeholder — for demo only
    sentiment_counts = {"positive":0,"neutral":0,"negative":0}
    for s in ["positive","negative","neutral"]:
        sentiment_counts[s] = summary.lower().count(s)

    st.plotly_chart(pie_sentiment(sentiment_counts), use_container_width=True)

    # Evaluation
    if show_eval:
        st.subheader("🔬 Evaluation")
        sem_scores = semantic_similarity_eval(docs, summary)
        st.write("Semantic similarity:", sem_scores)
        # LLM rubric eval (only works if OpenAI key present)
        try:
            rubric = llm_rubric_eval(summary)
            st.write("LLM rubric:", rubric)
        except Exception as e:
            st.info(f"LLM rubric not available: {e}")
