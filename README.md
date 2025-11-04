# LLM-Powered Insight Assistant

Turn raw text feedback into executive-ready insights. Toggle between OpenAI and a local HuggingFace model, use ChromaDB for retrieval, and evaluate outputs with both LLM rubrics and semantic similarity.

## Quickstart
1. `git clone <repo>`
2. `pip install -r requirements.txt`
3. Add OpenAI key if you want OpenAI backends:
   `export OPENAI_API_KEY=sk-...`
4. `streamlit run app.py`

## Features
- Toggleable LLM: OpenAI or local HF (default: `google/flan-t5-large`)
- Embeddings: OpenAI or SentenceTransformers local (`all-MiniLM-L6-v2`)
- Vector store: ChromaDB (DuckDB+Parquet persist)
- Evaluation: LLM-based rubric + cosine-similarity

See `config/` for prompt and model settings.
