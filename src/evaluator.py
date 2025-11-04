from typing import Dict
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from yaml import safe_load
import pathlib
import os

PROMPTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"
prompts = safe_load(open(PROMPTS_PATH))

# LLM evaluator (uses OpenAI via model_api if available)
try:
    from src.model_api import LLM
    llm_available = True
except Exception:
    llm_available = False

# Local embedder for semantic similarity
from src.embeddings import EmbeddingsClient
emb_client = EmbeddingsClient(provider="local")

def semantic_similarity_eval(reference_texts: list[str], summary_text: str) -> Dict[str, float]:
    # compute embedding of summary and references, take max cosine across references
    docs = reference_texts
    texts = docs + [summary_text]
    embs = emb_client.embed(texts)
    ref_embs = np.array(embs[:-1])
    sum_emb = np.array(embs[-1]).reshape(1, -1)
    sims = cosine_similarity(sum_emb, ref_embs).flatten()
    # return mean & max
    return {"mean_similarity": float(np.mean(sims)), "max_similarity": float(np.max(sims))}

def llm_rubric_eval(summary_text: str) -> Dict:
    """
    Uses LLM-based rubric. Requires OpenAI API available in LLM class.
    If OpenAI key not present, raises RuntimeError.
    """
    if not llm_available:
        raise RuntimeError("LLM evaluator requires model_api availability (OpenAI or HF).")
    # Use OpenAI backend if available
    llm = LLM(backend="openai" if os.getenv("OPENAI_API_KEY") else "hf")
    prompt = prompts["eval_rubric"] + "\n\nSummary:\n" + summary_text
    out = llm.generate(prompt, max_new_tokens=200)
    # Attempt to parse JSON out of output
    try:
        return json.loads(out)
    except Exception:
        # fallback: return raw text under 'raw'
        return {"raw": out}
