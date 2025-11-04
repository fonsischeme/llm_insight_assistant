import os
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

try:
    import openai
except Exception:
    openai = None

from yaml import safe_load
import pathlib

CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
cfg = safe_load(open(CFG_PATH))

class EmbeddingsClient:
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        if provider == "local" or (provider == "auto" and cfg["embeddings"]["provider"] in ["local", "auto"]):
            model_name = cfg["embeddings"]["local_model"]
            self._local = SentenceTransformer(model_name)
        else:
            self._local = None
            if openai is None:
                raise RuntimeError("OpenAI package not available.")
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._local:
            embs = self._local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [emb.tolist() for emb in embs]
        else:
            # OpenAI embeddings
            model = cfg["embeddings"]["openai_model"]
            resp = openai.Embedding.create(input=texts, model=model)
            return [r["embedding"] for r in resp["data"]]
