from src.vectorstore import create_or_get_collection
from src.embeddings import EmbeddingsClient
import numpy as np
from typing import List

class Retriever:
    def __init__(self, embedding_provider="auto", collection_name=None):
        self.emb_client = EmbeddingsClient(provider=embedding_provider)
        self.collection = create_or_get_collection(collection_name)

    def index_texts(self, ids: List[str], texts: List[str], persist: bool = True):
        embs = self.emb_client.embed(texts)
        # add or upsert
        try:
            self.collection.add(ids=ids, documents=texts, embeddings=embs)
        except Exception:
            # fallback upsert
            self.collection.upsert(ids=ids, documents=texts, embeddings=embs)
        # persist handled by chroma client settings

    def query(self, query_text: str, top_k: int = 8):
        q_emb = self.emb_client.embed([query_text])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
        docs = res["documents"][0]
        dists = res["distances"][0]
        return docs, dists
