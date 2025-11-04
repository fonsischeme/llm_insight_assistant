import chromadb
from chromadb.config import Settings
import os
from yaml import safe_load
import pathlib

CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
cfg = safe_load(open(CFG_PATH))

def get_chroma_client(persist_directory=None):
    persist_directory = persist_directory or cfg["vector_store"]["persist_directory"]
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    )
    client = chromadb.Client(settings=settings)
    return client

def create_or_get_collection(name=None):
    name = name or cfg["vector_store"]["collection_name"]
    client = get_chroma_client()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name)
