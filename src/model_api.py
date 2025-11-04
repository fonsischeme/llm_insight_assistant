import os
from typing import Optional
from yaml import safe_load
import pathlib

CFG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
cfg = safe_load(open(CFG_PATH))

# OpenAI client (if available)
try:
    import openai
except Exception:
    openai = None

# HuggingFace local generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class LLM:
    def __init__(self, backend: str = "auto"):
        # backend: "openai" or "hf" or "auto"
        self.backend = backend
        if backend == "auto":
            self.backend = "openai" if os.getenv("OPENAI_API_KEY") else "hf"
        if self.backend == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.openai_model = cfg["llm"]["openai_chat_model"]
        else:
            self.hf_model_name = cfg["llm"]["hf_model"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_name)
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.backend == "openai":
            resp = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        else:
            out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
            return out.strip()
