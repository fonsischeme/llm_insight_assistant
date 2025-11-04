from src.model_api import LLM
from yaml import safe_load
import pathlib

PROMPTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "config" / "prompts.yaml"
prompts = safe_load(open(PROMPTS_PATH))

class Summarizer:
    def __init__(self, backend="auto"):
        self.llm = LLM(backend=backend)

    def summarize(self, docs: list[str]) -> str:
        # chunk and feed prompt
        joined = "\n\n".join(docs)
        prompt = prompts["summary_prompt"] + "\n\nText:\n" + joined
        return self.llm.generate(prompt, max_new_tokens=400)

    def executive_report(self, themes_text: str) -> str:
        prompt = prompts["executive_prompt"] + "\n\nThemes:\n" + themes_text
        return self.llm.generate(prompt, max_new_tokens=240)
