import pandas as pd

def load_feedback_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df
