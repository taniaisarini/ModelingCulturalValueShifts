"""Shared loading for the reasoning analyses: every model's reply to the direction baseline, minus the final answer line.

    from reasoning_common import load_replies
    df = load_replies()    # one row per (model run, country, wave pair, question)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SET_DIR = ROOT / "results" / "direction_baseline" / "five_country_set"
OUT_DIR = SET_DIR / "reasoning_analysis"

# country / nationality words and years carry no reasoning; they would otherwise dominate the topics
NAME_WORDS = ["egypt", "egyptian", "egyptians", "thailand", "thai", "thais", "ukraine", "ukrainian", "ukrainians",
              "uruguay", "uruguayan", "uruguayans", "hong", "kong", "hongkong", "hong kong"]


def load_replies():
    files = {"gpt-4.1-mini": SET_DIR / "all_gpt-4.1-mini.csv"}
    for d in sorted((SET_DIR / "by_model").iterdir()):
        files[d.name.replace("__reasoning-", " (reasoning ") + (")" if "__reasoning-" in d.name else "")] = d / f"all_{d.name}.csv"
    frames = []
    for model, f in files.items():
        d = pd.read_csv(f)
        d.insert(0, "model", model)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["reasoning"] = df["response"].fillna("").map(
        lambda t: re.sub(r"\s*ANSWER:\s*\S*\s*$", "", t.strip()).strip())
    df["waves"] = df["wave_a"].astype(str) + "->" + df["wave_b"].astype(str)
    df["words"] = df["reasoning"].str.split().str.len()
    return df[df["reasoning"].str.len() > 0].reset_index(drop=True)
