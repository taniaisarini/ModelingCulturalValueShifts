"""Deterministic metric pulls for the value-shift indicator system (see README in this directory).

Reads two independent, pre-pulled local datasets (not modified here):
  - NYT Archive API monthly metadata: <NYT_DIR>/<year>-<month>.json (headline/abstract, no article body)
  - GDELT row-level events, US partition: <GDELT_DIR>/country=US/year=<year>/month=<month>.parquet

Paths default to this project's existing depot pulls, overridable with the VSI_NYT_DIR / VSI_GDELT_DIR
environment variables (kept separate from the shared code/config.py since this directory is meant to be
self-contained).
"""
import json
import os
from pathlib import Path

import pandas as pd

NYT_DIR = Path(os.environ.get("VSI_NYT_DIR", "/depot/dgoldwas/data/tchakrab/nyt-archive/raw"))
GDELT_DIR = Path(os.environ.get("VSI_GDELT_DIR", "/depot/dgoldwas/data/tchakrab/gdelt-data/raw-events"))


def nyt_year_rate(year, keywords, max_examples=3):
    """Fraction of that year's NYT articles whose headline or abstract contains any of `keywords`
    (case-insensitive substring match), per 1,000 articles, plus a few example headlines that matched.
    Returns (matches, total, rate_per_1000, example_headlines)."""
    keywords = [k.lower() for k in keywords]
    matches, total, examples = 0, 0, []
    for month in range(1, 13):
        path = NYT_DIR / f"{year}-{month:02d}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())["response"]["docs"]
        total += len(docs)
        for doc in docs:
            headline = (doc.get("headline") or {}).get("main") or ""
            abstract = doc.get("abstract") or ""
            text = f"{headline} {abstract}".lower()
            if any(k in text for k in keywords):
                matches += 1
                if len(examples) < max_examples:
                    examples.append(headline or abstract[:80])
    rate = (matches / total * 1000) if total else 0.0
    return matches, total, rate, examples


def gdelt_year_stats(year, event_roots, country="US"):
    """US-partitioned GDELT events in `year` whose EventCode's 2-digit root is in `event_roots`,
    per 1,000 events, plus avg tone of matched events vs. all events that year.
    Returns (matches, total, rate_per_1000, avg_tone_matched, avg_tone_all)."""
    roots = set(event_roots)
    matches, total, tone_sum_matched, tone_sum_all = 0, 0, 0.0, 0.0
    for month in range(1, 13):
        path = GDELT_DIR / f"country={country}" / f"year={year}" / f"month={month:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["EventCode", "AvgTone"])
        total += len(df)
        tone_sum_all += df["AvgTone"].sum()
        hit = df["EventCode"].str[:2].isin(roots)
        matches += int(hit.sum())
        tone_sum_matched += df.loc[hit, "AvgTone"].sum()
    rate = (matches / total * 1000) if total else 0.0
    avg_tone_matched = (tone_sum_matched / matches) if matches else None
    avg_tone_all = (tone_sum_all / total) if total else None
    return matches, total, rate, avg_tone_matched, avg_tone_all
