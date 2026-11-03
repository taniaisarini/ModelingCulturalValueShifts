"""Deterministic metric pulls for the value-shift indicator system (see README in this directory).

Reads two independent, pre-pulled local datasets (not modified here):
  - NYT Archive API monthly metadata: <NYT_DIR>/<year>-<month>.json (headline/abstract, no article body)
  - GDELT row-level events, US partition: <GDELT_DIR>/country=US/year=<year>/month=<month>.parquet

Paths default to this project's existing depot pulls, overridable with the VSI_NYT_DIR / VSI_GDELT_DIR
environment variables (kept separate from the shared code/config.py since this directory is meant to be
self-contained).
"""
import json
import operator
import os
from functools import reduce
from pathlib import Path

import pandas as pd

NYT_DIR = Path(os.environ.get("VSI_NYT_DIR", "/depot/dgoldwas/data/tchakrab/nyt-archive/raw"))
GDELT_DIR = Path(os.environ.get("VSI_GDELT_DIR", "/depot/dgoldwas/data/tchakrab/gdelt-data/raw-events"))


def nyt_year_rate(year, keywords, sections=None, max_examples=3):
    """Fraction of that year's NYT articles whose headline/abstract (substring match) or subject tags
    (`keywords[].name`, substring match) contain any of `keywords`, optionally restricted to `sections`
    (matched against `section_name` or `news_desk`), per 1,000 articles in that (possibly restricted)
    pool. Returns dict(matches, total, rate, examples)."""
    keywords_l = [k.lower() for k in keywords]
    sections_l = [s.lower() for s in sections] if sections else None
    matches, total, examples = 0, 0, []
    for month in range(1, 13):
        path = NYT_DIR / f"{year}-{month:02d}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())["response"]["docs"]
        for doc in docs:
            if sections_l is not None:
                sec = (doc.get("section_name") or "").lower()
                desk = (doc.get("news_desk") or "").lower()
                if sec not in sections_l and desk not in sections_l:
                    continue
            total += 1
            headline = (doc.get("headline") or {}).get("main") or ""
            abstract = doc.get("abstract") or ""
            text = f"{headline} {abstract}".lower()
            tags = [(kw.get("name") or "").lower() for kw in (doc.get("keywords") or [])]
            hit = any(k in text for k in keywords_l) or any(k in tag for tag in tags for k in keywords_l)
            if hit:
                matches += 1
                if len(examples) < max_examples:
                    examples.append(headline or abstract[:80])
    rate = (matches / total * 1000) if total else 0.0
    return dict(matches=matches, total=total, rate=rate, examples=examples)


def gdelt_year_stats(year, event_codes, actor_types=None, actor_name_contains=None, location_contains=None,
                      baseline_codes=None, country="US"):
    """US-partitioned GDELT events in `year` whose EventCode starts with one of `event_codes` (2-4
    digit CAMEO prefixes, e.g. "14" or "1411") AND (if given) has an Actor1Type1Code/Actor2Type1Code in
    `actor_types`, AND/OR matches substrings in `actor_name_contains` (Actor1Name/Actor2Name) and/or
    `location_contains` (ActionGeo_FullName) -- all filters AND'd together.

    The rate's denominator is, by default, every event that year (as before). Pass `baseline_codes`
    (a list of 2-digit CAMEO roots) to use a narrower comparison universe instead -- e.g. other
    political/civic-discourse categories -- so the share isn't distorted by GDELT's overall indexed
    volume changing between years for reasons unrelated to the topic at hand.

    Returns dict(matches, total [=the denominator actually used], rate, tone_matched, goldstein_matched)."""
    cols = ["EventCode", "AvgTone", "GoldsteinScale"]
    if actor_types:
        cols += ["Actor1Type1Code", "Actor2Type1Code"]
    if actor_name_contains:
        cols += ["Actor1Name", "Actor2Name"]
    if location_contains:
        cols += ["ActionGeo_FullName"]
    cols = list(dict.fromkeys(cols))

    matches = total = baseline_total = 0
    tone_sum = goldstein_sum = 0.0
    for month in range(1, 13):
        path = GDELT_DIR / f"country={country}" / f"year={year}" / f"month={month:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=cols)
        total += len(df)

        mask = reduce(operator.or_, (df["EventCode"].str.startswith(c) for c in event_codes))
        if actor_types:
            mask &= (df["Actor1Type1Code"].isin(actor_types) | df["Actor2Type1Code"].isin(actor_types))
        if actor_name_contains:
            mask &= reduce(operator.or_, (
                df["Actor1Name"].str.contains(s, case=False, na=False)
                | df["Actor2Name"].str.contains(s, case=False, na=False)
                for s in actor_name_contains))
        if location_contains:
            mask &= reduce(operator.or_, (
                df["ActionGeo_FullName"].str.contains(s, case=False, na=False) for s in location_contains))

        matches += int(mask.sum())
        tone_sum += df.loc[mask, "AvgTone"].sum()
        goldstein_sum += df.loc[mask, "GoldsteinScale"].sum()

        if baseline_codes:
            base_mask = df["EventCode"].str[:2].isin(baseline_codes)
            baseline_total += int(base_mask.sum())

    denom = baseline_total if baseline_codes else total
    rate = (matches / denom * 1000) if denom else 0.0
    return dict(matches=matches, total=denom, rate=rate,
                tone_matched=(tone_sum / matches) if matches else None,
                goldstein_matched=(goldstein_sum / matches) if matches else None)
