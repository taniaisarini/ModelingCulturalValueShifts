"""Improved ad-hoc analysis helpers for direct (non-pipeline) hypothesis checking against NYT/GDELT,
built after round 1 of manual experiments turned up concrete problems with the simpler data_sources.py
approach:
  - NYT matching only checked headline+abstract+keyword tags; lead_paragraph adds real recall.
  - A raw %-change on 1-30 matches is meaningless noise; report a two-proportion z-test p-value instead.
  - GDELT's total event volume moved ~10x between 2006/2011 (coverage growth, not real-world growth),
    and NYT's own total article count moved ~-45% the OTHER way (also not a real content change,
    confirmed even across the aggregated monthly counts) -- report both the raw-total-normalized rate
    AND a narrower baseline-normalized rate every time, never just one.
  - A fabricated GDELT actor-type code ("REL") doesn't exist in this data; REAL_GDELT_ACTOR_TYPES below
    is the verified list (checked against an actual month's distinct values).
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_sources import GDELT_DIR, NYT_DIR  # noqa: E402
import json
import operator
from functools import reduce

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

REAL_GDELT_ACTOR_TYPES = {"GOV", "BUS", "EDU", "COP", "JUD", "CVL", "MED", "LEG", "MIL", "MNC",
                          "HLH", "LAB", "CRM", "ELI", "IGO", "OPP", "UAF", "NGO", "IMG", "AGR",
                          "REB", "INS", "SPY", "HRI", "REF", "ENV", "UIS", "RAD", "SET", "SEP"}
CIVIC_BASELINE_CODES = ["01", "02", "10", "11", "12", "14"]  # statement/appeal/demand/disapprove/reject/protest


def load_nyt_texts(year):
    """All text per doc for one year, text = lowercased headline+abstract+lead_paragraph+tags, loaded
    once so many hypothesis tests don't each re-read 12 JSON files. Moved here from (now archived)
    test_language_hypotheses.py 2026-11-05 since the lexicon-hypothesis pipeline (test_lexicon_hypotheses.py)
    depends on it -- this is shared infrastructure, not part of either experiment specifically."""
    texts = []
    for month in range(1, 13):
        path = NYT_DIR / f"{year}-{month:02d}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())["response"]["docs"]
        for doc in docs:
            if doc.get("document_type") != "article":
                continue  # multimedia/audio entries are ~textless; their growing share (2% -> 11%
                          # from 2006 to 2011) mechanically depresses every rate if left in
            headline = (doc.get("headline") or {}).get("main") or ""
            tags = " ".join((kw.get("name") or "") for kw in (doc.get("keywords") or []))
            texts.append(f"{headline} {doc.get('abstract') or ''} {doc.get('lead_paragraph') or ''} {tags}".lower())
    return texts


def load_ground_truth():
    """{question: True/False}, True = mean increased ("A" in the case-builder convention), for every
    USA wave5->6 question. Moved here from (now archived) test_language_hypotheses.py 2026-11-05 --
    shared ground-truth lookup, used by the lexicon-hypothesis pipeline's accuracy checks."""
    truth = {}
    for path in (ROOT / "data" / "significant_changes" / "ks").glob("*_wave5_to_wave6.csv"):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["country"] == "USA":
                    truth[r["question"]] = float(r["mean_wave_6"]) > float(r["mean_wave_5"])
    return truth


def two_proportion_z(x1, n1, x2, n2):
    """Two-proportion z-test. Returns (p1, p2, z, p_value). None if either n is 0."""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1, p2, 0.0, 1.0)
    z = (p2 - p1) / se
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return (p1, p2, z, p_value)


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def nyt_rate2(year, keywords, sections=None, max_examples=3):
    """Like data_sources.nyt_year_rate, but also checks lead_paragraph (more recall)."""
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
            text = f"{headline} {doc.get('abstract') or ''} {doc.get('lead_paragraph') or ''}".lower()
            tags = [(kw.get("name") or "").lower() for kw in (doc.get("keywords") or [])]
            hit = any(k in text for k in keywords_l) or any(k in tag for tag in tags for k in keywords_l)
            if hit:
                matches += 1
                if len(examples) < max_examples:
                    examples.append(headline or (doc.get("abstract") or "")[:80])
    return dict(matches=matches, total=total, examples=examples)


def gdelt_counts2(year, event_codes, actor_types=None, actor_name_contains=None, country="US"):
    """Like data_sources.gdelt_year_stats, but always returns both the all-events total AND the
    civic-baseline total (CIVIC_BASELINE_CODES) in one pass, plus validates actor_types against the
    real vocabulary."""
    if actor_types:
        bad = set(actor_types) - REAL_GDELT_ACTOR_TYPES
        if bad:
            raise ValueError(f"Not real GDELT actor types: {bad}")
    cols = ["EventCode", "AvgTone", "GoldsteinScale"]
    if actor_types:
        cols += ["Actor1Type1Code", "Actor2Type1Code"]
    if actor_name_contains:
        cols += ["Actor1Name", "Actor2Name"]
    cols = list(dict.fromkeys(cols))

    matches = total = civic_total = 0
    tone_sum = goldstein_sum = 0.0
    for month in range(1, 13):
        path = GDELT_DIR / f"country={country}" / f"year={year}" / f"month={month:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=cols)
        total += len(df)
        civic_total += int(df["EventCode"].str[:2].isin(CIVIC_BASELINE_CODES).sum())

        mask = reduce(operator.or_, (df["EventCode"].str.startswith(c) for c in event_codes))
        if actor_types:
            mask &= (df["Actor1Type1Code"].isin(actor_types) | df["Actor2Type1Code"].isin(actor_types))
        if actor_name_contains:
            mask &= reduce(operator.or_, (
                df["Actor1Name"].str.contains(s, case=False, na=False)
                | df["Actor2Name"].str.contains(s, case=False, na=False)
                for s in actor_name_contains))
        matches += int(mask.sum())
        tone_sum += df.loc[mask, "AvgTone"].sum()
        goldstein_sum += df.loc[mask, "GoldsteinScale"].sum()

    return dict(matches=matches, total=total, civic_total=civic_total,
                tone=(tone_sum / matches) if matches else None,
                goldstein=(goldstein_sum / matches) if matches else None)


def report_nyt(label, keywords, year_a=2006, year_b=2011, sections=None):
    a = nyt_rate2(year_a, keywords, sections=sections)
    b = nyt_rate2(year_b, keywords, sections=sections)
    test = two_proportion_z(a["matches"], a["total"], b["matches"], b["total"])
    print(f"[NYT] {label}  kw={keywords} sections={sections}")
    print(f"   {a['matches']}/{a['total']} ({a['matches']/a['total']*1000:.2f}/1k) {year_a} -> "
          f"{b['matches']}/{b['total']} ({b['matches']/b['total']*1000:.2f}/1k) {year_b}"
          f"{f'  p={test[3]:.3f}' if test else ''}")
    if b["examples"]:
        print(f"   e.g. {year_b}: {b['examples'][:2]}")
    return dict(year_a=a, year_b=b, test=test)


def report_gdelt(label, event_codes, year_a=2006, year_b=2011, **kw):
    a = gdelt_counts2(year_a, event_codes, **kw)
    b = gdelt_counts2(year_b, event_codes, **kw)
    rate_a_all, rate_b_all = a["matches"] / a["total"] * 1000, b["matches"] / b["total"] * 1000
    rate_a_civ = a["matches"] / a["civic_total"] * 1000 if a["civic_total"] else None
    rate_b_civ = b["matches"] / b["civic_total"] * 1000 if b["civic_total"] else None
    test = two_proportion_z(a["matches"], a["total"], b["matches"], b["total"])
    print(f"[GDELT] {label}  codes={event_codes} {kw}")
    print(f"   vs all events:    {rate_a_all:.3f}/1k {year_a} -> {rate_b_all:.3f}/1k {year_b}"
          f"{f'  p={test[3]:.3f}' if test else ''}")
    if rate_a_civ is not None and rate_b_civ is not None:
        print(f"   vs civic baseline: {rate_a_civ:.3f}/1k {year_a} -> {rate_b_civ:.3f}/1k {year_b}")
    if a["tone"] is not None:
        print(f"   tone: {a['tone']:.2f} -> {b['tone']:.2f}   goldstein: {a['goldstein']:.2f} -> {b['goldstein']:.2f}")
    return dict(year_a=a, year_b=b, test=test)
