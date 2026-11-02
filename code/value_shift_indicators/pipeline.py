"""Core logic for the value-shift indicator system (see run_pipeline.py for the CLI, README.md for the design).

Two LLM calls (both through code/fast_llm.py, cached in llm_cache/):
  1. generate_indicators: given a value shift described WITHOUT naming which direction actually happened,
     produce >=5 indicators per direction of what we'd expect to see in NYT/GDELT data.
  2. evaluate_indicator: given one indicator plus the real NYT/GDELT numbers (pulled deterministically by
     data_sources.py), judge confirm/deny/inconclusive.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/ -> fast_llm, llm_cache
from fast_llm import plan_llm, run_llm  # noqa: E402

from data_sources import gdelt_year_stats, nyt_year_rate  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"
MODEL = "gpt-4.1-mini"
GEN_PARAMS = dict(temperature=0.0, max_tokens=2000)
JUDGE_PARAMS = dict(temperature=0.0, max_tokens=300)

GDELT_CATEGORIES = ("01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, "
                    "05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, "
                    "08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, "
                    "14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, "
                    "19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE")


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def build_case(country, year_a, year_b, attitude_description, direction_a, direction_b, direction_labels):
    """`direction_labels` is a short {"A": ..., "B": ...} phrase used only in the confirm/deny prompt,
    e.g. {"A": "attitudes became more accepting of X", "B": "attitudes became less accepting of X"}."""
    return dict(country=country, year_a=year_a, year_b=year_b, attitude_description=attitude_description,
                direction_a=direction_a, direction_b=direction_b, direction_labels=direction_labels,
                window_start=year_a - 1, window_end=year_b + 1)


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def _pct_change(a, b):
    if a == 0:
        return "n/a (zero baseline)" if b == 0 else "+inf% (from zero)"
    return f"{(b - a) / a * 100:+.0f}%"


def generate_indicators(case, model=MODEL, dry_run=False):
    """One call: >=5 indicators per direction. Returns the parsed {"A": [...], "B": [...]}, or if
    dry_run, {"plan": ..., "prompt": ...} without calling anything."""
    system = load_prompt("system", "generate_indicators")
    user = load_prompt("user", "generate_indicators").format(gdelt_categories=GDELT_CATEGORIES, **case)
    req = [{"id": "generate", "system": system, "user": user}]
    if dry_run:
        return {"plan": plan_llm(req, model=model, **GEN_PARAMS), "prompt": user}
    res = run_llm(req, model=model, **GEN_PARAMS)
    result = res["generate"]
    if result["error"]:
        raise RuntimeError(f"generate_indicators call failed: {result['error']}")
    parsed = json.loads(_strip_fences(result["text"]))
    for letter in ("A", "B"):
        if len(parsed.get(letter, [])) < 5:
            raise ValueError(f"Direction {letter} has only {len(parsed.get(letter, []))} indicators (need >=5)")
    return parsed


def _measure_nyt(indicator, case):
    keywords = indicator["nyt_keywords"]
    matches_a, total_a, rate_a, _ = nyt_year_rate(case["year_a"], keywords)
    matches_b, total_b, rate_b, examples_b = nyt_year_rate(case["year_b"], keywords)
    data_source_description = f"New York Times article headlines/abstracts, {case['year_a']} vs. {case['year_b']}"
    query_line = f"Keywords searched: {', '.join(keywords)}"
    measured_lines = (
        f"- {case['year_a']}: {matches_a} matching articles out of {total_a} total ({rate_a:.2f} per 1,000)\n"
        f"- {case['year_b']}: {matches_b} matching articles out of {total_b} total ({rate_b:.2f} per 1,000)\n"
        f"- Change: {_pct_change(rate_a, rate_b)}"
    )
    examples_block = ""
    if examples_b:
        examples_block = (f"Example matching headlines from {case['year_b']}:\n"
                           + "\n".join(f'- "{h}"' for h in examples_b) + "\n\n")
    metrics = dict(source="nyt", keywords=keywords, year_a_matches=matches_a, year_a_total=total_a,
                   year_a_rate=rate_a, year_b_matches=matches_b, year_b_total=total_b, year_b_rate=rate_b)
    return data_source_description, query_line, measured_lines, examples_block, metrics


def _measure_gdelt(indicator, case):
    roots = indicator["gdelt_event_roots"]
    matches_a, total_a, rate_a, tone_m_a, _ = gdelt_year_stats(case["year_a"], roots)
    matches_b, total_b, rate_b, tone_m_b, _ = gdelt_year_stats(case["year_b"], roots)
    data_source_description = f"GDELT US events, {case['year_a']} vs. {case['year_b']}"
    query_line = f"GDELT event categories searched (2-digit root codes): {', '.join(roots)}"
    tone_line = ""
    if indicator.get("gdelt_tone", "no_prediction") != "no_prediction" and tone_m_a is not None and tone_m_b is not None:
        tone_line = f"\n- Avg tone of matching events: {tone_m_a:.2f} ({case['year_a']}) vs. {tone_m_b:.2f} ({case['year_b']})"
    measured_lines = (
        f"- {case['year_a']}: {matches_a} matching events out of {total_a} total ({rate_a:.2f} per 1,000)\n"
        f"- {case['year_b']}: {matches_b} matching events out of {total_b} total ({rate_b:.2f} per 1,000)\n"
        f"- Change: {_pct_change(rate_a, rate_b)}{tone_line}"
    )
    metrics = dict(source="gdelt", event_roots=roots, year_a_matches=matches_a, year_a_total=total_a,
                   year_a_rate=rate_a, year_b_matches=matches_b, year_b_total=total_b, year_b_rate=rate_b,
                   tone_year_a=tone_m_a, tone_year_b=tone_m_b)
    return data_source_description, query_line, measured_lines, "", metrics


def evaluate_indicator(direction_letter, indicator, case, model=MODEL, dry_run=False):
    """Pull the real numbers for one indicator and judge confirm/deny/inconclusive."""
    source = indicator["source"]
    if source == "nyt":
        data_source_description, query_line, measured_lines, examples_block, metrics = _measure_nyt(indicator, case)
    elif source == "gdelt":
        data_source_description, query_line, measured_lines, examples_block, metrics = _measure_gdelt(indicator, case)
    else:
        raise ValueError(f"Unknown indicator source: {source!r}")

    system = load_prompt("system", "confirm_deny")
    user = load_prompt("user", "confirm_deny").format(
        direction_letter=direction_letter, direction_description=case["direction_labels"][direction_letter],
        indicator_text=indicator["indicator"], expected_change=indicator["expected_change"],
        data_source_description=data_source_description, query_description_line=query_line,
        measured_lines=measured_lines, examples_block=examples_block)
    req_id = f"confirm_{direction_letter}_{hashlib.md5(indicator['indicator'].encode()).hexdigest()[:8]}"
    req = [{"id": req_id, "system": system, "user": user}]
    if dry_run:
        return {"plan": plan_llm(req, model=model, **JUDGE_PARAMS), "prompt": user, "metrics": metrics}

    res = run_llm(req, model=model, **JUDGE_PARAMS)
    result = res[req_id]
    if result["error"]:
        raise RuntimeError(f"confirm_deny call failed for {req_id}: {result['error']}")
    verdict = json.loads(_strip_fences(result["text"]))
    return dict(direction=direction_letter, indicator=indicator["indicator"], source=source,
                expected_change=indicator["expected_change"], verdict=verdict["verdict"],
                reasoning=verdict["reasoning"], metrics=metrics)
