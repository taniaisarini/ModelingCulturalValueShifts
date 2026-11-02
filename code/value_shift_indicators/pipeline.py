"""Core logic for the value-shift indicator system (see run_pipeline.py for the CLI).

Per question, three kinds of step (both LLM steps go through code/fast_llm.py, cached in llm_cache/):
  1. generate_indicators (1 LLM call): given a value shift described WITHOUT naming which direction
     actually happened, produce >=5 indicators per direction of what we'd expect to see in NYT/GDELT data.
  2. measure_indicator (no LLM call, data_sources.py): pull the real NYT/GDELT numbers for one indicator.
  3. aggregate_judge (1 LLM call per round, up to 3 rounds): given ALL indicators checked so far for both
     directions and what was measured, decide which direction the evidence favors, or ask for up to 5 more
     indicators to check (round 3 must decide).
"""
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
JUDGE_PARAMS = dict(temperature=0.0, max_tokens=900)

GDELT_CATEGORIES = ("01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, "
                    "05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, "
                    "08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, "
                    "14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, "
                    "19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE")


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def build_case(country, year_a, year_b, attitude_description, direction_a, direction_b, direction_labels):
    """`direction_labels` is a short {"A": ..., "B": ...} phrase used only for display inside prompts,
    e.g. {"A": "attitudes became more accepting of X", "B": "attitudes became less accepting of X"}."""
    return dict(country=country, year_a=year_a, year_b=year_b, attitude_description=attitude_description,
                direction_a=direction_a, direction_b=direction_b, direction_labels=direction_labels,
                window_start=year_a - 1, window_end=year_b + 1)


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


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


def measure_indicator(indicator, case):
    """No LLM call: pull the real pre/post numbers for one indicator from NYT or GDELT."""
    if indicator["source"] == "nyt":
        keywords = indicator["nyt_keywords"]
        matches_a, total_a, rate_a, _ = nyt_year_rate(case["year_a"], keywords)
        matches_b, total_b, rate_b, examples_b = nyt_year_rate(case["year_b"], keywords)
        return dict(source="nyt", year_a_matches=matches_a, year_a_total=total_a, year_a_rate=rate_a,
                    year_b_matches=matches_b, year_b_total=total_b, year_b_rate=rate_b, examples=examples_b)
    elif indicator["source"] == "gdelt":
        roots = indicator["gdelt_event_roots"]
        matches_a, total_a, rate_a, tone_a, _ = gdelt_year_stats(case["year_a"], roots)
        matches_b, total_b, rate_b, tone_b, _ = gdelt_year_stats(case["year_b"], roots)
        metrics = dict(source="gdelt", year_a_matches=matches_a, year_a_total=total_a, year_a_rate=rate_a,
                       year_b_matches=matches_b, year_b_total=total_b, year_b_rate=rate_b)
        if indicator.get("gdelt_tone", "no_prediction") != "no_prediction":
            metrics["tone_year_a"], metrics["tone_year_b"] = tone_a, tone_b
        return metrics
    raise ValueError(f"Unknown indicator source: {indicator['source']!r}")


def _format_measured(metrics, case):
    unit = "matching articles" if metrics["source"] == "nyt" else "matching events"
    line = (f"{metrics['year_a_matches']}/{metrics['year_a_total']} {unit} "
            f"({metrics['year_a_rate']:.2f} per 1,000) in {case['year_a']} vs. "
            f"{metrics['year_b_matches']}/{metrics['year_b_total']} "
            f"({metrics['year_b_rate']:.2f} per 1,000) in {case['year_b']}")
    if metrics.get("tone_year_a") is not None and metrics.get("tone_year_b") is not None:
        line += f"; avg tone {metrics['tone_year_a']:.2f} vs. {metrics['tone_year_b']:.2f}"
    return line


def _format_indicator_block(items, case):
    lines = []
    for i, item in enumerate(items, 1):
        ind = item["indicator_obj"]
        lines.append(f'{i}. "{ind["indicator"]}" (source: {ind["source"]}, predicted: {ind["expected_change"]})\n'
                      f'   Measured: {_format_measured(item["metrics"], case)}')
    return "\n".join(lines) + "\n"


def aggregate_judge(case, evidence, round_num, model=MODEL, dry_run=False):
    """One call: judge all indicators gathered so far. Returns the parsed decision dict, or if dry_run,
    {"plan": ..., "prompt": ...}."""
    round_instruction = (
        "If the evidence below is not yet sufficient to decide, you may request up to 5 additional "
        "specific indicators to check instead of deciding now."
        if round_num < 3 else
        'This is the final round. You must answer "A" or "B" -- "need_more_evidence" will not be honored.'
    )
    system = load_prompt("system", "aggregate_judge")
    user = load_prompt("user", "aggregate_judge").format(
        round_num=round_num, round_instruction=round_instruction,
        direction_a_indicators=_format_indicator_block(evidence["A"], case),
        direction_b_indicators=_format_indicator_block(evidence["B"], case),
        **case)
    req_id = f"aggregate_round{round_num}"
    req = [{"id": req_id, "system": system, "user": user}]
    if dry_run:
        return {"plan": plan_llm(req, model=model, **JUDGE_PARAMS), "prompt": user}
    res = run_llm(req, model=model, **JUDGE_PARAMS)
    result = res[req_id]
    if result["error"]:
        raise RuntimeError(f"aggregate_judge call failed (round {round_num}): {result['error']}")
    return json.loads(_strip_fences(result["text"]))


def run_case(case, model=MODEL):
    """Full loop for one question: generate -> measure -> judge (up to 3 rounds). Returns
    {"evidence": {"A": [...], "B": [...]}, "rounds": [...], "final_decision": "A"|"B"|"undecided"}."""
    indicators = generate_indicators(case, model=model)
    evidence = {"A": [], "B": []}
    for letter in ("A", "B"):
        for ind in indicators[letter]:
            evidence[letter].append({"indicator_obj": ind, "metrics": measure_indicator(ind, case)})

    rounds, final = [], None
    for round_num in range(1, 4):
        result = aggregate_judge(case, evidence, round_num, model=model)
        rounds.append(dict(round=round_num, decision=result.get("decision"), reasoning=result.get("reasoning"),
                            requested=result.get("requested_evidence", [])))
        if result.get("decision") in ("A", "B"):
            final = result["decision"]
            break
        if round_num == 3:
            final = "undecided"
            break
        for req_ind in result.get("requested_evidence", []):
            letter = req_ind.get("direction")
            if letter not in ("A", "B") or req_ind.get("source") not in ("nyt", "gdelt"):
                continue  # model returned something we can't look up (bad direction/source); skip it
            try:
                metrics = measure_indicator(req_ind, case)
            except (KeyError, ValueError):
                continue  # e.g. missing nyt_keywords/gdelt_event_roots for the given source
            evidence[letter].append({"indicator_obj": req_ind, "metrics": metrics})
    return dict(evidence=evidence, rounds=rounds, final_decision=final)
