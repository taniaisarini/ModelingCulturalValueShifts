"""Run the direction baseline for the chosen five countries on waves 5->6 and 6->7.

Countries: Egypt, Thailand, Ukraine, Uruguay, Hong Kong. Only questions with a significant change are used.

    python code/run_direction_baseline_set.py            # dry run: builds every prompt, prints counts, NO LLM calls
    python code/run_direction_baseline_set.py --run      # sends all requests to the LLM as one batch

Results go to results/direction_baseline/five_country_set/ (one CSV per country and wave pair plus a combined CSV,
each row with the full response text). Every LLM call is also stored, with its complete response, in the
llm_cache/ (see llm_cache.py): calls already in the cache are never sent again, so an interrupted or repeated
run only sends what is missing.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_baseline import (LLM_PARAMS, RESULTS_DIR, assemble_direction_results,  # noqa: E402
                                load_countries_waves, prepare_direction_baseline, score)
from fast_llm import plan_llm, run_llm  # noqa: E402

COUNTRIES = ["EGY", "THA", "UKR", "URY", "HKG"]
WAVE_PAIRS = [(5, 6), (6, 7)]
OUT_ROOT = RESULTS_DIR / "five_country_set"


def call_params(model, reasoning):
    """LLM settings for a model, kept as close to the gpt-4.1-mini baseline (temperature 0, 250 tokens) as the model allows.

    Reasoning models (gpt-5*) reject temperature and max_tokens, so they get the default temperature and a larger
    max_completion_tokens (reasoning tokens count against it). reasoning: 'off' = lowest effort ('minimal'),
    'on' = 'medium'. Non-reasoning models ignore the flag. Returns (kwargs for run_llm/plan_llm, label for output).
    """
    if model.startswith("gpt-5"):
        effort, cap = ("minimal", 1000) if reasoning == "off" else ("medium", 6000)
        return (dict(temperature=None, max_tokens=None,
                     extra={"extra_body": {"max_completion_tokens": cap, "reasoning_effort": effort}}),   # extra_body: installed openai SDK is too old to know these
                f"{model}__reasoning-{reasoning}")
    return dict(LLM_PARAMS), model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="actually call the LLM (default is a dry run)")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--flip", action="store_true", help="reverse the order of the answer options (A/B swapped, option list reversed)")
    ap.add_argument("--reasoning", choices=["off", "on"], default="off", help="only for reasoning models (gpt-5*)")
    a = ap.parse_args()

    params, label = call_params(a.model, a.reasoning)
    if a.flip:
        OUT_DIR = RESULTS_DIR / "five_country_set_flipped" / "by_model" / label
    else:
        OUT_DIR = OUT_ROOT if a.model == "gpt-4.1-mini" else OUT_ROOT / "by_model" / label   # baseline files stay where they were
    df = load_countries_waves(COUNTRIES, sorted({w for pair in WAVE_PAIRS for w in pair}))
    preps = [prepare_direction_baseline(c, wa, wb, df, flip=a.flip) for c in COUNTRIES for wa, wb in WAVE_PAIRS]

    print("\nquestions with a significant change, by country and wave pair:")
    summary = pd.DataFrame([
        dict(country=p["country"], waves=f"{p['wave_a']}->{p['wave_b']}", years=f"{p['years_a']}->{p['years_b']}",
             questions=len(p["reqs"]), **p["truth"].assign(f=p["truth"]["format"].map(
                 lambda x: {"ordinal_scale": "ordered", "count": "ordered", "binary": "share",
                            "mention_flag": "share", "knowledge": "share"}.get(x, "option"))
                            )["f"].value_counts().to_dict())
        for p in preps]).fillna(0)
    print(summary.to_string(index=False))
    total = sum(len(p["reqs"]) for p in preps)
    all_reqs = [r for p in preps for r in p["reqs"]]
    plan = plan_llm(all_reqs, model=a.model, **params)
    print(f"\ntotal requests: {total} (model {a.model}, {params})")
    print(f"already in the cache: {plan['hits']}   would be sent to the LLM: {plan['unique_misses']}")

    if not a.run:
        print("\ndry run only: no LLM calls were made. Add --run to send them.")
        p = preps[0]
        print(f"\nexample request ({p['reqs'][0]['id']}):\n--- system ---\n{p['reqs'][0]['system']}"
              f"\n--- user ---\n{p['reqs'][0]['user']}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run_llm(all_reqs, model=a.model, **params)

    results = []
    for p in preps:
        r = assemble_direction_results(p, res)
        r.to_csv(OUT_DIR / f"{p['country']}_{p['wave_a']}_{p['wave_b']}_{label}.csv", index=False)
        results.append(r)
    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(OUT_DIR / f"all_{label}.csv", index=False)
    print(f"\nwrote {len(results)} CSVs and all_{label}.csv to {OUT_DIR}")
    print("\noverall by question family:\n" + score(allres).to_string())
    allres["waves"] = allres["wave_a"].astype(str) + "->" + allres["wave_b"].astype(str)
    print("\naccuracy by country and wave pair:\n" +
          allres.groupby(["country", "waves"])["correct"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
