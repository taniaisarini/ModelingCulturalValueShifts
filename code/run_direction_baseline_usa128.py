"""Direct direction-prediction baseline (no hypotheses, no NYT/GDELT data) for the same 128 USA wave5->6
significant changes used in the language-hypothesis experiment (code/value_shift_indicators/), run with
claude-haiku-5-5 -- same style of baseline already run for EGY/THA/UKR/URY/HKG with gpt-4.1-mini etc.
(run_direction_baseline_set.py), just for USA with Claude.

IMPORTANT: the 128 questions were selected by the OLDER WVS_dataloader.py KS-test significance criterion
(data/significant_changes/ks/*.csv, p<0.05 on raw per-respondent values), not by change_detection.py's
criterion that direction_baseline.py's own prepare_direction_baseline() normally gates on (only 13/128
pass that one -- confirmed by dry-running prepare_direction_baseline with questions=<the 128>). Per an
explicit 2026-11-04 user decision, this script bypasses that gate: it builds truth directly from the KS
csv files (mean_wave_6 vs mean_wave_5) for all 128 codes, and reuses direction_baseline.py's prompt
templates/build_prompt() (unchanged, already-approved) and parse_answer() unchanged.

    python code/run_direction_baseline_usa128.py            # dry run: builds prompts, shows cache plan
    python code/run_direction_baseline_usa128.py --run      # sends requests to the LLM
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_baseline import (COUNTRY_NAMES, FAMILY, LLM_PARAMS, RESULTS_DIR, build_prompt,  # noqa: E402
                                load_question_info, parse_answer, _template)
from fast_llm import plan_llm, run_llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = Path("/home/tchakrab/.claude/jobs/5c5383c4/tmp/usa_128_questions.json")
OUT_DIR = RESULTS_DIR / "usa_128_set"
MODEL = "claude-haiku-5-5"
YEARS_A, YEARS_B = "2006", "2011"   # USA wave5/wave6 fieldwork years (bulk_cases.FIELDWORK_YEARS)


def load_ks_truth(questions):
    """{code: (mean_wave_5, mean_wave_6, change_magnitude)} for USA wave5->6, from the KS-test
    significance csvs (same source test_language_hypotheses.py's load_ground_truth() reads)."""
    want = set(questions)
    out = {}
    for path in glob.glob(str(ROOT / "data" / "significant_changes" / "ks" / "*_wave5_to_wave6.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["country"] == "USA" and r["question"] in want:
                    out[r["question"]] = (float(r["mean_wave_5"]), float(r["mean_wave_6"]), float(r["change_magnitude"]))
    return out


def load_formats():
    fmt = {}
    with open(ROOT / "data" / "question_categories.csv") as f:
        for r in csv.DictReader(f):
            fmt[r["code"]] = r["format"]
    return fmt


def build_truth_row(code, fmt, mean_a, mean_b, magnitude):
    family = FAMILY[fmt]
    signed_change = mean_b - mean_a
    if family == "ordered":
        truth = "toward_high" if signed_change > 0 else "toward_low"
    elif family == "share":
        truth = "increase" if signed_change > 0 else "decrease"
    else:
        raise ValueError(f"{code}: family {family!r} (option) unexpected -- the 128 are restricted to "
                          f"ordinal_scale/numeric_scale/binary/mention_flag, which only map to ordered/share.")
    return dict(code=code, format=fmt, significant=True, effect=magnitude, truth=truth, family=family)


def call_params(model):
    """claude-haiku-5-5 (and apparently this whole Claude 5 family) rejects an explicit temperature."""
    if model.startswith("claude-"):
        return dict(temperature=None, max_tokens=800)
    return dict(LLM_PARAMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    params = call_params(a.model)

    questions = json.loads(QUESTIONS_PATH.read_text())
    ks = load_ks_truth(questions)
    fmts = load_formats()
    missing_ks = [q for q in questions if q not in ks]
    if missing_ks:
        print("WARNING: no KS truth row found for:", missing_ks)

    _, parsed = load_question_info()
    country_name = COUNTRY_NAMES["USA"]

    truth_rows, reqs, metas = [], [], {}
    for code in questions:
        if code not in ks:
            continue
        mean_a, mean_b, magnitude = ks[code]
        row = build_truth_row(code, fmts[code], mean_a, mean_b, magnitude)
        truth_rows.append(row)
        family, prompt, meta = build_prompt(row, parsed, country_name, YEARS_A, YEARS_B, allow_none=False)
        assert family == row["family"]
        rid = f"USA_5_6_{code}_ks"
        reqs.append({"id": rid, "system": _template("system"), "user": prompt})
        metas[rid] = (family, prompt, meta, row)

    print(f"{len(reqs)} of {len(questions)} questions have prompts built (KS truth available for all).")
    plan = plan_llm(reqs, model=a.model, **params)
    print(f"\nmodel {a.model}, params {params}")
    print("plan:", plan)

    if not a.run:
        print("\n--- example prompt ---")
        print("--- system ---\n" + reqs[0]["system"])
        print("--- user ---\n" + reqs[0]["user"])
        print("\ndry run only: no LLM calls were made. Add --run to send them.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run_llm(reqs, model=a.model, **params)

    rows = []
    for req in reqs:
        rid = req["id"]
        family, prompt, meta, truth_row = metas[rid]
        out = res[rid]
        pred = parse_answer(out["text"], meta)
        t = truth_row["truth"]
        rows.append(dict(country="USA", wave_a=5, wave_b=6, years_a=YEARS_A, years_b=YEARS_B,
                         code=truth_row["code"], format=truth_row["format"], family=family,
                         effect=truth_row["effect"], truth=t, pred=pred, correct=(pred == t) if pred else False,
                         parsed=pred is not None, error=out["error"], response=out["text"],
                         finish_reason=out.get("finish_reason"), cached=out.get("cached"), prompt=prompt))
    df_res = pd.DataFrame(rows)
    out_path = OUT_DIR / f"USA_5_6_{a.model}_ks128.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    n = len(df_res)
    print(f"\nn={n}  parsed={df_res['parsed'].mean():.3f}  accuracy={df_res['correct'].mean():.3f}")
    print("\nby family:\n" + df_res.groupby("family")["correct"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
