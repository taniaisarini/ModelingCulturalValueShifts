"""Value-shift indicator system (self-contained in code/value_shift_indicators/; see README.md for the design).

Test case (the only one wired up so far, approved 2026-11-02): USA, question F118 "justifiable:
homosexuality" (never=1 .. always=10), wave 5 (2006) -> wave 6 (2011). The real direction/magnitude
(data/significant_changes/ks/F_wave5_to_wave6.csv) is read only to label the final report -- it is never
shown to the model, which is given both possible directions and not told which is true.

Reads: NYT Archive API metadata and GDELT raw events, both pre-pulled, via code/value_shift_indicators/data_sources.py
(paths default to this project's existing depot pulls; see that file to point elsewhere).
Writes: results/indicators.json (generated hypotheses) and results/report.json + results/report.md
(confirm/deny verdicts per indicator, real direction revealed at the end only).

Usage:
    python run_pipeline.py              # dry run: shows both prompts and cache hit/miss, calls nothing
    python run_pipeline.py --run        # actually calls the LLM (cached; code/llm_cache/)
    python run_pipeline.py --run --model gpt-4.1-mini
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import build_case, evaluate_indicator, generate_indicators, MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = Path(__file__).resolve().parent / "results"

CASE = build_case(
    country="United States",
    year_a=2006, year_b=2011,
    attitude_description=('how justifiable people consider homosexuality, on a scale from 1 (never '
                           'justifiable) to 10 (always justifiable). A nationally representative survey '
                           'asked this question at both time points.'),
    direction_a='average answer moved toward "always justifiable" (attitudes became more accepting).',
    direction_b='average answer moved toward "never justifiable" (attitudes became less accepting).',
    direction_labels={"A": "attitudes became more accepting of homosexuality",
                       "B": "attitudes became less accepting of homosexuality"},
)

GROUND_TRUTH_CSV = ROOT / "data" / "significant_changes" / "ks" / "F_wave5_to_wave6.csv"


def load_ground_truth():
    with open(GROUND_TRUTH_CSV) as f:
        for row in csv.DictReader(f):
            if row["country"] == "USA" and row["question"] == "F118":
                mean_a, mean_b = float(row["mean_wave_5"]), float(row["mean_wave_6"])
                return dict(mean_wave_5=mean_a, mean_wave_6=mean_b, change=mean_b - mean_a,
                            true_direction="A" if mean_b > mean_a else "B", p_value=float(row["ks_p_value"]))
    raise ValueError("USA/F118 not found in ground-truth CSV")


def run(model, dry_run):
    if dry_run:
        plan = generate_indicators(CASE, model=model, dry_run=True)
        print("=== generate_indicators: dry run ===")
        print(plan["plan"])
        print("\n--- prompt (user) ---\n" + plan["prompt"])
        return

    indicators = generate_indicators(CASE, model=model)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "indicators.json").write_text(json.dumps(indicators, indent=2))
    print(f"Generated {len(indicators['A'])} indicators for A, {len(indicators['B'])} for B "
          f"-> results/indicators.json")

    report = []
    for letter in ("A", "B"):
        for indicator in indicators[letter]:
            report.append(evaluate_indicator(letter, indicator, CASE, model=model))

    truth = load_ground_truth()
    (RESULTS / "report.json").write_text(json.dumps({"report": report, "ground_truth": truth}, indent=2))

    lines = ["# Value-shift indicator report",
             "",
             f"Case: USA, F118 (justifiable: homosexuality), wave 5 (2006) -> wave 6 (2011).",
             f"Model never told the true direction while generating/judging indicators.",
             "",
             "| Direction | Source | Indicator | Expected | Verdict | Reasoning |",
             "|---|---|---|---|---|---|"]
    for r in report:
        lines.append(f"| {r['direction']} | {r['source']} | {r['indicator']} | {r['expected_change']} "
                      f"| **{r['verdict']}** | {r['reasoning']} |")
    lines += ["",
              "## Ground truth (revealed only here, not shown to the model)",
              f"- mean wave 5 (2006): {truth['mean_wave_5']:.3f}, mean wave 6 (2011): {truth['mean_wave_6']:.3f} "
              f"(change {truth['change']:+.3f}, p={truth['p_value']:.1e})",
              f"- true direction: **{truth['true_direction']}**"]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\n-> results/report.json, results/report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the LLM; default is a dry run")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
