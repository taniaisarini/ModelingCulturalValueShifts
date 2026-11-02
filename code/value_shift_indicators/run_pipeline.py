"""Value-shift indicator system (self-contained in code/value_shift_indicators/).

For each of the 10 USA wave5->6 (2006->2011) test cases in cases.py: an LLM generates >=5 indicators
per direction of a value shift (never told which direction is real) of what we'd expect to see in NYT
article metadata / GDELT events; code pulls the real numbers for each (data_sources.py, no LLM); then
the LLM judges which direction the evidence favors, or asks for up to 5 more indicators to check (up to
3 rounds total, must decide by round 3). The real direction (data/significant_changes/) is read only to
grade the final answer -- never shown to the model.

Usage:
    python run_pipeline.py                  # dry run: shows case[0]'s generate_indicators prompt + cache plan
    python run_pipeline.py --run             # actually run all 10 cases (cached; code/llm_cache/)
    python run_pipeline.py --run --model gpt-4.1-mini
    python run_pipeline.py --run --only F118 F121   # subset, by question code
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES  # noqa: E402
from pipeline import MODEL, generate_indicators, run_case  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def load_ground_truth(section, question):
    path = ROOT / "data" / "significant_changes" / "ks" / f"{section}_wave5_to_wave6.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["country"] == "USA" and row["question"] == question:
                mean_a, mean_b = float(row["mean_wave_5"]), float(row["mean_wave_6"])
                return dict(mean_wave_5=mean_a, mean_wave_6=mean_b, change=mean_b - mean_a,
                            p_value=float(row["ks_p_value"]))
    raise ValueError(f"USA/{question} not found in {path}")


def true_direction(entry, truth):
    up = entry["direction_if_mean_increases"]
    return up if truth["change"] > 0 else ("B" if up == "A" else "A")


def run(model, dry_run, only=None):
    cases = [c for c in CASES if only is None or c["question"] in only]

    if dry_run:
        plan = generate_indicators(cases[0]["case"], model=model, dry_run=True)
        print(f"=== generate_indicators: dry run ({cases[0]['question']}) ===")
        print(plan["plan"])
        print("\n--- prompt (user) ---\n" + plan["prompt"])
        return

    RESULTS.mkdir(exist_ok=True)
    all_results = []
    for entry in cases:
        question = entry["question"]
        print(f"=== {question} ===")
        out = run_case(entry["case"], model=model)
        truth = load_ground_truth(entry["section"], question)
        true_dir = true_direction(entry, truth)
        correct = out["final_decision"] == true_dir
        print(f"  final decision: {out['final_decision']}  (true: {true_dir}, "
              f"{'correct' if correct else 'WRONG'}, {len(out['rounds'])} round(s))")
        all_results.append(dict(question=question, final_decision=out["final_decision"], true_direction=true_dir,
                                 correct=correct, rounds_used=len(out["rounds"]), rounds=out["rounds"],
                                 evidence=out["evidence"], ground_truth=truth))

    (RESULTS / "multi_question_report.json").write_text(json.dumps(all_results, indent=2, default=str))

    n_correct = sum(r["correct"] for r in all_results)
    lines = ["# Value-shift indicator report: 10 questions", "",
             f"Model: {model}. {n_correct}/{len(all_results)} correct.", "",
             "| Question | Decision | True direction | Correct | Rounds used |",
             "|---|---|---|---|---|"]
    for r in all_results:
        lines.append(f"| {r['question']} | {r['final_decision']} | {r['true_direction']} | "
                      f"{'yes' if r['correct'] else 'no'} | {r['rounds_used']} |")
    (RESULTS / "multi_question_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> results/multi_question_report.json, results/multi_question_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually call the LLM; default is a dry run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--only", nargs="*", default=None, help="subset of question codes, e.g. F118 F121")
    a = ap.parse_args()
    run(a.model, dry_run=not a.run, only=a.only)
