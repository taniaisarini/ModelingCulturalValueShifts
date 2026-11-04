"""Order-bias probe: does swapping which direction is labeled "A" vs "B" change the model's decision,
even though the underlying evidence (same indicators, same measured NYT/GDELT numbers) is unchanged?

Reuses the evidence already gathered in results/multi_question_report.json (no new measuring, no
regenerating indicators) and cases.py's case text. For each of the 10 questions, makes exactly two
new LLM calls, both forced to decide immediately (round_num=3 style instruction, so neither can stall
on "need_more_evidence"):
  - normal:  direction A/B text and evidence in their original slots
  - swapped: direction A and B text swapped, AND the evidence lists swapped to match (so whatever
             content was "A" is now presented as "B" and vice versa) -- the evidence content attached
             to each piece of text is identical either way.
If the decision tracks the SLOT (flips to the other content when order is swapped) that's a position
bias; if it tracks the CONTENT (same substantive answer either way) that argues against one.

Usage: python test_order_bias.py --run   (dry run by default, shows cache plan for question 1 only)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES  # noqa: E402
from pipeline import MODEL, aggregate_judge  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FORCE_ROUND = 3  # reuses aggregate_judge's round-3 "must decide" instruction


def swap(case, evidence):
    swapped_case = dict(case)
    swapped_case["direction_a"], swapped_case["direction_b"] = case["direction_b"], case["direction_a"]
    swapped_case["direction_labels"] = {"A": case["direction_labels"]["B"], "B": case["direction_labels"]["A"]}
    swapped_evidence = {"A": evidence["B"], "B": evidence["A"]}
    return swapped_case, swapped_evidence


def run(model, dry_run):
    report = json.load(open(RESULTS / "multi_question_report.json"))
    by_question = {r["question"]: r for r in report}

    if dry_run:
        entry = CASES[0]
        ev = by_question[entry["question"]]["evidence"]
        plan = aggregate_judge(entry["case"], ev, round_num=FORCE_ROUND, model=model, dry_run=True)
        print(f"=== aggregate_judge dry run ({entry['question']}, normal order) ===")
        print(plan["plan"])
        print("\n--- prompt (user) ---\n" + plan["prompt"])
        return

    out = []
    for entry in CASES:
        q = entry["question"]
        case, evidence = entry["case"], by_question[q]["evidence"]
        true_dir = by_question[q]["true_direction"]
        original_decision = by_question[q]["final_decision"]

        normal = aggregate_judge(case, evidence, round_num=FORCE_ROUND, model=model)
        s_case, s_evidence = swap(case, evidence)
        swapped = aggregate_judge(s_case, s_evidence, round_num=FORCE_ROUND, model=model)
        # map the swapped-order decision back onto the ORIGINAL content labels
        swapped_as_content = {"A": "B", "B": "A"}.get(swapped.get("decision"), swapped.get("decision"))

        flipped = normal.get("decision") != swapped_as_content
        print(f"{q:6s} true={true_dir}  original(3-round)={original_decision}  "
              f"forced-normal={normal.get('decision')}  forced-swapped(as content)={swapped_as_content}  "
              f"{'FLIPPED' if flipped else 'stable'}")
        out.append(dict(question=q, true_direction=true_dir, original_multi_round_decision=original_decision,
                         forced_normal_decision=normal.get("decision"), forced_normal_reasoning=normal.get("reasoning"),
                         forced_swapped_decision_raw=swapped.get("decision"),
                         forced_swapped_decision_as_content=swapped_as_content,
                         forced_swapped_reasoning=swapped.get("reasoning"), flipped=flipped))

    n_flipped = sum(r["flipped"] for r in out)
    (RESULTS / "order_bias_test.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Order-bias probe: {n_flipped}/{len(out)} questions flipped their answer when A/B order was swapped "
             "(same evidence)", "",
             "| Question | True | Original (3-round) | Forced normal | Forced swapped (as content) | Flipped |",
             "|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['question']} | {r['true_direction']} | {r['original_multi_round_decision']} | "
                      f"{r['forced_normal_decision']} | {r['forced_swapped_decision_as_content']} | "
                      f"{'yes' if r['flipped'] else 'no'} |")
    (RESULTS / "order_bias_test.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\n-> results/order_bias_test.json, results/order_bias_test.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
