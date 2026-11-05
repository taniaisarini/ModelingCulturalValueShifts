"""Round 2 of the language-hypothesis experiment (prompts/{system,user}/generate_language_hypotheses_round2.txt,
approved 2026-11-04). Scope: the 30 USA wave5->6 questions that tied in round 1
(results/language_hypotheses_tested.json, predicted_direction == "tie") -- 4 pure 0-0 (no hypothesis
confirmed either way) and 26 genuine splits (equal confirmed votes for A and B). Each gets its own prompt
fed back its round-1 hypotheses and their real measured verdicts (confirm/deny/inconclusive, rates,
p-values -- not the true survey direction), asking for 5 NEW hypotheses per question.

Usage:
    python run_language_hypotheses_round2.py                 # dry run: lists questions + cache plan
    python run_language_hypotheses_round2.py --run            # actually call the LLM (cached; code/llm_cache/)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_llm import plan_llm, run_llm  # noqa: E402
from bulk_cases import build_generic_case  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"
RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "claude-haiku-5-5"


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def _hypotheses_block(hyps):
    lines = []
    for i, h in enumerate(hyps, 1):
        lines.append(f'{i}. [Direction {h.get("direction", "?")}, expected {h.get("expected_change", "?")}] '
                     f'"{h["hypothesis"]}"')
        if h.get("tested"):
            p = h["p_value"]
            lines.append(f'   Result: rate {{year_a}}={h["rate_a"]}, rate {{year_b}}={h["rate_b"]}, '
                         f'p={p:.4g} -> {h["verdict"]}' if p is not None else
                         f'   Result: rate {{year_a}}={h["rate_a"]}, rate {{year_b}}={h["rate_b"]}, '
                         f'p=N/A -> {h["verdict"]}')
        else:
            lines.append("   Result: not tested (no concrete marker extracted)")
    return "\n".join(lines)


def _votes_summary(votes):
    a, b = votes.get("A", 0), votes.get("B", 0)
    if a == 0 and b == 0:
        return "None of the hypotheses were confirmed in either direction."
    return f"As many confirmed hypotheses pointed toward Direction A as toward Direction B ({a} each)."


def find_tied_questions():
    rows = json.load(open(RESULTS / "language_hypotheses_tested.json"))
    return [r for r in rows if r["predicted_direction"] == "tie"]


def run(model, dry_run):
    tied = find_tied_questions()
    system = load_prompt("system", "generate_language_hypotheses_round2")
    user_template = load_prompt("user", "generate_language_hypotheses_round2")

    requests, row_by_id = [], {}
    for row in tied:
        q = row["question"]
        case = build_generic_case(q, "USA")
        block = _hypotheses_block(row["hypotheses"]).format(year_a=case["year_a"], year_b=case["year_b"])
        user = user_template.format(**case, hypotheses_block=block, votes_summary=_votes_summary(row["votes"]))
        rid = f"USA_{q}_r2"
        requests.append({"id": rid, "system": system, "user": user})
        row_by_id[rid] = q

    print(f"{len(tied)} tied questions.")
    if dry_run:
        plan = plan_llm(requests, model=model, temperature=1.0, max_tokens=1500)
        print("plan:", plan)
        print("\n--- example prompt (first question) ---")
        print(requests[0]["user"])
        return

    res = run_llm(requests, model=model, temperature=1.0, max_tokens=1500)

    out = []
    n_ok, n_fail = 0, 0
    for rid, q in row_by_id.items():
        r = res[rid]
        if r["error"]:
            n_fail += 1
            out.append(dict(country="USA", question=q, error=r["error"]))
            continue
        try:
            parsed = json.loads(_strip_fences(r["text"]))
        except json.JSONDecodeError:
            n_fail += 1
            out.append(dict(country="USA", question=q, error="bad JSON", raw=r["text"]))
            continue
        n_ok += 1
        out.append(dict(country="USA", question=q, hypotheses=parsed.get("hypotheses", [])))

    path = RESULTS / "language_hypotheses_round2.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{n_ok} succeeded, {n_fail} failed. -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
