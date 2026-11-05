"""Generic round-N (N>=3) step of the language-hypothesis feedback loop. Reuses the round-2 prompt
unchanged (prompts/{system,user}/generate_language_hypotheses_round2.txt, approved 2026-11-04) -- same
template, just fed a longer cumulative hypotheses list each round. Rounds 1 and 2 used dedicated
scripts/files for historical reasons (run_language_hypotheses_bulk.py, run_language_hypotheses_round2.py
+ its "_extra" patch); round 3+ uses this generic, parameterized version instead of one script per round.

For each question still tied under confirm-only scoring (deny does NOT count toward the opposite
direction -- 2026-11-04 correction) after all rounds up to N-1, feeds back EVERY hypothesis proposed so
far across all prior rounds, with its measured verdict (confirm/deny/inconclusive, rates, p-value -- never
the true survey direction), and asks for 5 new ones. Questions already decided by an earlier round are
never re-queried.

Usage:
    python run_language_hypotheses_round_n.py --round 3                 # dry run: plan + example prompt
    python run_language_hypotheses_round_n.py --round 3 --run           # actually call the LLM (cached)
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


def strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def confirm_only_votes(hyps):
    votes = {"A": 0, "B": 0}
    for h in hyps:
        if h.get("verdict") == "confirm":
            d = h.get("direction", "?")
            votes[d] = votes.get(d, 0) + 1
    return votes


def predict(votes):
    if votes["A"] > votes["B"]:
        return "A"
    if votes["B"] > votes["A"]:
        return "B"
    return "tie"


def votes_summary_text(votes):
    a, b = votes.get("A", 0), votes.get("B", 0)
    if a == 0 and b == 0:
        return "None of the hypotheses were confirmed in either direction."
    return f"As many confirmed hypotheses pointed toward Direction A as toward Direction B ({a} each)."


def hypotheses_block(hyps, year_a, year_b):
    lines = []
    for i, h in enumerate(hyps, 1):
        tag = f'[round {h["round"]}] ' if "round" in h else ""
        lines.append(f'{i}. {tag}[Direction {h.get("direction", "?")}, expected {h.get("expected_change", "?")}] '
                     f'"{h["hypothesis"]}"')
        if h.get("tested"):
            p = h.get("p_value")
            pstr = f"{p:.4g}" if p is not None else "N/A"
            lines.append(f'   Result: rate {year_a}={h["rate_a"]}, rate {year_b}={h["rate_b"]}, '
                         f'p={pstr} -> {h["verdict"]}')
        else:
            lines.append("   Result: not tested (no concrete marker extracted)")
    return "\n".join(lines)


def _load_json(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else []


def load_cumulative(upto_round):
    """{question: [hyp dicts tagged with 'round']} using every hypothesis from round 1 through
    `upto_round` (inclusive) that exists on disk. Round 2's "_extra" file (historical quirk) also
    contains a few questions round 1 alone already decided (tied under the old deny-counts-as-opposite
    rule but not under confirm-only) -- those must be excluded here, or merging their round-2 votes on
    top of an already-decided round-1 tally can spuriously flip or tie a question that round 1 alone
    had resolved."""
    by_q = {}

    def add(rows, round_num, allowed=None):
        for row in rows:
            q = row["question"]
            if allowed is not None and q not in allowed:
                continue
            for h in row.get("hypotheses", []):
                by_q.setdefault(q, []).append(dict(h, round=round_num))

    r1_rows = _load_json("language_hypotheses_tested.json")
    add(r1_rows, 1)
    if upto_round >= 2:
        tied_after_1 = {r["question"] for r in r1_rows if predict(confirm_only_votes(r["hypotheses"])) == "tie"}
        add(_load_json("language_hypotheses_round2_tested.json"), 2, allowed=tied_after_1)
        add(_load_json("language_hypotheses_round2_extra_tested.json"), 2, allowed=tied_after_1)
    for n in range(3, upto_round + 1):
        add(_load_json(f"language_hypotheses_round{n}_tested.json"), n)
    return by_q


def load_true_directions():
    return {r["question"]: r["true_direction"] for r in _load_json("language_hypotheses_tested.json")}


def still_inconclusive(round_n):
    """Questions tied under confirm-only scoring using all hypotheses through round `round_n - 1`."""
    cumulative = load_cumulative(round_n - 1)
    true_dir = load_true_directions()
    return [q for q in true_dir if predict(confirm_only_votes(cumulative.get(q, []))) == "tie"], cumulative


def run(round_n, dry_run):
    questions, cumulative = still_inconclusive(round_n)
    print(f"round {round_n}: {len(questions)} questions still inconclusive going in.")
    if not questions:
        print("nothing to do.")
        return

    system = load_prompt("system", "generate_language_hypotheses_round2")
    user_template = load_prompt("user", "generate_language_hypotheses_round2")

    requests, row_by_id = [], {}
    for q in questions:
        case = build_generic_case(q, "USA")
        block = hypotheses_block(cumulative[q], case["year_a"], case["year_b"])
        votes = confirm_only_votes(cumulative[q])
        user = user_template.format(**case, hypotheses_block=block, votes_summary=votes_summary_text(votes))
        rid = f"USA_{q}_r{round_n}"
        requests.append({"id": rid, "system": system, "user": user})
        row_by_id[rid] = q

    if dry_run:
        plan = plan_llm(requests, model=MODEL, temperature=1.0, max_tokens=1500)
        print("plan:", plan)
        print("\n--- example prompt ---")
        print(requests[0]["user"])
        return

    res = run_llm(requests, model=MODEL, temperature=1.0, max_tokens=1500)

    out = []
    n_ok = n_fail = 0
    failed_qs = []
    for rid, q in row_by_id.items():
        r = res[rid]
        if r["error"]:
            n_fail += 1
            failed_qs.append(q)
            out.append(dict(country="USA", question=q, error=r["error"]))
            continue
        try:
            parsed = json.loads(strip_fences(r["text"]))
        except json.JSONDecodeError:
            n_fail += 1
            failed_qs.append(q)
            out.append(dict(country="USA", question=q, error="bad JSON", raw=r["text"]))
            continue
        n_ok += 1
        out.append(dict(country="USA", question=q, hypotheses=parsed.get("hypotheses", [])))

    print(f"{n_ok} succeeded, {n_fail} failed at max_tokens=1500.")

    if failed_qs:
        print(f"retrying {len(failed_qs)} at max_tokens=2500: {failed_qs}")
        user_by_q = {q: req["user"] for req in requests for q in [row_by_id[req["id"]]] if q in failed_qs}
        retry_requests = [{"id": f"USA_{q}_r{round_n}_retry2500", "system": system, "user": user_by_q[q]}
                           for q in failed_qs]
        retry_res = run_llm(retry_requests, model=MODEL, temperature=1.0, max_tokens=2500)
        for q, req in zip(failed_qs, retry_requests):
            r = retry_res[req["id"]]
            if r["error"]:
                print(" ", q, "still failing:", r["error"])
                continue
            try:
                parsed = json.loads(strip_fences(r["text"]))
            except json.JSONDecodeError as e:
                print(" ", q, "still bad JSON:", e)
                continue
            for row in out:
                if row.get("question") == q:
                    row.pop("error", None)
                    row.pop("raw", None)
                    row["hypotheses"] = parsed.get("hypotheses", [])
            print(" ", q, "ok on retry")

    path = RESULTS / f"language_hypotheses_round{round_n}.json"
    path.write_text(json.dumps(out, indent=2))
    n_final_ok = sum(1 for r in out if "hypotheses" in r)
    print(f"final: {n_final_ok}/{len(out)} succeeded. -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    run(a.round, dry_run=not a.run)
