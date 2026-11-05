"""Configurable multi-round feedback loop for the country/time-blind lexicon-hypothesis experiment
(prompts/{system,user}/generate_lexicon_hypotheses_blind_feedback.txt -- stays blind to country/time
the same way generation does; shown and approved before use). Mirrors
run_lexicon_hypotheses_feedback.py exactly except for which prompt/round-1 file it uses.

Starts from results/lexicon_hypotheses_blind_bulk_tested.json ("round 1"). A question is "tied" if
confirm-only voting reaches no majority under EITHER the presence-fraction or the words-per-100k
metric. For every question still tied going into a round, feeds back ALL of its hypotheses from every
prior round (lexica, topic, kind, measured verdict under both metrics -- never the true direction,
and still never any country/time/publication info) and asks for 5 NEW hypotheses, same four-kind
schema. Repeats up to --max-rounds (default 5). No forced final-round decision.

Usage:
    python run_lexicon_hypotheses_blind_feedback.py                    # dry run, shows round 2's plan
    python run_lexicon_hypotheses_blind_feedback.py --max-rounds 3      # cap at 3 rounds total
    python run_lexicon_hypotheses_blind_feedback.py --run               # actually call the LLM
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
from test_lexicon_hypotheses import (  # noqa: E402
    score_hypothesis, tally_votes, effective_verdicts, load_nyt_texts, YEAR_A, YEAR_B)

PROMPTS = Path(__file__).resolve().parent / "prompts"
RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 3000


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def load_round1():
    data = json.load(open(RESULTS / "lexicon_hypotheses_blind_bulk_tested.json"))
    return {row["question"]: [dict(h, round=1) for h in row["hypotheses"]] for row in data}


def is_tied(hyps):
    tally = tally_votes(hyps)
    return tally["pred_presence"] == "tie" or tally["pred_words"] == "tie"


def _fmt_lexica(h):
    kind = h["kind"]
    parts = []
    if h.get("topic"):
        parts.append(f"topic={h['topic']}")
    if kind.startswith("trend"):
        parts.append(f"lexicon={h.get('lexicon')}")
        if h.get("expected_change"):
            parts.append(f"expected={h['expected_change']}")
    else:
        parts.append(f"early={h.get('lexicon_early')}")
        parts.append(f"late={h.get('lexicon_late')}")
    return ", ".join(parts)


def _hypotheses_block(hyps):
    lines = []
    for i, h in enumerate(hyps, 1):
        lines.append(f"{i}. [round {h.get('round', '?')}, {h['kind']}, direction {h.get('direction', '?')}] "
                      f"{_fmt_lexica(h)}")
        if h.get("tested"):
            pv, wv = effective_verdicts(h)
            lines.append(f"   Result: presence={pv}, words={wv}")
        else:
            lines.append(f"   Result: not tested ({h.get('untestable_reason', 'unknown')})")
    return "\n".join(lines)


def _votes_summary(tally):
    vp, vw = tally["votes_presence"], tally["votes_words"]
    return (f"Current tally -- presence: A={vp['A']} B={vp['B']} ({tally['pred_presence']}); "
            f"words: A={vw['A']} B={vw['B']} ({tally['pred_words']}).")


def run_round(round_num, tied_questions, history, texts_a, texts_b, model, dry_run):
    system = load_prompt("system", "generate_lexicon_hypotheses_blind_feedback")
    user_template = load_prompt("user", "generate_lexicon_hypotheses_blind_feedback")

    requests, case_by_id = [], {}
    for q in tied_questions:
        case = build_generic_case(q, "USA")
        tally = tally_votes(history[q])
        block = _hypotheses_block(history[q])
        # Only attitude_description/direction_a/direction_b -- no country/year_a/year_b/window.
        user = user_template.format(attitude_description=case["attitude_description"],
                                    direction_a=case["direction_a"], direction_b=case["direction_b"],
                                    hypotheses_block=block, votes_summary=_votes_summary(tally))
        rid = f"USA_{q}_r{round_num}"
        requests.append({"id": rid, "system": system, "user": user})
        case_by_id[rid] = q

    if dry_run:
        plan = plan_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)
        print(f"  round {round_num}: {len(tied_questions)} tied questions, plan: {plan}")
        if requests:
            print(f"\n--- example round {round_num} prompt ({case_by_id[requests[0]['id']]}) ---")
            print(requests[0]["user"])
        return

    res = run_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)
    raw_out, tested_out, n_ok, n_fail = [], [], 0, 0
    for rid, q in case_by_id.items():
        r = res[rid]
        if r["error"]:
            n_fail += 1
            raw_out.append(dict(country="USA", question=q, error=r["error"]))
            continue
        try:
            parsed = json.loads(_strip_fences(r["text"]))
        except json.JSONDecodeError:
            n_fail += 1
            raw_out.append(dict(country="USA", question=q, error="bad JSON", raw=r["text"]))
            continue
        n_ok += 1
        new_hyps = parsed.get("hypotheses", [])
        scored = [dict(score_hypothesis(h, texts_a, texts_b), round=round_num) for h in new_hyps]
        history[q].extend(scored)
        raw_out.append(dict(country="USA", question=q, hypotheses=new_hyps))
        tested_out.append(dict(country="USA", question=q, hypotheses=scored))
    print(f"  round {round_num}: {n_ok} succeeded, {n_fail} failed")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"lexicon_hypotheses_blind_feedback_round{round_num}.json").write_text(json.dumps(raw_out, indent=2))
    (RESULTS / f"lexicon_hypotheses_blind_feedback_round{round_num}_tested.json").write_text(
        json.dumps(tested_out, indent=2))


def run(model, max_rounds, dry_run):
    history = load_round1()
    tied = [q for q, hyps in history.items() if is_tied(hyps)]
    print(f"round 1 (already run): {len(history)} questions, {len(tied)} tied (either metric).")

    texts_a = texts_b = None
    if not dry_run:
        print("loading NYT 2006/2011 into memory...")
        texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)

    for round_num in range(2, max_rounds + 1):
        if not tied:
            print(f"no questions still tied -- stopping before round {round_num}.")
            break
        run_round(round_num, tied, history, texts_a, texts_b, model, dry_run)
        if dry_run:
            print("\n(dry run only shows round 2's plan -- later rounds' tie lists depend on round 2's "
                  "real results, so they can't be previewed without sending it.)")
            return
        tied = [q for q in tied if is_tied(history[q])]
        print(f"  -> {len(tied)} still tied after round {round_num}.")

    (RESULTS / "lexicon_hypotheses_blind_feedback_full_history.json").write_text(json.dumps(history, indent=2))

    final = []
    for q, hyps in history.items():
        tally = tally_votes(hyps)
        final.append(dict(question=q, n_hypotheses=len(hyps), **tally))
    (RESULTS / "lexicon_hypotheses_blind_feedback_final.json").write_text(json.dumps(final, indent=2))
    n_tied_final = sum(1 for f in final if f["pred_presence"] == "tie" or f["pred_words"] == "tie")
    print(f"\nFinal: {len(final)} questions, {n_tied_final} still tied (either metric) after all rounds run.")
    print(f"-> {RESULTS / 'lexicon_hypotheses_blind_feedback_final.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, a.max_rounds, dry_run=not a.run)
