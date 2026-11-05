"""Country/time-blind variant of the lexicon-hypothesis experiment (prompts/{system,user}/
generate_lexicon_hypotheses_blind.txt -- new, shown to the user before use). Unlike
run_lexicon_hypotheses.py, the model is given ONLY the attitude description and the two possible
directions -- no country, no year_a/year_b, no window, no mention of New York Times. The point: the
2026-11-05 confound diagnosis found that many "confirmed" hypotheses just tracked real secular events
the model knew happened in the USA 2006-2011 (Facebook's rise, the 2008 financial crisis) rather than
anything about the attitude itself. This variant tests whether hypotheses generated blind to that
context -- from the value question alone -- do any better or worse.

The mechanical test (test_lexicon_hypotheses.py) is unchanged: hypotheses are still scored against the
real USA NYT 2006/2011 corpus, same as every other run. Only generation is blind; testing isn't (and
can't be -- the corpus the hypotheses are checked against is fixed).

Scope: same 5-question pilot used throughout (F118/A001/A168A/A165/A124_02), for a first look before
deciding whether to scale to the full 128-question set.

Usage:
    python run_lexicon_hypotheses_blind.py                 # dry run: lists cases + cache plan, no calls
    python run_lexicon_hypotheses_blind.py --run            # actually call the LLM (cached)
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
MAX_TOKENS = 3000

PILOT_QUESTIONS = ["F118", "A001", "A168A", "A165", "A124_02"]


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def run(model, dry_run):
    system = load_prompt("system", "generate_lexicon_hypotheses_blind")
    user_template = load_prompt("user", "generate_lexicon_hypotheses_blind")

    requests, case_by_id = [], {}
    for q in PILOT_QUESTIONS:
        case = build_generic_case(q, "USA")
        rid = f"USA_{q}"
        # Only attitude_description/direction_a/direction_b are used -- country/year_a/year_b/window
        # are deliberately NOT passed to the template at all.
        user = user_template.format(attitude_description=case["attitude_description"],
                                    direction_a=case["direction_a"], direction_b=case["direction_b"])
        requests.append({"id": rid, "system": system, "user": user})
        case_by_id[rid] = q

    print(f"{len(PILOT_QUESTIONS)} pilot questions (blind to country/time): {PILOT_QUESTIONS}")
    if dry_run:
        plan = plan_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)
        print("plan:", plan)
        print("\n--- example prompt (first question) ---")
        print(requests[0]["user"])
        return

    res = run_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)

    RESULTS.mkdir(exist_ok=True)
    out = []
    n_ok, n_fail = 0, 0
    for rid, q in case_by_id.items():
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

    path = RESULTS / "lexicon_hypotheses_blind.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{n_ok} succeeded, {n_fail} failed. -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
