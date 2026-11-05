"""Pilot run of the structured, lexicon-based hypothesis experiment (prompts/{system,user}/
generate_lexicon_hypotheses.txt, approved 2026-11-05). Unlike the earlier language-hypothesis experiment
(any kind of language pattern, free text, markers inferred post hoc by regex), each hypothesis here is
one of two fixed JSON shapes the model must fill in directly:
  - "trend": a single word/phrase list whose frequency rises or falls between the two time points.
  - "contrast": two word/phrase lists, one expected to dominate near year_a and the other near year_b.
Either type may optionally carry a `topic` word/phrase list restricting which articles count at all.

Scope: 5 USA wave5->6 questions, one per usable significant-change format (ordinal_scale, numeric_scale
x2 -- including F118, the original pilot question from the earlier experiment, for continuity --
binary, mention_flag), not the full 128-question set, per 2026-11-05 decision to pilot small first.

Usage:
    python run_lexicon_hypotheses.py                 # dry run: lists cases + cache plan, no calls
    python run_lexicon_hypotheses.py --run            # actually call the LLM (cached; code/llm_cache/)
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

# One USA wave5->6 question per usable significant-change format; F118 repeats the earlier
# language-hypothesis experiment's pilot question for continuity.
PILOT_QUESTIONS = ["F118", "A001", "A168A", "A165", "A124_02"]


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def run(model, dry_run):
    system = load_prompt("system", "generate_lexicon_hypotheses")
    user_template = load_prompt("user", "generate_lexicon_hypotheses")

    requests, case_by_id = [], {}
    for q in PILOT_QUESTIONS:
        case = build_generic_case(q, "USA")
        rid = f"USA_{q}"
        user = user_template.format(**case)
        requests.append({"id": rid, "system": system, "user": user})
        case_by_id[rid] = q

    print(f"{len(PILOT_QUESTIONS)} pilot questions: {PILOT_QUESTIONS}")
    if dry_run:
        plan = plan_llm(requests, model=model, temperature=1.0, max_tokens=1500)
        print("plan:", plan)
        print("\n--- example prompt (first question) ---")
        print(requests[0]["user"])
        return

    res = run_llm(requests, model=model, temperature=1.0, max_tokens=1500)

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

    path = RESULTS / "lexicon_hypotheses_pilot.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{n_ok} succeeded, {n_fail} failed. -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
