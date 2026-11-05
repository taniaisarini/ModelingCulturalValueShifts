"""Full-scale run of the structured, lexicon-based hypothesis experiment (prompts/{system,user}/
generate_lexicon_hypotheses.txt, approved and schema-fixed 2026-11-05 -- see run_lexicon_hypotheses.py's
docstring for the four-kind schema). Same prompt as the 5-question pilot, scaled to every significant
USA wave5->6 change whose question has a usable (two-ended) format -- see bulk_cases.py. One LLM call
per question, same scope restriction as run_language_hypotheses_bulk.py (USA only: NYT is a plausible
domestic-discourse proxy for the US, not for the other 5 WVS countries in this project).

max_tokens=3000, not the pilot's initial 1500: the pilot showed 1500 truncates roughly half of calls
under this richer 4-kind schema (more structured fields per hypothesis than the old free-text schema),
so retrying at a higher budget per-failure would be wasteful at this scale -- start high instead.

Usage:
    python run_lexicon_hypotheses_bulk.py                 # dry run: lists pairs + cache plan, no calls
    python run_lexicon_hypotheses_bulk.py --run            # actually call the LLM (cached; code/llm_cache/)
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_llm import plan_llm, run_llm  # noqa: E402
from bulk_cases import build_generic_case, usable_question  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
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


def find_pairs():
    """All USA significant-change pairs (ks, wave5->6) with a usable question format."""
    pairs = []
    for path in glob.glob(str(ROOT / "data" / "significant_changes" / "ks" / "*_wave5_to_wave6.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["country"] == "USA" and usable_question(r["question"]):
                    pairs.append((r["country"], r["question"]))
    return sorted(set(pairs))


def run(model, dry_run):
    pairs = find_pairs()
    system = load_prompt("system", "generate_lexicon_hypotheses")
    user_template = load_prompt("user", "generate_lexicon_hypotheses")

    requests, case_by_id = [], {}
    for country, question in pairs:
        case = build_generic_case(question, country)
        rid = f"{country}_{question}"
        user = user_template.format(**case)
        requests.append({"id": rid, "system": system, "user": user})
        case_by_id[rid] = (country, question)

    print(f"{len(pairs)} (country, question) pairs with a usable format.")
    if dry_run:
        plan = plan_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)
        print("plan:", plan)
        print("\n--- example prompt (first pair) ---")
        print(requests[0]["user"])
        return

    res = run_llm(requests, model=model, temperature=1.0, max_tokens=MAX_TOKENS)

    RESULTS.mkdir(exist_ok=True)
    out = []
    n_ok, n_fail = 0, 0
    for rid, (country, question) in case_by_id.items():
        r = res[rid]
        if r["error"]:
            n_fail += 1
            out.append(dict(country=country, question=question, error=r["error"]))
            continue
        try:
            parsed = json.loads(_strip_fences(r["text"]))
        except json.JSONDecodeError:
            n_fail += 1
            out.append(dict(country=country, question=question, error="bad JSON", raw=r["text"]))
            continue
        n_ok += 1
        out.append(dict(country=country, question=question, hypotheses=parsed.get("hypotheses", [])))

    path = RESULTS / "lexicon_hypotheses_bulk.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{n_ok} succeeded, {n_fail} failed. -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    run(a.model, dry_run=not a.run)
