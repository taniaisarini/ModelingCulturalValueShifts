"""Pilot: ask a Claude model to generate language-pattern hypotheses (not topic-keyword indicators)
for one WVS case, blind to which direction actually happened. See prompts/{system,user}/
generate_language_hypotheses.txt for the exact prompt (approved before use). No measurement step yet --
that's deliberately deferred until we see what kind of hypotheses come back.

Usage: python pilot_language_hypotheses.py [--model claude-haiku-5-5] [--question F118]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fast_llm import run_llm  # noqa: E402
from cases import CASES  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"


def load_prompt(kind, name):
    return (PROMPTS / kind / f"{name}.txt").read_text()


def _strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def run(model, question):
    entry = next(c for c in CASES if c["question"] == question)
    case = entry["case"]
    system = load_prompt("system", "generate_language_hypotheses")
    user = load_prompt("user", "generate_language_hypotheses").format(**case)
    req = [{"id": "gen", "system": system, "user": user}]
    res = run_llm(req, model=model, temperature=1.0, max_tokens=1500)
    result = res["gen"]
    if result["error"]:
        raise RuntimeError(f"call failed: {result['error']}")
    print("--- raw response ---")
    print(result["text"])
    parsed = json.loads(_strip_fences(result["text"]))
    print(f"\n--- parsed: {len(parsed.get('hypotheses', []))} hypotheses ---")
    for h in parsed.get("hypotheses", []):
        print(f"[{h.get('direction')}, expect {h.get('expected_change')}] {h.get('hypothesis')}")
    return parsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--question", default="F118")
    a = ap.parse_args()
    run(a.model, a.question)
