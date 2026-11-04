"""One-off: dump the EXACT prompts and EXACT raw LLM responses for the last 20-question run, pulled
straight from llm_cache (every call here is a cache hit -- this makes no new LLM calls, costs nothing).
Writes results/full_transcript.md. Not part of the normal pipeline; run once, read, delete if unwanted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402
from cases import CASES  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TRANSCRIPT = []

_real_run_llm = pipeline.run_llm
_current = {"question": None, "label": None}


def _logging_run_llm(req, model=None, **kwargs):
    res = _real_run_llm(req, model=model, **kwargs)
    r = req[0]
    TRANSCRIPT.append(dict(question=_current["question"], label=_current["label"],
                            system=r["system"], user=r["user"],
                            response=res[r["id"]]["text"], cached=res[r["id"]].get("cached")))
    return res


pipeline.run_llm = _logging_run_llm

for entry in CASES:
    q = entry["question"]
    print(f"replaying {q} (cache hits only)...")
    _current["question"] = q
    _current["label"] = "generate_indicators"
    indicators = pipeline.generate_indicators(entry["case"], model=pipeline.MODEL)
    evidence = {"A": [], "B": []}
    for letter in ("A", "B"):
        for ind in indicators[letter]:
            if ind.get("source") not in ("nyt", "gdelt"):
                continue
            try:
                metrics = pipeline.measure_indicator(ind, entry["case"])
            except (KeyError, ValueError):
                continue
            evidence[letter].append({"indicator_obj": ind, "metrics": metrics})
    for round_num in range(1, 4):
        _current["label"] = f"aggregate_judge round {round_num}"
        result = pipeline.aggregate_judge(entry["case"], evidence, round_num, model=pipeline.MODEL)
        if result.get("decision") in ("A", "B") or round_num == 3:
            break
        for req_ind in result.get("requested_evidence", []):
            letter = req_ind.get("direction")
            if letter not in ("A", "B") or req_ind.get("source") not in ("nyt", "gdelt"):
                continue
            try:
                metrics = pipeline.measure_indicator(req_ind, entry["case"])
            except (KeyError, ValueError):
                continue
            evidence[letter].append({"indicator_obj": req_ind, "metrics": metrics})

pipeline.run_llm = _real_run_llm

lines = ["# Full transcript: exact prompts and exact raw LLM responses, all 20 questions",
         "", f"{len(TRANSCRIPT)} calls total, all cache hits (no new cost).", ""]
for i, c in enumerate(TRANSCRIPT, 1):
    lines += [f"## {i}. {c['question']} -- {c['label']} (cached={c['cached']})",
              "", "**System:**", "```", c["system"].strip(), "```",
              "", "**User:**", "```", c["user"].strip(), "```",
              "", "**Response:**", "```", c["response"].strip(), "```", ""]

TRANSCRIPT_PATH = RESULTS / "full_transcript.md"
TRANSCRIPT_PATH.write_text("\n".join(lines))
print(f"\nWrote {len(TRANSCRIPT)} calls -> {TRANSCRIPT_PATH}")
