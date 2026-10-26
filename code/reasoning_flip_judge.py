"""LLM judge: for each question, do the original-order and flipped-order replies cite the same reason or event?

    python code/reasoning_flip_judge.py            # dry run: counts calls and cache hits, sends nothing
    python code/reasoning_flip_judge.py --run      # sends them (default judge gpt-4.1-mini, temperature 0)

Prompt: code/prompts/user/flip_judge/same_reason.txt (no system prompt; the final "ANSWER:" line is removed from both replies, and the
judge is not told which reply came from which order or whether the answers matched). One call per (model run, country, wave pair,
question); text 1 is always the original-order reply. Writes results/direction_baseline/five_country_set_flipped/flip_judgements.csv
and prints, per model, how often the judge said YES when the answer changed and when it stayed the same.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_llm import plan_llm, run_llm  # noqa: E402
from reasoning_flip_consistency import FLIP_DIR, RUNS, strip_answer  # noqa: E402

PROMPT = (Path(__file__).resolve().parent / "prompts" / "user" / "flip_judge" / "same_reason.txt").read_text().strip()
PARAMS = dict(temperature=0.0, max_tokens=5)
# gpt-5 rejects temperature/max_tokens, so it gets the default temperature, medium reasoning effort and a larger cap (reasoning tokens count);
# passed through extra_body because the installed openai SDK is too old to know these arguments
PARAMS_GPT5 = dict(temperature=None, max_tokens=None, extra={"extra_body": {"max_completion_tokens": 2000, "reasoning_effort": "medium"}})


def judge_params(model):
    return PARAMS_GPT5 if model.startswith("gpt-5") else PARAMS


def build_pairs():
    rows = []
    for model, fo, ff in RUNS:
        a, b = pd.read_csv(fo), pd.read_csv(ff)
        assert (a.code == b.code).all() and (a.country == b.country).all() and (a.wave_a == b.wave_a).all()
        for i in range(len(a)):
            t1, t2 = strip_answer(a.response[i]), strip_answer(b.response[i])
            rows.append(dict(id=f"{model}|{a.country[i]}|{a.wave_a[i]}->{a.wave_b[i]}|{a.code[i]}", model=model, country=a.country[i],
                             waves=f"{a.wave_a[i]}->{a.wave_b[i]}", code=a.code[i], answer_changed=bool(a.pred[i] != b.pred[i]),
                             text_1=t1, text_2=t2,
                             user=PROMPT.replace("{text_1}", t1).replace("{text_2}", t2)))   # replace, not format: replies may contain braces
    return pd.DataFrame(rows)


def parse(text):
    m = re.match(r"\s*\W*(?:answer\s*:?\s*)?\W*(YES|NO)\b", text or "", re.I)   # also accepts "Answer: YES"
    return m.group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="gpt-4.1-mini")
    a = ap.parse_args()
    pairs = build_pairs()
    reqs = [{"id": r.id, "user": r.user} for r in pairs.itertuples()]
    params = judge_params(a.model)
    plan = plan_llm(reqs, model=a.model, **params)
    print(f"{len(reqs)} pairs; judge {a.model} {params}; in cache {plan['hits']}, to send {plan['unique_misses']}")
    if not a.run:
        print("dry run only. Example prompt:\n" + pairs.user[0])
        return
    res = run_llm(reqs, model=a.model, **params)
    pairs["judge_raw"] = [res[i]["text"] for i in pairs.id]
    pairs["judge"] = pairs.judge_raw.map(parse)
    out = FLIP_DIR / ("flip_judgements.csv" if a.model == "gpt-4.1-mini" else f"flip_judgements_{a.model}.csv")
    pairs.drop(columns=["user"]).to_csv(out, index=False)
    print("unparsed:", int(pairs.judge.isna().sum()), " errors:", sum(res[i]["error"] is not None for i in pairs.id))
    t = pairs.assign(yes=pairs.judge == "YES").groupby(["model", "answer_changed"])["yes"].agg(["size", "mean"]).round(3)
    print(t.to_string())


if __name__ == "__main__":
    main()
