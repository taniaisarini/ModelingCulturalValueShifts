"""When the answer order was flipped and the model's answer changed, did it cite the same reasoning?

    python code/reasoning_flip_consistency.py

Pairs each reply from the original run with the same model's reply to the same question in the flipped run. For pairs where the
predicted direction changed ("flipped") and pairs where it did not ("same"), measures how similar the reasoning is:
  - cites a shared event: both replies mention at least one common year or proper noun (country names ignored)
  - first-sentence similarity: TF-IDF cosine of the first sentence of each reply (the sentence that sets out the context or event,
    before the conclusion), country names removed
Writes results/direction_baseline/five_country_set_flipped/flip_consistency.csv and flip_examples.md.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reasoning_common import NAME_WORDS, SET_DIR  # noqa: E402

FLIP_DIR = SET_DIR.parent / "five_country_set_flipped"
RUNS = [("gpt-4.1-mini", SET_DIR / "all_gpt-4.1-mini.csv", FLIP_DIR / "by_model/gpt-4.1-mini/all_gpt-4.1-mini.csv")] + [
    (f"{m} (reasoning {r})", SET_DIR / f"by_model/{m}__reasoning-{r}/all_{m}__reasoning-{r}.csv",
     FLIP_DIR / f"by_model/{m}__reasoning-{r}/all_{m}__reasoning-{r}.csv") for m in ["gpt-5-mini", "gpt-5"] for r in ["off", "on"]]
NAMES = {w.lower() for w in NAME_WORDS} | {"between", "during", "after", "before", "therefore", "the", "this", "these", "in", "as", "with", "under"}


def strip_answer(t):
    return re.sub(r"\s*ANSWER:\s*\S*\s*$", "", str(t).strip()).strip()


def first_sentence(t):
    return re.split(r"(?<=[.!?])\s+", t, maxsplit=1)[0]


def entities(t, given=()):
    """Years and capitalised words that are not at the start of a sentence (events, people, places).
    `given` are the survey years stated in the prompt; every reply repeats them, so they do not count as a cited event."""
    out = set(re.findall(r"\b(?:19|20)\d\d\b", t)) - set(given)
    for sent in re.split(r"(?<=[.!?])\s+", t):
        words = re.findall(r"\b[A-Z][a-zA-Z\-]+\b", sent)
        first = re.match(r"\W*([A-Za-z\-]+)", sent)
        out |= {w.lower() for w in words if not (first and w == first.group(1))} - NAMES
    return out - {w.lower() for w in NAME_WORDS}


def main():
    rows, examples = [], []
    for model, fo, ff in RUNS:
        a, b = pd.read_csv(fo), pd.read_csv(ff)
        assert (a.code == b.code).all() and (a.country == b.country).all() and (a.wave_a == b.wave_a).all()
        ra, rb = a.response.map(strip_answer), b.response.map(strip_answer)
        fa, fb = ra.map(first_sentence), rb.map(first_sentence)
        stop = list(ENGLISH_STOP_WORDS | {w.lower() for w in NAME_WORDS})
        vec = TfidfVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True).fit(pd.concat([fa, fb]))
        A, B = vec.transform(fa), vec.transform(fb)
        cos = np.asarray(A.multiply(B).sum(axis=1)).ravel()          # rows are L2-normalised
        given = [set(re.findall(r"\d{4}", f"{ya} {yb}")) for ya, yb in zip(a.years_a, a.years_b)]
        ea = [entities(x, g) for x, g in zip(ra, given)]
        eb = [entities(y, g) for y, g in zip(rb, given)]
        shared = np.array([bool(x & y) for x, y in zip(ea, eb)])
        flipped = (a.pred != b.pred).to_numpy()
        for name, m in [("answer changed", flipped), ("answer same", ~flipped)]:
            rows.append(dict(model=model, group=name, n=int(m.sum()), cites_shared_event_pct=round(100 * shared[m].mean(), 1),
                             first_sentence_cosine_median=round(float(np.median(cos[m])), 3),
                             cosine_above_0_5_pct=round(100 * (cos[m] > 0.5).mean(), 1)))
        idx = np.where(flipped)[0]
        for i in idx[:4]:
            examples.append((model, a.country[i], f"{a.wave_a[i]}->{a.wave_b[i]}", a.code[i], a.response[i], b.response[i],
                             float(cos[i]), bool(shared[i])))
    out = pd.DataFrame(rows)
    out.to_csv(FLIP_DIR / "flip_consistency.csv", index=False)
    lines = ["# Examples where the answer changed when the option order was flipped\n"]
    for m, c, w, code, ra_, rb_, cs, sh in examples:
        lines.append(f"## {m}: {c} {w} {code} (first-sentence cosine {cs:.2f}, shared event: {sh})\n\n**Original order:** {ra_}\n\n**Flipped order:** {rb_}\n")
    (FLIP_DIR / "flip_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
