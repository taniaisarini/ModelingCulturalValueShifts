"""Quick test of the 643 language hypotheses (results/language_hypotheses_bulk.json) against real NYT
text, with NO new LLM calls: markers are extracted mechanically from each hypothesis's own quoted
example words/phrases (most hypotheses already name them, e.g. "...uses 'equality,' 'civil rights'...").
~96% of hypotheses have >=1 extractable marker; the rest (pure structural/framing claims with no quoted
example) are left untested rather than forced.

For each testable hypothesis: count NYT articles (headline+abstract+lead_paragraph+subject tags, see
analysis_tools.nyt_rate2) matching ANY of its markers, in 2006 vs. 2011, two-proportion z-test for
significance. Verdict: confirm (direction matches AND p<0.05), deny (opposite direction AND p<0.05),
inconclusive otherwise. Per question, direction with more confirms is this round's "predicted direction",
compared against the real wave5->6 result for a rough accuracy number.

Usage: python test_language_hypotheses.py
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_tools import two_proportion_z, NYT_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
YEAR_A, YEAR_B = 2006, 2011


def extract_markers(text):
    raw = re.findall(r"['\"]([^'\"]{2,40})['\"]", text or "")
    cleaned = [m.strip(" ,.;:!?").strip() for m in raw]
    return [m for m in cleaned if m and 1 <= len(m.split()) <= 5 and m.lower() not in ("a", "the", "and")]


def load_nyt_texts(year):
    """All (text, total_count) for one year, text = lowercased headline+abstract+lead_paragraph+tags
    per doc, loaded once so 600+ marker tests don't each re-read 12 JSON files."""
    texts = []
    for month in range(1, 13):
        path = NYT_DIR / f"{year}-{month:02d}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())["response"]["docs"]
        for doc in docs:
            if doc.get("document_type") != "article":
                continue  # multimedia/audio entries are ~textless; their growing share (2% -> 11%
                          # from 2006 to 2011) mechanically depresses every rate if left in
            headline = (doc.get("headline") or {}).get("main") or ""
            tags = " ".join((kw.get("name") or "") for kw in (doc.get("keywords") or []))
            texts.append(f"{headline} {doc.get('abstract') or ''} {doc.get('lead_paragraph') or ''} {tags}".lower())
    return texts


def count_matches(texts, markers):
    markers_l = [m.lower() for m in markers]
    return sum(1 for t in texts if any(m in t for m in markers_l))


def load_ground_truth():
    truth = {}
    for path in (ROOT / "data" / "significant_changes" / "ks").glob("*_wave5_to_wave6.csv"):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["country"] == "USA":
                    truth[r["question"]] = float(r["mean_wave_6"]) > float(r["mean_wave_5"])
    return truth  # True = mean increased (direction "A" in our case-builder convention)


def run():
    print("loading NYT 2006/2011 into memory...")
    texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)
    total_a, total_b = len(texts_a), len(texts_b)
    print(f"  {total_a} articles ({YEAR_A}), {total_b} articles ({YEAR_B})")

    data = json.load(open(RESULTS / "language_hypotheses_bulk.json"))
    truth = load_ground_truth()

    n_tested = n_untestable = n_confirm = n_deny = n_inconclusive = 0
    per_question = []
    for d in data:
        q = d["question"]
        row = dict(question=q, hypotheses=[])
        votes = {"A": 0, "B": 0}
        for h in d.get("hypotheses", []):
            markers = extract_markers(h.get("hypothesis", ""))
            if not markers:
                n_untestable += 1
                row["hypotheses"].append(dict(h, markers=[], tested=False))
                continue
            m_a, m_b = count_matches(texts_a, markers), count_matches(texts_b, markers)
            test = two_proportion_z(m_a, total_a, m_b, total_b)
            rate_a, rate_b = m_a / total_a * 1000, m_b / total_b * 1000
            observed = "increase" if rate_b > rate_a else "decrease"
            significant = test is not None and test[3] < 0.05
            matches_expected = observed == h.get("expected_change")
            if significant and matches_expected:
                verdict = "confirm"
                n_confirm += 1
                votes[h.get("direction", "?")] = votes.get(h.get("direction", "?"), 0) + 1
            elif significant and not matches_expected:
                verdict = "deny"
                n_deny += 1
            else:
                verdict = "inconclusive"
                n_inconclusive += 1
            n_tested += 1
            row["hypotheses"].append(dict(h, markers=markers, tested=True,
                                          matches_a=m_a, matches_b=m_b, rate_a=round(rate_a, 3),
                                          rate_b=round(rate_b, 3), p_value=(test[3] if test else None),
                                          observed=observed, verdict=verdict))
        predicted = "A" if votes["A"] > votes["B"] else ("B" if votes["B"] > votes["A"] else "tie")
        true_dir = "A" if truth.get(q) else "B" if q in truth else None
        row["votes"] = votes
        row["predicted_direction"] = predicted
        row["true_direction"] = true_dir
        row["correct"] = (predicted == true_dir) if true_dir and predicted != "tie" else None
        per_question.append(row)

    (RESULTS / "language_hypotheses_tested.json").write_text(json.dumps(per_question, indent=2))

    decided = [r for r in per_question if r["predicted_direction"] != "tie"]
    correct = [r for r in decided if r["correct"]]
    print(f"\nhypotheses: {n_tested} tested, {n_untestable} untestable (no quoted marker)")
    print(f"verdicts: {n_confirm} confirm, {n_deny} deny, {n_inconclusive} inconclusive")
    print(f"questions: {len(decided)}/{len(per_question)} reached a non-tie predicted direction")
    print(f"accuracy on those: {len(correct)}/{len(decided)} ({100*len(correct)/len(decided):.1f}%)"
          if decided else "no decided questions")
    print(f"\n-> {RESULTS / 'language_hypotheses_tested.json'}")


if __name__ == "__main__":
    run()
