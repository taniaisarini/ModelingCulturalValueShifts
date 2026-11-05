"""Generic mechanical test for round N (N>=3) of the language-hypothesis feedback loop
(results/language_hypotheses_round{N}.json). Same method as test_language_hypotheses_round2.py:
markers extracted from each hypothesis's own quoted example, matched against real NYT text (2006 vs
2011, rate per 100,000 corpus characters, two-proportion z-test), confirm/deny/inconclusive per
hypothesis, confirm-only voting (deny does NOT count toward the opposite direction -- 2026-11-04
correction). No new LLM calls.

Usage: python test_language_hypotheses_round_n.py --round 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_tools import two_proportion_z  # noqa: E402
from test_language_hypotheses import extract_markers, load_nyt_texts, count_matches, total_chars  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
YEAR_A, YEAR_B = 2006, 2011


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


def load_true_directions():
    rows = json.load(open(RESULTS / "language_hypotheses_tested.json"))
    return {r["question"]: r["true_direction"] for r in rows}


def run(round_n, texts_a=None, texts_b=None):
    if texts_a is None:
        print("loading NYT 2006/2011 into memory...")
        texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)
    chars_a, chars_b = total_chars(texts_a), total_chars(texts_b)

    data = json.load(open(RESULTS / f"language_hypotheses_round{round_n}.json"))
    true_dir = load_true_directions()

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
            test = two_proportion_z(m_a, len(texts_a), m_b, len(texts_b))
            rate_a, rate_b = m_a / chars_a * 100_000, m_b / chars_b * 100_000
            observed = "increase" if rate_b > rate_a else "decrease"
            significant = test is not None and test[3] < 0.05
            matches_expected = observed == h.get("expected_change")
            direction = h.get("direction", "?")
            if significant and matches_expected:
                verdict = "confirm"
                n_confirm += 1
                votes[direction] = votes.get(direction, 0) + 1
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
        predicted = predict(votes)
        td = true_dir.get(q)
        row["votes"] = votes
        row["predicted_direction"] = predicted
        row["true_direction"] = td
        row["correct"] = (predicted == td) if td and predicted != "tie" else None
        per_question.append(row)

    (RESULTS / f"language_hypotheses_round{round_n}_tested.json").write_text(json.dumps(per_question, indent=2))

    n = len(per_question)
    decided = [r for r in per_question if r["predicted_direction"] != "tie"]
    correct = [r for r in decided if r["correct"]]
    print(f"round {round_n}: {n_tested} hyps tested, {n_untestable} untestable (no quoted marker)")
    print(f"verdicts: {n_confirm} confirm, {n_deny} deny, {n_inconclusive} inconclusive")
    if decided:
        print(f"this round alone decided: {len(decided)}/{n}, accuracy on those: {len(correct)}/{len(decided)}")
    else:
        print("this round alone decided: 0 (all still tied)")
    print(f"-> {RESULTS / f'language_hypotheses_round{round_n}_tested.json'}")
    return per_question


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    run(a.round)
