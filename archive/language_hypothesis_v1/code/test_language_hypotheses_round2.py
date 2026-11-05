"""Mechanical test of round 2's 150 hypotheses (results/language_hypotheses_round2.json, the 30
questions that tied in round 1), same method as test_language_hypotheses.py: markers extracted from
each hypothesis's own quoted example, matched against real NYT text (2006 vs 2011, rate per 100,000
corpus characters, two-proportion z-test), confirm/deny/inconclusive per hypothesis. Deny counts as a
vote for the OPPOSITE direction (not just discarded), matching the current round-1 scoring.

No new LLM calls. Usage: python test_language_hypotheses_round2.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_tools import two_proportion_z  # noqa: E402
from test_language_hypotheses import extract_markers, load_nyt_texts, count_matches, total_chars, load_ground_truth  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
YEAR_A, YEAR_B = 2006, 2011


def run():
    print("loading NYT 2006/2011 into memory...")
    texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)
    chars_a, chars_b = total_chars(texts_a), total_chars(texts_b)

    data = json.load(open(RESULTS / "language_hypotheses_round2.json"))
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
            test = two_proportion_z(m_a, len(texts_a), m_b, len(texts_b))
            rate_a, rate_b = m_a / chars_a * 100_000, m_b / chars_b * 100_000
            observed = "increase" if rate_b > rate_a else "decrease"
            significant = test is not None and test[3] < 0.05
            matches_expected = observed == h.get("expected_change")
            direction = h.get("direction", "?")
            opposite = "B" if direction == "A" else "A" if direction == "B" else "?"
            if significant and matches_expected:
                verdict = "confirm"
                n_confirm += 1
                votes[direction] = votes.get(direction, 0) + 1
            elif significant and not matches_expected:
                verdict = "deny"
                n_deny += 1
                votes[opposite] = votes.get(opposite, 0) + 1
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

    (RESULTS / "language_hypotheses_round2_tested.json").write_text(json.dumps(per_question, indent=2))

    n = len(per_question)
    decided = [r for r in per_question if r["predicted_direction"] != "tie"]
    correct = [r for r in decided if r["correct"]]
    wrong = [r for r in decided if r["correct"] is False]
    tied = [r for r in per_question if r["predicted_direction"] == "tie"]
    print(f"\nhypotheses: {n_tested} tested, {n_untestable} untestable (no quoted marker)")
    print(f"verdicts: {n_confirm} confirm, {n_deny} deny, {n_inconclusive} inconclusive")
    print(f"\nof {n} round-1-tied questions, re-tested with round-2 hypotheses:")
    print(f"  correct:      {len(correct)}/{n} ({100*len(correct)/n:.1f}%)")
    print(f"  wrong:        {len(wrong)}/{n} ({100*len(wrong)/n:.1f}%)")
    print(f"  inconclusive: {len(tied)}/{n} ({100*len(tied)/n:.1f}%)  (ties, incl. 0-0 no-confirm-either-way)")
    print(f"\n-> {RESULTS / 'language_hypotheses_round2_tested.json'}")


if __name__ == "__main__":
    run()
