"""Mechanical test of the structured lexicon hypotheses (results/lexicon_hypotheses_pilot.json) against
real NYT text, no new LLM calls. Unlike test_language_hypotheses.py (markers inferred post hoc from a
free-text sentence), each hypothesis here already carries its own structured word/phrase lists, so no
marker-extraction step is needed.

Two hypothesis types:
  - "trend": one lexicon, tested for a frequency change from year_a to year_b.
  - "contrast": two lexicons (lexicon_early/lexicon_late), tested for a dominance crossover.
Either type may carry a non-empty `topic` list; if present, articles are first filtered to those
containing >=1 topic phrase before any lexicon counting.

Every hypothesis is scored by BOTH rate metrics, kept side by side rather than merged, so the pilot can
compare which tracks reality better:
  - presence: fraction of (topic-filtered) articles containing >=1 lexicon word (article-level, like
    test_language_hypotheses.py's count_matches()).
  - chars: raw word/phrase occurrences per 100,000 characters of (topic-filtered) text.
Each metric's comparison is a two-proportion z-test (analysis_tools.two_proportion_z), same statistic
test_language_hypotheses.py already uses -- for "chars" this applies the test to (occurrences, total
characters) instead of (matching articles, total articles), a valid normal-approximation for a rare-event
rate comparison.

Contrast hypotheses are additionally scored by TWO methods, also kept side by side, each under both rate
metrics above (4 verdicts total):
  - independent: two separate trend tests, one per lexicon (lexicon_early expected to decrease,
    lexicon_late expected to increase) -- confirmed only if both hold, denied if either contradicts.
  - dominance: one test on lexicon_early's SHARE of (lexicon_early + lexicon_late) mentions, at year_a vs
    year_b -- expected to decrease (early dominates less as time passes).

Usage: python test_lexicon_hypotheses.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_tools import two_proportion_z  # noqa: E402
from test_language_hypotheses import load_nyt_texts, total_chars  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
YEAR_A, YEAR_B = 2006, 2011


def filter_by_topic(texts, topic):
    """Subset to articles containing >=1 topic phrase (case-insensitive substring). Empty/missing
    topic = no filtering (returns texts unchanged)."""
    if not topic:
        return texts
    topic_l = [t.lower() for t in topic]
    return [t for t in texts if any(k in t for k in topic_l)]


def count_presence(texts, markers):
    """Number of articles containing >=1 marker (article-level, like test_language_hypotheses.py)."""
    markers_l = [m.lower() for m in markers]
    return sum(1 for t in texts if any(m in t for m in markers_l))


def count_occurrences(texts, markers):
    """Total marker occurrences (not just presence), summed across all articles and markers."""
    markers_l = [m.lower() for m in markers]
    return sum(t.count(m) for t in texts for m in markers_l)


def rate_test(x_a, n_a, x_b, n_b):
    """Two-proportion z-test wrapper. Returns a dict with the raw counts, proportions, p-value and
    observed increase/decrease; p_value is None if either n is 0 (nothing to test)."""
    test = two_proportion_z(x_a, n_a, x_b, n_b)
    if test is None:
        return dict(x_a=x_a, n_a=n_a, x_b=x_b, n_b=n_b, p_a=None, p_b=None, p_value=None, observed=None)
    p_a, p_b, _z, p_value = test
    return dict(x_a=x_a, n_a=n_a, x_b=x_b, n_b=n_b, p_a=round(p_a, 6), p_b=round(p_b, 6),
                p_value=p_value, observed=("increase" if p_b > p_a else "decrease"))


def verdict_for(rate, expected_change):
    if rate["p_value"] is None:
        return "inconclusive"
    significant = rate["p_value"] < 0.05
    matches = rate["observed"] == expected_change
    if significant and matches:
        return "confirm"
    if significant and not matches:
        return "deny"
    return "inconclusive"


def combine_verdicts(v1, v2):
    """Both must confirm to confirm; either denying is a contradiction (deny); else inconclusive."""
    if v1 == "confirm" and v2 == "confirm":
        return "confirm"
    if v1 == "deny" or v2 == "deny":
        return "deny"
    return "inconclusive"


def test_trend(sub_a, sub_b, chars_a, chars_b, lexicon, expected_change):
    presence = rate_test(count_presence(sub_a, lexicon), len(sub_a),
                          count_presence(sub_b, lexicon), len(sub_b))
    chars = rate_test(count_occurrences(sub_a, lexicon), chars_a,
                       count_occurrences(sub_b, lexicon), chars_b)
    return dict(presence=dict(presence, verdict=verdict_for(presence, expected_change)),
                chars=dict(chars, verdict=verdict_for(chars, expected_change)))


def test_contrast(sub_a, sub_b, chars_a, chars_b, lexicon_early, lexicon_late):
    out = {}
    for metric_name, count_fn, total_a, total_b in (
        ("presence", count_presence, len(sub_a), len(sub_b)),
        ("chars", count_occurrences, chars_a, chars_b),
    ):
        early_a, early_b = count_fn(sub_a, lexicon_early), count_fn(sub_b, lexicon_early)
        late_a, late_b = count_fn(sub_a, lexicon_late), count_fn(sub_b, lexicon_late)

        early_trend = rate_test(early_a, total_a, early_b, total_b)
        late_trend = rate_test(late_a, total_a, late_b, total_b)
        v_early, v_late = verdict_for(early_trend, "decrease"), verdict_for(late_trend, "increase")
        independent = dict(early=early_trend, late=late_trend, verdict=combine_verdicts(v_early, v_late))

        dominance_rate = rate_test(early_a, early_a + late_a, early_b, early_b + late_b)
        dominance = dict(dominance_rate, verdict=verdict_for(dominance_rate, "decrease"))

        out[metric_name] = dict(independent=independent, dominance=dominance)
    return out


def run():
    print("loading NYT 2006/2011 into memory...")
    texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)
    print(f"  {len(texts_a)} articles ({YEAR_A}), {len(texts_b)} articles ({YEAR_B})")

    data = json.load(open(RESULTS / "lexicon_hypotheses_pilot.json"))

    n_trend = n_contrast = n_untestable = 0
    out = []
    for d in data:
        row = dict(country=d["country"], question=d["question"], hypotheses=[])
        for h in d.get("hypotheses", []):
            htype = h.get("type")
            topic = h.get("topic") or []
            sub_a, sub_b = filter_by_topic(texts_a, topic), filter_by_topic(texts_b, topic)
            chars_a, chars_b = total_chars(sub_a), total_chars(sub_b)

            if htype == "trend" and h.get("lexicon") and h.get("expected_change"):
                n_trend += 1
                result = test_trend(sub_a, sub_b, chars_a, chars_b, h["lexicon"], h["expected_change"])
            elif htype == "contrast" and h.get("lexicon_early") and h.get("lexicon_late"):
                n_contrast += 1
                result = test_contrast(sub_a, sub_b, chars_a, chars_b, h["lexicon_early"], h["lexicon_late"])
            else:
                n_untestable += 1
                row["hypotheses"].append(dict(h, tested=False))
                continue
            row["hypotheses"].append(dict(h, tested=True, n_articles_a=len(sub_a), n_articles_b=len(sub_b),
                                          chars_a=chars_a, chars_b=chars_b, result=result))
        out.append(row)

    (RESULTS / "lexicon_hypotheses_pilot_tested.json").write_text(json.dumps(out, indent=2))
    print(f"\nhypotheses: {n_trend} trend, {n_contrast} contrast, {n_untestable} untestable (missing fields)")
    print(f"-> {RESULTS / 'lexicon_hypotheses_pilot_tested.json'}")


if __name__ == "__main__":
    run()
