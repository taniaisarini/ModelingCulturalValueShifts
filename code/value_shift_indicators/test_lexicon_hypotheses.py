"""Mechanical test of the structured lexicon hypotheses (results/lexicon_hypotheses_pilot.json) against
real NYT text, no new LLM calls. Unlike test_language_hypotheses.py (markers inferred post hoc from a
free-text sentence), each hypothesis here already carries its own structured word/phrase lists, so no
marker-extraction step is needed.

Exactly four hypothesis kinds (2026-11-05 schema fix -- `topic` used to be an optional field on a
generic trend/contrast shape, which let the model leave it empty by choice instead of committing to
whether the pattern was topic-restricted at all; now it's a required part of which of four kinds a
hypothesis is):
  - "trend_general": one lexicon, tested for a frequency change from year_a to year_b, across ALL articles.
  - "trend_topic": one lexicon, same frequency-change test, but first retrieved down to only the articles
    matching the hypothesis's own `topic` word/phrase list (a keyword-based retrieval step) -- a simple
    retrieve-then-check pipeline: `topic` retrieves the relevant subset, `lexicon` is what's checked
    within it. No LLM involved in this step, only in generating the two lexica up front.
  - "contrast_general": two lexicons (lexicon_early/lexicon_late), tested for a dominance crossover,
    across all articles.
  - "contrast_topic": same crossover test, but within the `topic`-retrieved subset only.
A "_topic" hypothesis MUST carry a non-empty `topic` list (retrieval step applies); a "_general"
hypothesis has no `topic` field at all (no retrieval filter -- the full corpus is "all articles").

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

Usage:
    python test_lexicon_hypotheses.py          # scores results/lexicon_hypotheses_pilot.json (default)
    python test_lexicon_hypotheses.py bulk      # scores results/lexicon_hypotheses_bulk.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_tools import two_proportion_z  # noqa: E402
from test_language_hypotheses import load_nyt_texts, total_chars  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
YEAR_A, YEAR_B = 2006, 2011
VALID_KINDS = {"trend_general", "trend_topic", "contrast_general", "contrast_topic"}


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


def chars_from_presence(presence_test, occ_a, chars_a, occ_b, chars_b):
    """The chars metric's significance test, FIXED (2026-11-05 bug found by inspecting 20 cases where
    the old free-text experiment was right and this one was wrong): testing raw occurrence counts
    against total corpus CHARACTERS (tens to hundreds of millions for an unrestricted hypothesis) makes
    a two-proportion z-test absurdly over-powered -- almost any word whose raw count didn't shrink by
    exactly the same ratio as the whole corpus (2006 corpus is ~2.8x the characters of 2011's) comes
    back "significant" (p=0.0 or 1e-15) regardless of real relevance to the hypothesis. Fix: the chars
    metric's *direction* still comes from the occurrences-per-character rate (that's the whole point of
    having a second metric), but its *significance* is NOT tested independently -- it reuses
    `presence_test`'s p-value, exactly how the original language-hypothesis experiment's char-rate
    fix worked (char rate only ever decided direction there, never significance; the z-test's n was
    always the article count)."""
    rate_a = occ_a / chars_a if chars_a else 0.0
    rate_b = occ_b / chars_b if chars_b else 0.0
    observed = None if (chars_a == 0 or chars_b == 0) else ("increase" if rate_b > rate_a else "decrease")
    return dict(x_a=occ_a, n_a=chars_a, x_b=occ_b, n_b=chars_b,
                rate_a=round(rate_a * 100_000, 4), rate_b=round(rate_b * 100_000, 4),
                p_value=presence_test["p_value"], observed=observed, significance_source="presence")


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
    occ_a, occ_b = count_occurrences(sub_a, lexicon), count_occurrences(sub_b, lexicon)
    chars = chars_from_presence(presence, occ_a, chars_a, occ_b, chars_b)
    return dict(presence=dict(presence, verdict=verdict_for(presence, expected_change)),
                chars=dict(chars, verdict=verdict_for(chars, expected_change)))


def test_contrast(sub_a, sub_b, chars_a, chars_b, lexicon_early, lexicon_late):
    n_a, n_b = len(sub_a), len(sub_b)
    pres_early_a, pres_early_b = count_presence(sub_a, lexicon_early), count_presence(sub_b, lexicon_early)
    pres_late_a, pres_late_b = count_presence(sub_a, lexicon_late), count_presence(sub_b, lexicon_late)

    pres_early_trend = rate_test(pres_early_a, n_a, pres_early_b, n_b)
    pres_late_trend = rate_test(pres_late_a, n_a, pres_late_b, n_b)
    v_early_p, v_late_p = verdict_for(pres_early_trend, "decrease"), verdict_for(pres_late_trend, "increase")
    pres_independent = dict(early=pres_early_trend, late=pres_late_trend,
                            verdict=combine_verdicts(v_early_p, v_late_p))
    pres_dominance_rate = rate_test(pres_early_a, pres_early_a + pres_late_a,
                                    pres_early_b, pres_early_b + pres_late_b)
    pres_dominance = dict(pres_dominance_rate, verdict=verdict_for(pres_dominance_rate, "decrease"))

    # chars: direction from occurrence-per-char rate, significance reused from the matching
    # presence-based lexicon trend test (see chars_from_presence) -- dominance is untouched, its own
    # n (early+late occurrence counts) was never affected by the huge-character-count bug.
    occ_early_a, occ_early_b = count_occurrences(sub_a, lexicon_early), count_occurrences(sub_b, lexicon_early)
    occ_late_a, occ_late_b = count_occurrences(sub_a, lexicon_late), count_occurrences(sub_b, lexicon_late)
    chars_early_trend = chars_from_presence(pres_early_trend, occ_early_a, chars_a, occ_early_b, chars_b)
    chars_late_trend = chars_from_presence(pres_late_trend, occ_late_a, chars_a, occ_late_b, chars_b)
    v_early_c, v_late_c = verdict_for(chars_early_trend, "decrease"), verdict_for(chars_late_trend, "increase")
    chars_independent = dict(early=chars_early_trend, late=chars_late_trend,
                             verdict=combine_verdicts(v_early_c, v_late_c))
    chars_dominance_rate = rate_test(occ_early_a, occ_early_a + occ_late_a, occ_early_b, occ_early_b + occ_late_b)
    chars_dominance = dict(chars_dominance_rate, verdict=verdict_for(chars_dominance_rate, "decrease"))

    return dict(presence=dict(independent=pres_independent, dominance=pres_dominance),
                chars=dict(independent=chars_independent, dominance=chars_dominance))


def score_hypothesis(h, texts_a, texts_b):
    """Score one hypothesis dict against pre-loaded NYT text for both years. Returns the same dict
    with tested=True and n_articles_a/n_articles_b/chars_a/chars_b/result added, or tested=False and
    untestable_reason added. Pulled out of run() so the feedback-loop script can score new hypotheses
    the same way without duplicating this logic."""
    kind = h.get("kind")
    if kind not in VALID_KINDS:
        return dict(h, tested=False, untestable_reason=f"unknown kind {kind!r}")

    is_topic_kind = kind.endswith("_topic")
    topic = h.get("topic") or []
    if is_topic_kind and not topic:
        return dict(h, tested=False, untestable_reason=f"{kind} requires a non-empty topic")

    # Retrieval step: "_topic" kinds filter down to the topic-matched subset first;
    # "_general" kinds use the full corpus as-is (no retrieval filter).
    sub_a = filter_by_topic(texts_a, topic) if is_topic_kind else texts_a
    sub_b = filter_by_topic(texts_b, topic) if is_topic_kind else texts_b
    chars_a, chars_b = total_chars(sub_a), total_chars(sub_b)

    if kind in ("trend_general", "trend_topic") and h.get("lexicon") and h.get("expected_change"):
        result = test_trend(sub_a, sub_b, chars_a, chars_b, h["lexicon"], h["expected_change"])
    elif kind in ("contrast_general", "contrast_topic") and h.get("lexicon_early") and h.get("lexicon_late"):
        result = test_contrast(sub_a, sub_b, chars_a, chars_b, h["lexicon_early"], h["lexicon_late"])
    else:
        return dict(h, tested=False, untestable_reason="missing lexicon field(s)")
    return dict(h, tested=True, n_articles_a=len(sub_a), n_articles_b=len(sub_b),
                chars_a=chars_a, chars_b=chars_b, result=result)


def effective_verdicts(h):
    """The single confirm/deny/inconclusive verdict per metric used for voting -- for a trend
    hypothesis this is just its own verdict; for a contrast hypothesis it's confirm if EITHER the
    independent or dominance method confirms (deny if either denies and neither confirms). Returns
    (presence_verdict, chars_verdict); only call on a hypothesis with tested=True."""
    r = h["result"]
    if h["kind"].startswith("trend"):
        return r["presence"]["verdict"], r["chars"]["verdict"]
    out = []
    for metric in ("presence", "chars"):
        ind, dom = r[metric]["independent"]["verdict"], r[metric]["dominance"]["verdict"]
        if ind == "confirm" or dom == "confirm":
            out.append("confirm")
        elif ind == "deny" or dom == "deny":
            out.append("deny")
        else:
            out.append("inconclusive")
    return tuple(out)


def tally_votes(hyps):
    """Confirm-only voting across a question's (tested) hypotheses, for both metrics independently.
    Returns dict(votes_presence, votes_chars, pred_presence, pred_chars) where pred is 'A', 'B', or
    'tie'. Untested hypotheses are skipped."""
    votes_p, votes_c = {"A": 0, "B": 0}, {"A": 0, "B": 0}
    for h in hyps:
        if not h.get("tested"):
            continue
        pv, cv = effective_verdicts(h)
        d = h["direction"]
        if pv == "confirm":
            votes_p[d] += 1
        if cv == "confirm":
            votes_c[d] += 1
    pred_p = "A" if votes_p["A"] > votes_p["B"] else "B" if votes_p["B"] > votes_p["A"] else "tie"
    pred_c = "A" if votes_c["A"] > votes_c["B"] else "B" if votes_c["B"] > votes_c["A"] else "tie"
    return dict(votes_presence=votes_p, votes_chars=votes_c, pred_presence=pred_p, pred_chars=pred_c)


def run(dataset="pilot"):
    print("loading NYT 2006/2011 into memory...")
    texts_a, texts_b = load_nyt_texts(YEAR_A), load_nyt_texts(YEAR_B)
    print(f"  {len(texts_a)} articles ({YEAR_A}), {len(texts_b)} articles ({YEAR_B})")

    data = json.load(open(RESULTS / f"lexicon_hypotheses_{dataset}.json"))

    counts = {k: 0 for k in VALID_KINDS}
    n_untestable = 0
    out = []
    for d in data:
        row = dict(country=d["country"], question=d["question"], hypotheses=[])
        for h in d.get("hypotheses", []):
            scored = score_hypothesis(h, texts_a, texts_b)
            if scored["tested"]:
                counts[scored["kind"]] += 1
            else:
                n_untestable += 1
            row["hypotheses"].append(scored)
        out.append(row)

    out_path = RESULTS / f"lexicon_hypotheses_{dataset}_tested.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nhypotheses by kind: {counts}, {n_untestable} untestable")
    print(f"-> {out_path}")


if __name__ == "__main__":
    import sys as _sys
    run(_sys.argv[1] if len(_sys.argv) > 1 else "pilot")
