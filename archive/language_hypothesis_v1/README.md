# Archived: free-text language-hypothesis experiment

Archived 2026-11-05. Marked invalid/superseded, not because of a bug, but because the whole
methodology was superseded by the structured lexicon-hypothesis experiment
(`code/value_shift_indicators/{run,test}_lexicon_hypotheses*.py`), which replaced free-text hypotheses
with a fixed four-kind JSON schema so markers never had to be inferred from prose after the fact. Final
result here (round 1-5 combined, 128 USA wave5->6 questions): 68/122 decided-only correct (55.7%,
z~1.27, p~0.20) -- not statistically significant, i.e. indistinguishable from chance. The lexicon-based
successor reached the same "not significant" conclusion through a cleaner pipeline, so this round is
superseded rather than disproven; nothing here has been deleted.

Paths inside this code still point at the old locations (`code/value_shift_indicators/...`), so nothing
here will run as-is -- same situation as `archive/llm_prediction_v1/`. Two functions this code defined
(`load_nyt_texts`, `load_ground_truth`, both in the archived `test_language_hypotheses.py`) turned out to
be genuinely shared infrastructure the lexicon-hypothesis pipeline also needs -- those were moved to
`code/value_shift_indicators/analysis_tools.py` before archiving, not copied, so there is exactly one
live copy.

## What is here

- `code/pilot_language_hypotheses.py` -- the very first pilot (one question, F118, no measurement step).
- `code/run_language_hypotheses_bulk.py` -- round 1, scaled to all 128 usable USA significant changes.
- `code/run_language_hypotheses_round2.py`, `run_language_hypotheses_round_n.py` -- the feedback-loop
  rounds (2, then a generic round-N script for 3-5), re-querying whatever was still tied.
- `code/test_language_hypotheses.py`, `test_language_hypotheses_round2.py`,
  `test_language_hypotheses_round2_extra.py`, `test_language_hypotheses_round_n.py` -- the mechanical
  scorer (markers extracted from quoted phrases in the free-text hypothesis, tested against real NYT
  text) for each round.
- `prompts/{system,user}/generate_language_hypotheses.txt` (round 1) and `..._round2.txt` (rounds 2-5,
  reused unchanged) -- both approved before use at the time.
- `results/language_hypotheses_bulk.json` / `_tested.json` (round 1) and
  `language_hypotheses_round{2,2_extra,3,4,5}.json` / `_tested.json` (feedback rounds) -- all raw
  generations and scored results, kept in full.

## Known problems found before archiving (see STATUS.md's full history for details)

- Markers were inferred post hoc by regex-extracting quoted phrases out of the hypothesis's own
  sentence; ~4% of hypotheses had no extractable marker and were left untested, and some "confirmed"
  hypotheses were really testing only a fragment of what the sentence claimed.
- A hypothesis pairing a rising term with a falling term (e.g. "'cohabiting couples' rises while 'living
  in sin' declines") had all its markers dumped into one undifferentiated bucket, not tested as the
  contrast it actually claimed -- this is exactly the gap the lexicon-hypothesis schema's `trend` vs.
  `contrast` kinds were designed to close.
- Accuracy climbing round-over-round (53.1% -> 55.7%) was the shrinking tie bucket, not the model
  getting better with more tries -- newly-decided questions each round stayed at chance.
