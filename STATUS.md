# Status

Inventory of what exists in this repo; the dated notes below are the running log (last updated 2026-11-05). No research question is
settled yet — this is a snapshot to work from, not a plan. Update this doc as code/data/results change.

> **Chars-per-100k significance bug found and fixed (2026-11-05, `test_lexicon_hypotheses.py`).** Found by comparing, question
> by question, where the earlier free-text language-hypothesis experiment was correct and this structured one was wrong (20
> such cases). **The bug:** the chars metric's significance test ran a two-proportion z-test with `n` = total corpus
> *characters* (hundreds of millions for an unrestricted hypothesis, e.g. `n_a=117,867,391`). At that scale the test is
> absurdly over-powered -- almost any word whose raw count didn't shrink by exactly the corpus's own shrinkage ratio
> (2006's corpus is ~2.8x 2011's in characters) came back "significant" (`p=0.0` or `p≈1e-15`) regardless of real relevance.
> **Fix (`chars_from_presence()`):** the chars metric's *direction* still comes from the occurrences-per-character rate, but
> its *significance* is no longer tested independently against the huge character total -- it reuses the matching
> presence-based (article-count) test's p-value, exactly how the original free-text experiment's own chars-rate fix worked
> (char rate decided direction only, never significance, there either). Verified offline with a fabricated huge-exposure
> fixture before rescoring anything for real. Rescored pilot, full 128-question bulk run, and the entire feedback-loop
> history (no new LLM calls -- same generated hypotheses, corrected test only).
> - **Accuracy barely moved (49.6%-52.8% both before and after, still chance either way) -- revealing a SECOND, different
>   problem the fix exposed rather than solved.** Re-ran the same 20 "old-experiment-correct, new-experiment-wrong"
>   diagnostic after the fix: **all 20 are still wrong.** Inspecting why: many confirmed hypotheses are now significant
>   under the FIXED, article-count-based test too (e.g. A002's Facebook/Twitter/social-network lexicon, p=2.5e-05; E218's
>   unemployment/recession lexicon, p=0.0) -- these aren't statistical artifacts anymore, they're real, large, correctly-
>   detected trends in NYT coverage 2006->2011. The problem is relevance, not significance: social media's real adoption
>   curve and the 2008 financial crisis are huge secular trends that coincide with this exact time window and get picked up
>   by any coarse word-frequency test, regardless of whether they have any causal connection to the specific WVS attitude
>   being tested (A002 is about friendship quality, not social media; A001 is about family importance, not the 2008 crisis).
>   **No p-value threshold fixes a confound** -- this needs either more specific/targeted lexica or a mechanism that reads
>   content for relevance rather than counting raw frequency (the LLM-judge/RAG idea discussed but not built: see the
>   "simple keyword-based RAG" discussion below).
> - **Final numbers with the fix, all 128 questions:** round 1 alone, chars-per-100k 55/107 decided (51.4%, z=0.29, p=0.77);
>   round 1 + feedback loop combined, chars-per-100k 60/121 decided (49.6%, z=-0.09, p=0.93, i.e. indistinguishable from
>   chance in either direction). presence-fraction unchanged throughout (not touched by this bug): 53/101 (52.5%) round 1
>   alone, 62/123 (50.4%) with feedback. 9 questions still tied (either metric) after feedback, up slightly from 7 (the fix
>   made the chars metric more conservative, so a few more hypotheses land as "inconclusive" instead of a spurious confirm).

> **Structured, lexicon-based hypothesis experiment (2026-11-05, new, `code/value_shift_indicators/`; prompt approved before
> use).** Follow-up to the language-hypothesis experiment below, constraining the hypothesis *type* instead of leaving it fully
> open-ended: each hypothesis is a fixed JSON shape, not a free-text sentence with markers inferred post hoc. New files:
> `prompts/{system,user}/generate_lexicon_hypotheses.txt`, `run_lexicon_hypotheses.py` (5-question pilot, dry run by default),
> `run_lexicon_hypotheses_bulk.py` (full 128-question set, dry run by default), `test_lexicon_hypotheses.py` (mechanical
> scoring, no LLM calls, takes a dataset name arg -- `pilot` default or `bulk` -- reuses
> `test_language_hypotheses.load_nyt_texts`/`total_chars` and `analysis_tools.two_proportion_z`).
> - **Schema, exactly four kinds (fixed same day after an initial looser version -- see below):** `trend_general` (one
>   lexicon, frequency change across ALL coverage), `trend_topic` (one lexicon, frequency change only within coverage matching
>   a required `topic` word/phrase list -- `topic` retrieves the relevant articles first, `lexicon` is checked within them: a
>   simple keyword-based retrieve-then-check pipeline, no LLM involved in the checking step), `contrast_general` (two lexicons,
>   one dominates near year_a and the other near year_b, across all coverage), `contrast_topic` (same crossover, only within
>   topic-matched coverage). `topic` is a required field for the two "_topic" kinds and does not exist at all for the two
>   "_general" kinds -- not an optional add-on either way, so the model must commit to whether a pattern is topic-restricted.
> - **Initial version was looser and got corrected before the real run (explicit user catch):** the first prompt draft had
>   only two types (trend/contrast) with `topic` as an *optional* field models could leave `[]` by choice -- this let the
>   model dodge committing to whether a pattern was topic-restricted, and 3/5 of the first pilot's hypotheses did exactly that.
>   Rewrote the prompt to the four named kinds above and `test_lexicon_hypotheses.py` to dispatch on `kind` (a "_topic"
>   hypothesis with an empty/missing `topic` is now marked untestable rather than silently treated as unconditioned); checked
>   offline against hand-built fake text for all four kinds plus the missing-topic/unknown-kind edge cases before any real call.
> - Every hypothesis is scored by **both** rate metrics side by side (not merged into one verdict): article-presence fraction
>   (fraction of matching articles) and raw occurrences per 100,000 characters. Contrast hypotheses are additionally scored by
>   **both** of two methods, each under both rate metrics (4 verdicts total): `independent` (two separate trend tests, one per
>   lexicon, confirmed only if both hold) and `dominance` (one test on the early lexicon's *share* of early+late mentions,
>   expected to fall over time). All of this reuses `two_proportion_z` exactly as-is -- no new statistic.
> - **5-question pilot re-run under the fixed schema (claude-haiku-5-5):** F118/A001/A168A/A165/A124_02, all 25 hypotheses
>   testable, each question got the required mix of all 4 kinds. Needed `max_tokens=3000` (not the pilot's original 1500 --
>   the richer 4-kind schema truncates more often than the old free-text one; same per-failure-retry fix as before, not a
>   shared-default bump). **Confirm-only decided-only accuracy, n=5: presence-fraction 2/4 (50%), chars-per-100k 2/4 (50%).**
> - **Scaled to the full 128-question USA wave5->6 set (`run_lexicon_hypotheses_bulk.py`, claude-haiku-5-5, `max_tokens=3000`
>   default since the pilot showed 1500 truncates about half the time under this schema):** 126/128 succeeded first pass, 2
>   truncated even at 3000 and were retried individually at 4500. 632 hypotheses total (8 of 128 questions returned only 4 --
>   satisfied "at least one of each kind" literally without padding to 5 -- not treated as an error), 0 untestable.
> - **Full 128-question confirm-only decided-only accuracy: presence-fraction 53/101 (52.5%, z=0.50, p=0.62); chars-per-100k
>   57/111 (51.4%, z=0.28, p=0.78).** Both at chance, not significant -- same conclusion as every other approach tried in this
>   project (free-text language hypotheses: 55.7%; no-hypothesis baseline: 51.6%). The precise 4-kind schema did not improve
>   prediction accuracy over the looser free-text version at scale, either.
> - **Verdict counts by kind (confirm/deny/inconclusive), all 128 questions:** `trend_general` (n=241) presence
>   [92,49,100] chars [101,45,95]; `trend_topic` (n=134) presence [21,31,82] chars [58,13,63]; `contrast_general` (n=129)
>   presence [77,18,34] chars [82,18,29]; `contrast_topic` (n=128) presence [24,19,85] chars [31,42,55].
>   `contrast_general` confirms far more than it denies (the strongest signal of any kind) while both "_topic" kinds are
>   mostly inconclusive -- plausibly just lower statistical power from smaller topic-filtered article subsets, not evidence
>   topic-conditioning itself is a worse mechanism; not disentangled from sample size here.
> - **Configurable multi-round feedback loop built and run (`run_lexicon_hypotheses_feedback.py`, `--max-rounds` flag,
>   default 5; new prompt `prompts/{system,user}/generate_lexicon_hypotheses_feedback.txt`, shown and approved before use).**
>   Unlike the old free-text experiment's one-off round2/round_n scripts, this is one script, looped internally. A question
>   is "tied" (needs more hypotheses) if confirm-only voting has no majority under EITHER the presence-fraction or the
>   chars-per-100k metric (both must independently decide before a question drops out) -- explicit 2026-11-05 decision, since
>   this experiment tracks two metrics in parallel where the old one had only one. Each round feeds back every hypothesis
>   from every prior round (lexica, topic, kind, measured verdict under both metrics -- never the true direction) and asks
>   for 5 new ones, same four-kind schema. No forced final-round decision -- a question that never resolves just stays tied.
> - **Run for real: 31 of 128 questions tied after round 1.** Ties cut round over round: 31 -> 17 -> 15 -> 11 -> 8 across the
>   5 rounds. One question (D078) failed to parse on every single round (consistent truncation at `max_tokens=3000` --
>   same question, same failure, every time since its history never advanced) -- retried once at `max_tokens=6000`, resolved
>   to Direction A under both metrics. **Final: 7 of 128 still tied** (`A006, A098, A124_12, E023, F114A, F199, G007_33_B`).
> - **Final accuracy, all 128 questions, round 1 + feedback combined: presence-fraction 62/123 decided (50.4%, z=0.09,
>   p=0.93); chars-per-100k 66/125 decided (52.8%, z=0.63, p=0.53).** Both still at chance. The feedback loop did cut ties
>   substantially (31->7, a 77% reduction, comparable to the old free-text experiment's 46->6), but matches that same
>   experiment's core finding: resolving a tie is not the same as resolving it correctly -- newly-decided questions landed
>   close to a coin flip, not above it. No dashboard built for this experiment yet.

> **Value-shift indicator system (2026-11-02 to 2026-11-04, new, self-contained in `code/value_shift_indicators/`; built on an
> unmerged worktree branch `worktree-value-shift-indicators`).** Explores whether an LLM can predict which direction a WVS question
> moved using only independent evidence (NYT Archive metadata, GDELT US events) -- the LLM is never told it's WVS and (for the
> prediction test cases) never told the true direction. Three steps: `generate_indicators` (1 call: >=5 falsifiable indicators per
> direction, e.g. "NYT coverage of X should increase"), `measure_indicator`/`data_sources.py` (no LLM, pulls real NYT keyword-match
> rates and GDELT event-type rates for the indicator), `aggregate_judge` (1 call per round, up to 3 rounds: given all evidence so
> far, decide a direction or ask for up to 5 more indicators, forced to decide by round 3). All prompts approved before use per
> project convention; all calls through `fast_llm.py`, cached.
> - **First test case (2026-11-02): USA, F118 ("justifiable: homosexuality"), wave5->6 (2006->2011), gpt-4.1-mini.** Correctly
>   landed on the true direction (more accepting) in round 1, but the per-indicator confirm/deny design (10 independent judgments,
>   no aggregation) left no single answer -- redesigned to the aggregate-judge loop above.
> - **10-question run (gpt-4.1-mini): 5/10 correct, but decisions landed 8 "B" vs 2 "A"** despite a near-50/50 true split --
>   investigated with an order-bias probe (`test_order_bias.py`: same evidence, A/B labels swapped) and ruled out: only 2/10
>   flipped, so the skew was evidence-driven, not a position/order artifact. Leading suspect: GDELT's indexed event volume grew
>   ~10x from 2006 to 2011 (more digitized sources, not more real events), so "rate per 1,000 total events" indicators are
>   mechanically biased toward "decrease" regardless of topic.
> - **Schema expansion (2026-11-02), approved then added:** GDELT indicators can specify actor types, actor-name/location
>   substrings, Goldstein score, and an alternate `gdelt_baseline_codes` set (narrower denominator than "all events", to fight the
>   volume-growth bias above); NYT indicators can restrict to sections and now also match `keywords[].name` tags. Hard guard added
>   in `build_case()` against picking `year_a`/`year_b` on opposite sides of the GDELT 1.0/2.0 boundary (2015-02-18).
> - **20-question run (10 new + the original 10, gpt-4.1-mini): 8/20 correct.** Usage of the new fields was lopsided --
>   `gdelt_actor_types` and `nyt_sections` used on 100% of applicable indicators, but `gdelt_baseline_codes` (the volume-growth
>   fix) only 4% and `gdelt_actor_name_contains` (the only real topic-targeting mechanism GDELT has) only 2% -- the model mostly
>   skipped the two fields meant to fix the diagnosed problems. **Also found: `REL` (religious), invented for the
>   `gdelt_actor_types` prompt vocabulary, is not a real GDELT actor-type code** (checked a real month's distinct values: GOV, BUS,
>   EDU, COP, JUD, CVL, MED, LEG, MIL, MNC, HLH, LAB, CRM, ... -- no REL). 9 of 19 `REL`-filtered indicators across 4 questions got
>   a silent, wrong 0/0 "no evidence" instead of actually being measured. Not yet fixed in the schema.

> **Direct hands-on evidence rounds (2026-11-02, by Claude, not the automated pipeline above): 15 USA questions total (5 then
> 10 more), real NYT/GDELT lookups, own reasoning and real-world historical knowledge used openly (not blind).** Findings:
> - **Media volume vs. survey-measured sentiment can point in exactly opposite directions.** Life satisfaction rose slightly
>   2006->2011 despite recession/unemployment-crisis NYT coverage rising +686%; confidence in companies rose despite Occupy Wall
>   Street coverage spiking 0.06->4.05/1k; belief that "hard work brings success" strengthened even as "American Dream"-framed
>   coverage *fell*. Three independent cases, all economic-attitude questions -- political/cultural/religious questions (Tea
>   Party/debt-ceiling, the 2010 Park51/"Ground Zero mosque" controversy, gay-rights coverage) tracked the true direction well.
> - **GDELT's institutional-actor disapprove/reject rate failed on every domestic-institutional-confidence question tested (3/3:
>   confidence in government, Congress, political parties), while NYT confirmed all three of the same questions.** Likely cause,
>   untested: the raw-events partition filters by event *location* (US soil) not actor *nationality*, so a "GOV actor disapproves"
>   event can be a foreign government's statement made in the US (UN, embassies) rather than Americans disapproving of their own
>   government. `Actor1CountryCode`/`Actor2CountryCode` columns exist in the stored data and were never filtered on.
> - **NYT keyword search is brittle to exact phrasing.** `["Wisconsin collective bargaining", "Scott Walker union"]` returned 0/0
>   for 2011 despite the Wisconsin Act 10 fight being one of the biggest domestic stories that year; single looser terms
>   ("Wisconsin", "Scott Walker", "collective bargaining") found it immediately (28-209 matches). Confirmed independently: GDELT's
>   `REL` actor-type bug above.
> - Confirmed real-world knowledge is doing real work here: several hand-picked indicators (Tea Party, debt-ceiling standoff,
>   Park51) and one *independently* generated one in the later language-hypothesis round converged on the same specific events --
>   but this also means the exercise shows the data *contains* a detectable signal when you already know what to look for, not
>   that a blind system could *find* it (see explicit discussion in conversation; not written up as code).

> **Claude/Anthropic support added to `code/fast_llm.py` (2026-11-04).** Dispatches on model-name prefix (`"claude-"` vs default)
> to the Anthropic Messages API or the existing OpenAI path; same `run_llm`/`plan_llm` call either way. Claude calls require
> `max_tokens` explicitly (no API default) and `ANTHROPIC_API_KEY`. Ported into the isolated worktree from an uncommitted copy in
> the shared checkout -- **not yet committed on `main`.** `anthropic` added to the environment (installed via pip in this
> session; not yet added to `environment.yml`).

> **Language-hypothesis experiment (2026-11-04, new prompt, new scope, also in `code/value_shift_indicators/`): can an LLM
> predict value shifts from patterns in *how* people write, not just what gets covered?** Modeled loosely on the
> post-9/11 "I" -> "we" pronoun-shift finding (Cohn, Mehl & Pennebaker 2004), but the model is given no worked example, just an
> explanation of the hypothesis category, to avoid anchoring it to one pattern type. `generate_language_hypotheses` prompt: given
> a value shift (blind to true direction, no "markers"/operationalization forced -- deliberately deferred), produce 5 hypotheses
> about language patterns in NYT coverage. Piloted on F118 with `claude-haiku-5-5`: 5 genuinely style/framing-flavored hypotheses
> (e.g. "equality/civil-rights" vs. "sin/immoral" framing; one specifically about quote-attribution, not just word frequency).
> - **Scaled to all 128 USA wave5->6 significant changes with a usable (two-ended) scale** -- `bulk_cases.py` builds attitude
>   descriptions mechanically from the codebook (`parsed_question` + `choices` in `parsed_survey_data_cleaned.jsonl`), filtered via
>   `question_categories.csv` to `ordinal_scale`/`numeric_scale`/`binary`/`mention_flag` formats (not hand-authored per question
>   like `cases.py`). Checked real per-country wave5/6 fieldwork years for the usual 5-country set (EGY 2008/2013, THA 2007/2013,
>   UKR 2006/2011, URY 2006/2011, HKG 2005/2014) but **scoped this run to USA only**: NYT is a plausible domestic-discourse proxy
>   for the US but not for the other 5 (foreign-correspondent coverage of another country is a different, weaker premise).
>   128/128 succeeded (claude-haiku-5-5, 643 hypotheses, ~15s wall-clock; 2 initial JSON truncations at `max_tokens=1500`, retried
>   individually at 2500 and patched in rather than bumping the shared default and invalidating the other 126's cache).
>   Published as a browsable artifact (private): https://claude.ai/artifact/9ieYsZZgwq6MBTQgzGSf8c.
> - **Quick mechanical test against real NYT text, no new LLM calls** (`test_language_hypotheses.py`): markers extracted from each
>   hypothesis's own quoted example words/phrases (616/643 had >=1; the rest are pure structural claims, left untested). Two
>   real bugs found and fixed: (1) `document_type != "article"` entries (multimedia/audio, ~textless) grew from ~2% to ~11% of the
>   NYT archive 2006->2011, inflating the denominator -- filtered to articles only; (2) marker extraction kept trailing
>   punctuation from quotes like `'crucial,'` -- stripped. **Result: 101 confirm / 219 deny / 296 inconclusive; only 59/128
>   questions reached a non-tie predicted direction, accuracy on those 28/59 (47.5%, roughly chance).** Root cause of the
>   persistent deny-heavy skew: NYT's indexed article count fell ~42% 2006->2011 (160,430 -> 92,647) even restricted to articles,
>   so most quoted-phrase rates mechanically decline regardless of topic, while ~90% of generated hypotheses predict "increase" --
>   a structural mismatch between the hypotheses and this test, not necessarily evidence the hypotheses are wrong. Known
>   remaining weakness: some extracted markers are generic dictionary words quoted for emphasis (e.g. "friends" alone, 3,863
>   matches) -- not fixed, needs a specificity judgment call rather than a quick regex tweak.

> **Language-hypothesis normalization fix + multi-round feedback loop (2026-11-04, same day, continuation).** Root-caused the
> deny-heavy skew above: NYT text per article shrank from ~734 chars (2006) to ~460 chars (2011), more than the ~42% article-count
> drop alone -- `test_language_hypotheses.py` now rates matches per 100,000 corpus *characters* instead of per article
> (`total_chars()`); confirmed by the confirm/deny balance flipping (101/219 -> 218/102) and more questions deciding (59 -> 82/128).
> **Scoring correction (explicit user call): a "deny" verdict does NOT vote for the opposite direction** -- only "confirm"
> casts a vote; a direction wins a question only if it has strictly more confirmed hypotheses than the other side, else "tie".
> Confirm-only on round 1 alone: 42 correct / 40 wrong / 46 tied (accuracy on the 82 decided: 51.2%, chance).
> - **Round 2 (new prompt, approved before use, `prompts/{system,user}/generate_language_hypotheses_round2.txt`):** for every
>   question still tied, feed the model its own prior hypotheses with their real measured verdict/rates (not the true direction)
>   and ask for 5 new ones. Ran on all 46 confirm-only ties (`run_language_hypotheses_round2.py` + a second pass,
>   `run_language_hypotheses_round_n.py`, for the 28 that the first pass's deny-based tie list had missed -- see
>   `run_language_hypotheses_round_n.py:load_cumulative()` for the exact, bug-fixed cumulative-vote logic). Combined round 1+2,
>   confirm-only: 52 correct / 49 wrong / 27 tied (decided-only accuracy 51.5%, chance; z~0.30, p~0.76).
> - **Rounds 3-5 (same reused prompt template, cumulative hypothesis history each round, run by a background agent):** only
>   re-queried whatever was still tied going into that round (27 -> 14 -> 9 -> 6). **Final result, all 128 questions, rounds
>   1-5 combined: 68 correct (53.1%) / 54 wrong (42.2%) / 6 tied (4.7%); decided-only accuracy 68/122 = 55.7% (z~1.27 vs 50%,
>   p~0.20 -- directionally above chance but not significant).** Of the 6 still tied after 5 rounds: 3 have never had a single
>   hypothesis confirmed in either direction across all 5 rounds (`A124_42`, `E025B`, `F121`), 3 are genuine nonzero ties
>   (`D060` 1-1, `E012` 2-2, `E222B` 3-3). **Takeaway: 5 rounds cut the tie rate 46->6 (87%), but accuracy on each round's
>   newly-decided questions stays at chance -- the climbing overall-accuracy number is the shrinking tie bucket, not the model
>   getting better with more tries.** New files (not committed): `run_language_hypotheses_round2.py`,
>   `run_language_hypotheses_round_n.py`, `test_language_hypotheses_round2(_extra|_round_n).py`,
>   `results/language_hypotheses_round{2,2_extra,3,4,5}(_tested)?.json`.
> - **Dashboard updated:** https://claude.ai/artifact/9ieYsZZgwq6MBTQgzGSf8c now shows, per question, a correct/wrong/tie
>   status chip, a small two-row bar chart (Direction A / Direction B, segments = confirm/deny/inconclusive/untested
>   hypothesis counts, colors matched to the per-hypothesis verdict badges), percent + a client-side chance-test
>   p-value on the summary tiles, and a shared legend (versions 2-6; dataviz skill's status-color convention used
>   throughout -- color is never the only cue, counts/symbols always sit next to it).

> **No-hypothesis direct baseline for comparison (2026-11-04/05): claude-haiku-5-5 asked to just predict the
> direction with no NYT/GDELT evidence at all** -- the same style of baseline already run for EGY/THA/UKR/URY/HKG
> with gpt-4.1-mini etc. (`run_direction_baseline_set.py`), now for the same 128 USA questions as the
> language-hypothesis experiment above (`code/run_direction_baseline_usa128.py`, not committed). Reused
> `direction_baseline.py`'s existing, already-approved prompt unchanged.
> - **Found a real mismatch first:** `direction_baseline.py`'s own significance gate (`change_detection.py`) only
>   keeps 13 of the 128 -- the 128 were actually selected by a different, older significance test
>   (`WVS_dataloader.py`'s per-respondent KS test, `data/significant_changes/ks/*.csv`, p<0.05). Per explicit user
>   decision, the new script bypasses `change_detection.py`'s gate entirely and builds truth straight from the KS
>   csvs for all 128, reusing `build_prompt()`/`parse_answer()` unchanged.
> - **claude-haiku-5-5 needed `max_tokens=800`, not the 250 the gpt baselines use** (and rejects an explicit
>   `temperature` entirely, unlike every OpenAI model run so far) -- at 250, 118/128 responses were cut off mid-
>   reasoning before the required `ANSWER:` line; at 500, still 24/128; at 800, 128/128 parsed.
> - **Result: 66/128 correct (51.6%), z=0.35 vs. chance, p=0.72 -- not significant**, i.e. indistinguishable from
>   guessing. Essentially the same accuracy as the full 5-round NYT-hypothesis pipeline above (53.1%/55.7%
>   decided-only, itself only marginally closer to significant at p=0.20) -- feeding in measured NYT language
>   evidence has not yet beaten the model just guessing from its own background knowledge.
> - **The raw survey CSV (`data/WVS_Time_Series_1981-2022_csv_v5_0.csv`, gitignored, 1.4 GB) is not in this
>   worktree** (worktrees don't get gitignored files) -- symlinked from the main checkout's `data/` for this
>   session; a fresh worktree needs the same symlink or a copy before `direction_baseline.py`-based scripts work.
> - **Alternate scoring tried (recomputed from existing stored verdicts, no new measurement): deny as a -0.5
>   penalty on its own hypothesis's direction** (confirm +1, deny -0.5, inconclusive 0; net score decides) instead
>   of confirm-only voting. Result: 67 correct / 58 wrong / 3 tie (vs. 68/54/6 confirm-only) -- fewer ties but
>   accuracy *drops slightly* (53.6% decided-only vs. 55.7%, p=0.42 vs. 0.20). Not adopted; confirm-only remains
>   the scoring rule in use.
> - **Known unresolved methodology gap, found while explaining a specific hypothesis** (the cohabiting-couples one
>   for D060): `extract_markers()` pulls every quoted phrase out of a hypothesis with no idea which clause it
>   belongs to, so a hypothesis that pairs a rising term with a falling term (e.g. "'cohabiting couples' rises
>   while 'living in sin' declines") gets all markers dumped into one undifferentiated bucket and tested as a
>   single combined rate -- not what the hypothesis actually claims. Also no context/topic conditioning at all
>   (a hypothesis asking for a term "in housing/census contexts" is matched anywhere in the article). Not fixed;
>   likely affects a meaningful fraction of the 1,123 generated hypotheses, not just this one.
> - **Proposed but not yet run:** a forced-final-decision prompt (new prompt, drafted and shown to the user for
>   approval, not yet confirmed/sent) for the 6 questions still tied after 5 rounds (`A124_42`, `E025B`, `F121`
>   at 0-0 confirmed-either-way; `D060` 1-1, `E012` 2-2, `E222B` 3-3) -- feed back every hypothesis from every
>   round and force a single A/B answer instead of proposing more hypotheses.

> **Row-level GDELT pull, "Step 2" (done 2026-10-29): `code/gdelt_pull_raw_events.py`, full range 2003-11-01..2023-07-31, all 29 WVS
> countries.** Row-level (not aggregated) events via `events_raw()`: 18 of the 23 columns in `gdelt_bq.ALLOWED_COLUMNS` (drops
> `MonthYear`, `Year`, `EventBaseCode`, `EventRootCode`, `QuadClass` -- exact derivations of `SQLDATE`/`EventCode`, verified to add
> nothing). Stored at `RAW_EVENTS_DIR` (new `config.py` entry; this machine's `config.local.toml` points it at
> `/depot/dgoldwas/data/tchakrab/gdelt-data/raw-events`), partitioned `country=<FIPS>/year=<YYYY>/month=<MM>.parquet` (6,873 files).
> **Final result: 497,631,396 rows, ~26 GB on disk, verified against the independent aggregated sweep (`events_daily_*.parquet`):
> 22/29 countries exact row-count match, the other 7 off by 1-23 rows out of 1.3M-2.2M each (all in October 2016 -- present in the very
> first scan before any of the fixes below, most likely GDELT backfilling/revising a handful of events between pull times; not a
> pipeline bug, not chased further).** Run as Slurm batch jobs (`slurm/job.sbatch`, csml/a30/standby, `--gres=gpu:1` required even
> though this is pure CPU/network work -- see `slurm/README.md`) specifically so the multi-hour pull survives the submitting
> terminal/session closing.
> - **Three real incidents during this, each corrected by hand in the local ledger (billed bytes do not revert just because the
>   local process crashed) and all now fixed and re-verified:**
>   1. A first attempt pulling a whole year in one query was billed 18.66 GB then OOM-killed (cgroup limit ~9.5 GB RSS) converting
>      results to a Python list-of-dicts, before writing anything or updating the ledger. Fixed by chunking to one calendar month at
>      a time, written and released before the next, plus a soft `RLIMIT_AS` cap (`--mem-cap-gb`, default 7 GB for interactive
>      sessions, 12 GB used under the 20G Slurm allocation) so an overrun raises a catchable `MemoryError` instead of a silent
>      SIGKILL.
>   2. The first retry (chunked) was billed 1.73 GB then crashed on a missing `db_dtypes` package needed by the now-safer
>      `to_dataframe()` conversion path (`gdelt_bq.py`: new `run_to_dataframe()`/`fetch_dataframe()`, additive, does not touch the
>      existing `run()`/`fetch()` used by the aggregated pipelines). Fixed by installing it and verifying offline before retrying;
>      the retry was a BigQuery cache hit (0 GB billed). `db-dtypes` and `google-cloud-bigquery-storage` added to `environment.yml`.
>   3. A `MemoryError` on one chunk crashed the *entire* multi-year job (only the fetch step was wrapped in `try/except`, not the
>      subsequent pandas transform where it actually hit) -- fixed to wrap the whole per-chunk body and skip-and-continue instead of
>      stopping the run. Separately, that same month (Feb 2015, the one month `chunks()` splits across the GDELT 1.0/2.0 table
>      boundary) had *already* been silently corrupted before this: both halves of the split month wrote to the same output file,
>      so the second piece overwrote the first's data (~half the month's rows lost, all 29 countries) -- caught by the
>      cross-check against the aggregated sweep, fixed by grouping pieces by calendar month before writing.
>   4. **Self-inflicted, not a real bug:** scattered duplicate `GLOBALEVENTID` rows (141 total, several otherwise-correct months)
>      were wrongly assumed to be a GDELT data-quality issue and deduplicated -- this was wrong. Cross-checking against the
>      aggregated sweep's `COUNT(1)` afterward showed the "duplicates" are part of the true, expected count (GDELT's table
>      legitimately has more than one row for some event IDs); removing them *created* a mismatch that hadn't existed before.
>      Reverted (`to_frame()` no longer drops them) and the 7 affected months were re-fetched (free cache hits) to restore them.
> - **BigQuery Storage API set up and verified (2026-10-29):** `create_bqstorage_client=True` in `run_to_dataframe()` -- ~80x faster
>   result-fetching (3.6M rows in 17.9s vs. ~24 min via REST paging), confirmed zero additional cost on both a cache-hit replay
>   (`cache_hit=True`, 0.0000 GB billed) and a fresh, uncached query (billed normally, no extra charge). Needed, one-time in the
>   Cloud console: enable the BigQuery Storage API + grant the service account `BigQuery Read Session User` (in addition to Job
>   User) -- confirmed this did NOT require re-attaching billing to the sandbox project. Full details in `GDELT_BIGQUERY.md`'s new
>   "Row-level pulls" section.
> - Month total after all of this: ~292 GB of 1,100 GB free (800 GB self-imposed cap).

> **NYT Archive API pull (done 2026-10-29): `code/nyt_archive_pull.py`, full WVS range, metadata only.** One call per calendar month
> returns every NYT article for that month (`https://api.nytimes.com/svc/archive/v1/{year}/{month}.json`) -- confirmed via NYT's own
> OpenAPI spec that there is no full article body text field, just headline/abstract/snippet/keywords/section/byline/word_count/etc.
> No per-country filter exists on the endpoint (returns all NYT content, undifferentiated); the `keywords[].name == "glocations"` tag
> was checked against a sample month and does cover the 29 WVS countries (28/29 had at least one hit in the one month spot-checked;
> CYP had zero that month specifically). Stored at `NYT_ARCHIVE_DIR` (new `config.py` entry; this machine's `config.local.toml`
> points it at `/depot/dgoldwas/data/tchakrab/nyt-archive`), one raw JSON file per month. Run as a Slurm batch job for the same
> terminal-survival reason as the GDELT pull. **Result: 237/237 months fetched, 0 errors, ~52 minutes at a conservative 5 calls/min
> pace** (the actual current rate limit is unconfirmed -- developer.nytimes.com was not reachable to verify; third-party sources
> disagree from 5/min+500/day to 10/min+4000/day -- 5/min was used to be safe regardless of which is correct). API key via
> `NYT_API_KEY` in the environment, never in a file. Not yet done: any filtering/use of the pulled metadata.

> **2026-09-24: the LLM prediction code, prompts and results described below (the "LLM pipelines" section and
> most of "Results") were moved to `archive/llm_prediction_v1/` (see its README) because the results are not
> trusted. Paths below refer to their old locations. New LLM baselines will be built fresh.**

> **LLM calling:** use only `code/fast_llm.py` (`run_llm(requests, model=...)`). The old `thread_gpt_suite/` and `llm/` were archived to `archive/llm_calling_v1/` (~5.7x slower on a 50-request benchmark). fast_llm: asyncio,
> self-tuning concurrency (cuts on 429, honors Retry-After, ramps back up). Tested against a
> simulated rate limit; used for the five-country direction run (697 calls in 23 s, no throttling).

> **LLM cache:** `code/llm_cache.py` + `fast_llm.py`. Every call is stored (full raw response + the exact request) in
> `llm_cache/<model>/<xx>.jsonl`, keyed by model + messages + generation params + `replicate`; a cached call is never
> re-sent. `plan_llm()` shows hits/misses without calling. JSONL not SQLite because the repo is on NFS.
> Tested offline with a fake model (hits, misses, changed params, duplicates, failures, corrupt lines).

> **Direction baseline (2026-09-24):** `code/direction_baseline.py` (prompts in `code/prompts/{system,user}/direction_baseline/`). Given
> country + question + the two waves' survey years, an LLM predicts the direction of change; prompt differs by question
> family (ordered / share / option). Truth = `change_detection.py` result. Only questions with a significant change are run.
> (An earlier unrequested Egypt test run was deleted; nothing from it should be cited.)
> **Dashboard "LLM baseline" tab:** artifact https://claude.ai/artifact/FkQt4BjPckXnMrqpH7u8Ww. Data file made by
> `python code/export_dashboard_baseline.py` (-> `results/direction_baseline/five_country_set/baseline_results.js`, then
> republish with `baseline.js`). Views: accuracy by country / wave pair / type / format / question group / true direction /
> size of change / individual question, with 95% intervals and a best-single-guess baseline; click-to-drill filters;
> per-question prompts and full replies. It contains the full prompts and model replies, so keep the artifact private.

> **Five-country run (done 2026-09-24):** `code/run_direction_baseline_set.py --run`: EGY, THA, UKR, URY, HKG on waves 5->6 and
> 6->7, changed questions only, gpt-4.1-mini, temperature 0; 697 calls, all parsed. Results in
> `results/direction_baseline/five_country_set/` (per-pair CSVs + `all_gpt-4.1-mini.csv`, full responses included); every call
> is in `llm_cache/`. Accuracy: 53.7% overall (scales 54.8%, yes/no-type 49.3%, option-type 39.3%); by country and wave pair
> it ranges 39% (THA 6->7) to 65% (HKG 6->7). Not yet analysed beyond these headline numbers.

> **More models (2026-09-30):** `run_direction_baseline_set.py --model gpt-5-mini|gpt-5 --reasoning off|on --run`, same 697
> questions and prompts. gpt-5* use default temperature, `max_completion_tokens` (1000 off / 6000 on) and reasoning effort
> minimal (off) / medium (on), passed via `extra_body` (installed openai SDK is too old). Accuracy: gpt-5-mini 53.5% off / 56.8% on,
> gpt-5 55.1% off / 56.1% on (gpt-4.1-mini 53.7%); differences are within noise. Results in
> `results/direction_baseline/five_country_set/by_model/<model>__reasoning-<off|on>/`. Not yet run: Claude (needs
> ANTHROPIC_API_KEY + support in fast_llm.py) and Qwen. **Dashboard:** "LLM baseline" tab now has a model comparison
> chart and selector; data via `python code/export_dashboard_baseline.py --models` (-> `baseline_models.js`); source of
> `index.html` / `baseline.js` kept in `code/dashboard/`.

> **Reasoning analysis (2026-10-16):** run in the existing conda env `wvs` (an attempt to create `wvs-shifts` from `environment.yml`
> never finished solving; that file is unused). `code/reasoning_tfidf.py` (TF-IDF + NMF, 12 topics) and `code/reasoning_embed.py`
> (text-embedding-3-small via the API, cached in `embedding_cache/`; k-means, k=6) over the 3,485 replies of the 5 model runs ->
> `results/direction_baseline/five_country_set/reasoning_analysis/`. Findings: embedding clusters follow country/history, so all models
> look alike; TF-IDF topics differ by model (gpt-4.1-mini: 40% of replies are one generic "significant political changes" phrase vs
> 0-4% for gpt-5 runs). Accuracy differences between topics not tested for noise. **Dashboard "Model reasoning" tab**
> (`code/dashboard/reasoning.js`, data from `code/export_dashboard_reasoning.py`) shows both views with example replies.

> **Per country / per theme (2026-10-16):** `reasoning_tfidf.py` now also fits an 8-topic model per country
> (`tfidf_country_*.csv`). `code/question_themes.py` groups the 192 questions into 13 hand-made themes (religion, gender and
> family, confidence in institutions, ...; no "children" theme exists because no child-quality questions had a significant
> change). The dashboard's Model reasoning tab has a country selector (own topics per country) and a "Related questions" section
> (country -> theme -> every model's answer and reasoning per question). Counts per country-theme are small (2-46 questions).

> **Option-order flip (2026-10-20):** same 5 OpenAI runs and prompts, with answer order reversed (`run_direction_baseline_set.py --flip`;
> `build_prompt(flip=True)`: A/B swapped for scale and yes/no, numbered option list reversed for pick-one). 3,485 calls, all parsed.
> Results in `results/direction_baseline/five_country_set_flipped/by_model/<model>/`. Accuracy original -> flipped: gpt-4.1-mini
> 53.7 -> 54.8, gpt-5-mini off 53.5 -> 54.2, on 56.8 -> 55.7, gpt-5 off 55.1 -> 54.8, on 56.1 -> 58.0. The predicted direction
> changed on 8-15% of questions per model. Not yet in the dashboard.

> **Flip reasoning check (2026-10-20):** `code/reasoning_flip_consistency.py` (crude: shared year/proper noun + first-sentence TF-IDF)
> and `code/reasoning_flip_judge.py` (gpt-4.1-mini judge, prompt `prompts/user/flip_judge/same_reason.txt`, 3,485 original-vs-flipped
> pairs, each once; results `five_country_set_flipped/flip_judgements.csv`). Judge says "same reason or event" for 99-100% of pairs
> where the answer stayed the same, but only 36% (gpt-4.1-mini), 19% (gpt-5 off), 7% (gpt-5 on), 6% (gpt-5-mini off), 3% (gpt-5-mini on)
> where it changed. Manual check by Claude of 32 judgements (20 random + 12 random changed-answer): 28 agree; all 4 disagreements are
> judge NO where the same event is cited for the opposite conclusion, so the changed-answer YES rates are probably underestimates.

> **gpt-5 judge (2026-10-20):** same prompt, gpt-5 at medium reasoning effort (`reasoning_flip_judge.py --model gpt-5`; results
> `flip_judgements_gpt-5.csv`). Same reason/event on changed-answer pairs: gpt-4.1-mini 67%, gpt-5 off 61%, gpt-5 on 48%, gpt-5-mini off 72%,
> on 63% (vs 3-36% from the gpt-4.1-mini judge); 97-100% on unchanged pairs. Against Claude's manual reading of 41 pairs, gpt-4.1-mini
> judge agrees 80%, gpt-5 judge 83% (not a meaningful difference; the 10 gpt-5 pairs were picked using the 4.1-mini verdicts).

> **GDELT cost notes (2026-10-27):** the reasoning behind the wrapper (how BigQuery bills, partitioned tables, v1/v2 table names,
> no-billing sandbox, setup on a new machine, open risks) is written up in [GDELT_BIGQUERY.md](GDELT_BIGQUERY.md).

> **GDELT BigQuery wrapper (added 2026-10-19 by another session, copied into main 2026-10-22):** `code/gdelt_bq.py` + offline tests
> `code/test_gdelt_bq.py` (10 pass in env `wvs`; both untracked in main; also on branch `worktree-gdelt-wrapper`, 3 commits, unmerged,
> worktree in `.claude/worktrees/gdelt-wrapper`). Cost-guarded access to GDELT events on BigQuery: no raw SQL (fixed templates:
> `events_by_month` = monthly counts per CAMEO root code with mentions, avg tone, avg Goldstein; `events_raw` = whitelisted columns),
> bounded partition filter, per-query cap 20 GB, monthly budget 800 GB (local ledger), result cache, dry run by default (`--run` to
> execute), refuses ranges straddling 2015-02-18 (GDELT 1.0 table before, 2.0 after; counts not comparable across them), max 366 days
> per query. Country codes ISO3 -> FIPS for EGY THA UKR URY HKG USA IND TUR. Dry run: `python code/gdelt_bq.py --country EGY
> --start 2011-01-01 --end 2011-03-31`. Needs `GOOGLE_APPLICATION_CREDENTIALS` -> the sandbox service-account key
> `gdelt-sandbox-512120-6f3eb941f0e3.json` in the repo root (gitignored; never commit it) and `google-cloud-bigquery` (installed in
> `wvs`). Only 2 real queries so far (both Egypt, monthly counts, Jan-Mar only): 2008 on the 1.0 table (0.18 GB) and 2016 on the 2.0
> table (1.37 GB). The `.gitignore` is itself untracked in main. Not connected to the WVS or LLM code; no decision yet on how GDELT
> would be used.

> **GDELT size test (2026-10-22):** all GDELT cache, ledger and pulled data now live in `/u/gorman2_s/tchakrab/gdelt_data` (env var
> `GDELT_DIR` overrides; the old ledger and cache were copied there). `gdelt_bq.py` gained `events_daily_stats` (per day, country and full
> CAMEO code: n, sums, sums of squares, min/max of mentions and tone) and country lists (all 29 WVS wave 5-7 countries have FIPS codes);
> `code/gdelt_pull.py` runs a pull and writes daily + weekly (Monday start, rolled up locally) parquet and csv to `pulls/`. Jan-Jun 2016,
> 29 countries: 3.29 GB billed (month total 4.83 GB of 1,100), 28.7M events, 255 event codes; daily 540,751 rows (parquet 27.9 MB, csv
> 91.6 MB), weekly 119,192 rows (parquet 7.7 MB, csv 21.1 MB). Full-range plan (Nov 2004 - Jul 2023, one query per year) not run yet.

> **GDELT full sweep (done 2026-10-22):** `code/gdelt_sweep.py` pulled daily statistics for the 29 WVS wave 5-7 countries, 2003-11-01 to
> 2023-07-31 (one year before wave 5 fieldwork start, through wave 7 fieldwork end), 23 chunk queries, 51.1 GB billed (month total
> 55.9 GB of 1,100). Result: 13,646,757 rows (day x country x full CAMEO code), 497.6M events, no duplicate keys, all 29 countries.
> Stored in `/u/gorman2_s/tchakrab/gdelt_data/pulls/events_daily_2003-11-01_2023-07-31.{parquet,csv}` (422 MB / 1.5 GB) plus per-chunk
> parquet in `pulls/chunks/`. Missing rows mean zero events (not zero-filled). Columns: date, country (FIPS), event_code, n_events,
> sum_mentions(+_sq), sum_sources, sum_articles, sum_tone(+_sq), min_tone, max_tone, goldstein. GDELT 1.0 table before 2015-02-18, 2.0 after
> (counts and tone not comparable across the switch). Step 2 (event-level detail with 33 columns) not started.

> **GDELT plots (2026-10-22):** `code/gdelt_plot_monthly.py` -> `gdelt_plots/` (repo dir): per country (EGY THA UKR URY HKG) one PNG with 20
> panels (CAMEO root categories, own y-axis each) of mean events per day averaged over each month, plus the monthly table as csv.
> Vertical line = GDELT 1.0 -> 2.0 switch (2015-02-18); levels jump there. One panel has one series, so no cross-country overlay yet.
> Added `--measure share` (default): each category as % of all events recorded for that country that month (`monthly_share_of_events_*`),
> to correct for GDELT recording more or less overall; the count plots (`monthly_mean_events_per_day_*`) are kept.

> **Event shares vs survey answers (2026-10-22):** `code/event_value_inputs.py` (event level = mean monthly share of each of the 20 CAMEO
> root categories in the country's survey year; survey anchor = % of valid respondents on the high side of the scale, for 363 ordinal / yes-no /
> select-many questions, unweighted, >=100 respondents) and `code/event_value_correlations.py` (Pearson r for every event x question pair in six slices:
> wave 5, 6, 7 across the 29 countries; waves pooled with wave effects removed; within country with country+wave effects removed; wave-to-wave
> changes; 100-shuffle permutation null with family-wise p). Output in `gdelt_plots/event_value_correlations/`: six PNG sheets, `correlations_long.csv`,
> `top_pairs.csv`, `event_levels.csv`, `anchors.parquet`. Result: observed |r| distributions are close to the shuffled null; only the strongest
> "changes" pair (11 Disapprove x "Secure in neighborhood", r = +0.81, n = 29, family-wise p = 0.02) clearly beats chance, out of ~2,500 pairs in that slice.

> **Same analysis on change over time (2026-10-22):** `code/event_value_changes.py` -> `gdelt_plots/event_value_changes/` (6 PNG sheets,
> `correlations_long.csv`, `top_pairs.csv`). Change per year (points) in each event share vs in each survey anchor between waves, six slices: rate 5->6,
> 6->7, 5->7, consecutive rates pooled (pair effects removed), acceleration (rate 6->7 minus 5->6), pooled Spearman. 100-shuffle null as before.
> Two pairs beat chance (family-wise p ~ 0.01): 11 Disapprove x "Secure in neighborhood" (6->7, r = +0.85, n = 29; without Egypt r = +0.72;
> question asked in waves 6 and 7 only) and 02 Appeal x "see myself as member of local community" (5->6, r = -0.88, n = 22). Everything else is near the null.

> **GDELT Explorer artifact (2026-10-23):** https://claude.ai/artifact/VWSWketd1jDskcHTRfuoGm (private). Three views: one country with all 20 event
> categories; one event category compared across countries; one country and category with its sub-events (3-digit or full CAMEO codes, official names from
> `code/cameo_codes.json`). Scales: all years (monthly, rolling window of 1-12 months), one year (daily, rolling 1-30 days), one month (daily). Measures: share of
> the country's events, events per day, share of the category. Source `code/explorer/index.html`; data built by `code/export_gdelt_explorer.py` into
> `/u/gorman2_s/tchakrab/gdelt_data/explorer/` (root_<ISO3>.json 12 MB total, sub_<ISO3>.json 73 MB total, meta.json) and published under `data/` in two batches.
> Tested locally (headless Chrome against a local server); the hosted page itself was not opened.

> **Significance tests + explorer tab (2026-10-23):** `code/event_value_significance.py` reruns both analyses (levels and changes, 6 slices each, 28,120
> correlations) with 2,000 country shuffles: pair-level p, family-wise p, BH q, leave-one-country-out; output `gdelt_plots/event_value_significance/`
> (`all_pairs.csv`, `significant.csv`, `corr.json`). Significant (family-wise p < 0.05): 5 slice-rows = 2 distinct pairs: 11 Disapprove x H001 "secure in
> neighborhood" (counted side = NOT secure; r = +0.85 for 6->7 change, n = 29, +0.72 without Egypt) and 02 Appeal x G020 "member of local community"
> (counted side = disagree; r = -0.89 for 5->6 change, n = 22). Suggestive (BH q < 0.10 only): 39 pairs, all in the wave-7 cross-country slice.
> Added as the fourth tab "Events and survey answers" of the GDELT Explorer (same artifact link; data in `data/corr.json`).
> **Explorer update (2026-10-26):** the suggestive pairs are now a grid of 39 small scatter plots (not a table), and a fifth tab "What the codes mean" lists the 20
> CAMEO categories with one-line summaries (own wording, drawn from the official sub-code names) and every sub-event / detailed code with its official name
> (`code/cameo_codes.json`), with search; category summaries also appear under the charts in the compare and sub-event views. Explorer tabs now: one country /
> compare countries / sub-events / events and survey answers / code meanings. Scope of the correlations: the 20 root categories x 363 of the 905 WVS questions (scale,
> yes/no and select-many only; pick-one, ranked and numeric questions excluded). Not yet done: average-answer (mean) version of the survey side, sub-event-level
> correlations, Step 2 (event-level detail with the 33 columns).

> **Slurm on Gilbreth (2026-10-26):** `slurm/job.sbatch` + `slurm/README.md`. No CPU-only partition exists, and the
> scheduler rejects any job requesting zero GPUs (`Job rejected: No GPUs requested`, undocumented) — every job, even
> pure-CPU ones, must pass `--gres=gpu:1`. Account/partition/QOS/time/cpus/mem/gpu are all sbatch flags at submit time
> (not hardcoded), since the right choice depends on live availability (`slist`, `squeue --qos=standby`). Two lanes:
> `--qos=standby` (free, 4h max, cluster-wide cap of 24 GPUs in use across all users, preemptible, works on any of
> the 3 accounts x any partition since `AllowAccounts=ALL` everywhere — not limited to each account's own GPU type)
> for short/experimental jobs; `--qos=normal` (up to 14 days, charged against that account's allocation) for
> long-running jobs. `job.sbatch` activates conda env `wvs` (`set -u` had to be dropped — conda's own
> activate/deactivate hooks reference unbound vars) and `exec`s the command passed to it. Verified end-to-end: a real
> standby job (csml/a30/gres=gpu:1) activated `wvs`, imported `pandas`, exited 0 in 12s.

> **Planned next experiment: can GDELT predict WVS shifts? (2026-10-26, not started — plan only, drafted on another
> machine)** Two ideas, both pending:
> - **Idea 1, no-LLM signal check.** All 29 countries, waves 5-7, ordered/share-type questions first (thousands of
>   examples). Features: change in the 20 event-category shares between survey years + change in total volume/tone;
>   question section, format, and its usual direction across training countries; the first-wave answer level
>   (regression to the mean — the LLM baseline never saw this, so including it changes the setup vs. the existing
>   53.7% number). Models: "question's usual direction" baseline -> logistic regression -> gradient-boosted trees.
>   Eval: leave-one-country-out, over a grid (GDELT window = survey year vs. whole interval; shares vs. rates vs.
>   volumes; root categories vs. sub-events; question section/format), then a shuffle-country null over the same
>   grid so "best of grid" is judged against chance.
> - **Idea 2, LLM + GDELT summary.** Rerun the same 697 questions (gpt-4.1-mini) adding a short summary of how the
>   country's event mix changed between the two survey years; compare against the 53.7% baseline. Prompt text (must
>   never mention the survey) needs to be shown and approved before any call.
> - **Capturing the timeline** (open question, not decided): two-endpoint change may be too thin. Options: monthly
>   path summaries (trend, volatility, biggest shock + recency, gap from pre-survey baseline); a fixed-length
>   resampled shape + PCA; a small 1D conv sequence model. Should use each country's actual fieldwork months, not
>   just years.
> - **Architecture for the timeline model** (not decided, opinion only, nothing chosen): must not be an LLM/black
>   box, must capture semantics (word embeddings) and time/relations. Options: (1) question-conditioned temporal
>   encoder — question embedding attends over the monthly event series, event codes embedded from official names;
>   (2) low-rank bilinear model, question embedding x GDELT-trajectory embedding — simplest, most interpretable;
>   (3) relational graph over countries/questions/event codes — likely overfits with only 29 countries; (4)
>   latent dynamic/state-space model, hidden value trajectories driven by events, survey waves as noisy readings.
>   Leaning (opinion, not a decision) was (2) first, then (1) if timing turns out to matter.
> - **Why this is a hard sell:** the correlation analysis already run (`event_value_correlations.py` /
>   `event_value_changes.py` / `event_value_significance.py`, see above) is the only part of this actually done, and
>   found the GDELT signal alone looks weak — only 2 of 28,120 correlations survive correction (Disapprove x
>   "secure in neighborhood" 6->7, r=+0.85 n=29, +0.72 without Egypt outlier; Appeal x "member of local community"
>   5->6, r=-0.89 n=22), plus 39 suggestive-only pairs in the wave-7 cross-country slice. Also: GDELT 1.0/2.0 counts
>   and tone aren't comparable across the 2015-02-18 switch, which the wave 5-7 window straddles; the 20 event
>   shares sum to 100% so they move against each other; GDELT measures news coverage, not events themselves.
>
> **Monthly-trajectory correlation, first pass (2026-10-27, done, result: no signal found):** decided against a
> per-question predictive model (median 26, often <20 countries per question once both waves require >=100
> respondents -> p>>n for any multi-feature logistic regression; even 1 feature would make test-set accuracy swing
> ~+-10pp on luck alone) in favor of staying correlational, like the existing yearly-level analysis above, but with
> the monthly path instead of just the two endpoint years. New: `event_value_inputs.event_shares_monthly()`
> (monthly share matrix, factored out of `event_levels()`, output unchanged -- verified against the old
> `event_levels.csv` to 1e-14) and `code/event_value_monthly_changes.py`. For each country x transition (5->6,
> 6->7), the monthly share of each of the 20 CAMEO categories over all months from Jan of the first wave's survey
> year through Dec of the second (24-132 months depending on country) is reduced to 4 fixed-size features: level
> (mean), trend (slope, points/year), volatility (std), shift (last 6mo mean - first 6mo mean) = 80 features,
> correlated (Pearson, across the 29 countries) against each question's change in anchor % (same Y as
> `event_value_changes.py`, for comparability), with a 200-shuffle permutation null (whole country feature vector
> moved to another country) for a family-wise p per slice. Result: nothing beats chance at p<0.05 in either slice
> (29/29 and 29/29 countries covered). Best pairs: 5->6 Appeal-trend x "member of local community" r=-0.83 n=22
> p=0.12; 6->7 Disapprove-trend x "secure in neighborhood" r=+0.82 n=29 p=0.15 -- same pair and sign as the one
> yearly-level correlation above that *did* survive correction, just diluted here by testing 4x more features
> (80 vs 20) against the same permutation budget. Conclusion so far: more granular GDELT features don't surface
> anything the yearly endpoints were missing; the signal (if any) stays weak at n=29. Output:
> `gdelt_plots/event_value_monthly_changes/{correlations_long.csv,top_pairs.csv}`, no plots (first look only).
> Not done yet: Benjamini-Hochberg q-values (only family-wise p so far, unlike `event_value_significance.py`'s
> fuller treatment of the yearly version) and a robustness check (e.g. leave-one-country-out) on the two near-miss
> pairs above.
>
> **Functional-PCA + elastic net (2026-10-27, in progress): supersedes the univariate correlation tests above per
> explicit direction — "forget the earlier univariate correlation tests; this replaces them".** New approach: does
> the SHAPE of a country's monthly GDELT trajectory (not single summary stats) predict the SIZE of a question's
> change, fit per question with regularization instead of tested one feature at a time. New file
> `code/event_value_fpca.py`; `event_value_inputs.py` gained `event_shares_monthly()` (monthly share matrix,
> factored out of `event_levels()`, verified unchanged to 1e-14) and `rescaled_means()` (per question/country/wave,
> mean response rescaled 0-1, so changes are comparable across questions with different native scales -- replaces
> `anchors()`'s share-above-midpoint as the Y source for this line of work; `anchors()` itself is untouched and
> still used by `event_value_correlations.py`/`event_value_changes.py`/`event_value_monthly_changes.py` above).
> - **Fieldwork month** (`fieldwork_months()`): median(S022) per country-wave, not the exact interview date -- `S012`
>   (exact date) only covers 46/87 country-waves, `S022` (interview year+month) covers all 87. Cached in
>   `gdelt_plots/event_value_fpca/fieldwork_months.csv`.
> - **Curve** (`log_centered_curves()`): per country x category x transition, log(share + 0.1) each month from the
>   earlier wave's fieldwork month through the month before the later wave's, minus the mean of log(share + 0.1)
>   over the 24 months before the earlier wave's fieldwork month (baseline), resampled to a fixed 12 points
>   (`np.array_split` into 12 chunks, mean each) so interval length (48-132 months, varies by country) doesn't
>   matter -> `(29 countries, 20 categories, 12 points)` per transition, 29/29 countries covered both transitions.
> - **Functional PCA** (`fpca_per_category()`): per category x transition, PCA across the 29 countries' 12-point
>   curves, first 3 components kept (44-82% variance on component 1 alone, a smooth rising/falling shape in every
>   category checked; components 2-3 are localized early/mid-interval bumps) -> 60 features/country (20x3; a
>   20-feature, component-1-only version also implemented). X-only, fit on all 29 countries (no Y leakage).
> - **Per question** (`fit_question()`): ElasticNetCV (l1_ratio in [0.5, 1.0], 5-fold inner CV -- narrowed from
>   [0.2,0.5,0.8,1.0] for speed, cut per-question fit time 6.4s -> 1.6s) on standardized fPCA scores + the
>   country's earlier-wave mean_rescaled (regression-to-mean control), leave-one-country-out CV refitting the
>   scaler and both models inside each fold (no held-out-country leakage into standardization). Reported: r2_oos
>   (vs. the training-fold mean), r2_ctrl_only (plain OLS on the control alone, no CV grid), baseline =
>   max(0, r2_ctrl_only), gain = r2_oos - baseline, sign accuracy vs. the training-fold majority-sign rate.
>   88 eligible questions (>=20 countries with both waves matched) for 5->6, 113 for 6->7 (fixed a bug in the
>   eligibility filter that was checking ">=1 country per wave independently" instead of the matched-pair count).
> - **Dry run (3 questions, done):** pipeline runs end to end, no crashes; shapes and plots match spec. Sanity
>   check worth noting: A008 ("feeling of happiness") 6->7 has r2_oos=0.618 almost entirely from the control
>   (r2_ctrl_only=0.625, gain=-0.008) -- i.e. regression-to-the-mean explains it, GDELT adds nothing, consistent
>   with the weak-signal conclusion above. G020/H001 are genuinely single-transition questions (not a bug): G020
>   only asked in waves 5-6, H001 only in 6-7.
> - **Steps 5 (permutation null, parallel via `ProcessPoolExecutor`) and 6 (fit 5->6 full-data, freeze the model,
>   predict 6->7 by projecting 6->7 curves onto the 5->6 PCA components, not refit) are built** (`permutation_null()`,
>   `replicate()`, `full_run()`). 2-shuffle local sanity check passed (no crash; shuffled stats were statistically
>   unremarkable vs. observed, e.g. 6->7 observed mean_gain=-0.189 fell between the two null shuffles' -0.153/-0.109).
> - **First full-run submission (job 12032557, csml/a30/standby, 16 cpus, --time=02:30:00): 5->6 completed, 6->7
>   hit the time limit mid-permutation (112/200) and was killed.** 5->6 result (88 questions, 200 perms): **no
>   summary statistic beats chance** -- p=0.81 (n_beat_baseline), p=0.21 (mean_gain), p=0.63
>   (mean sign_acc_minus_majority); none near the 1000-shuffle-extension threshold. Saved:
>   `gdelt_plots/event_value_fpca/{results_5to6.csv,null_stats_5to6.csv,pvalues_5to6.csv}`.
>   5->6 wall time was 4495s for 88 questions x 200 perms over 16 workers -- ~4.1s/question-fit, 2.5x slower than
>   the 1.6s measured for one fit in isolation. Suspected cause: each worker's own BLAS calls (numpy/sklearn)
>   spawning threads on top of the already-parallel 16 worker processes, oversubscribing the allocated CPUs.
>   Fix applied: `threadpoolctl.threadpool_limits(1)` per worker + `OMP/OPENBLAS/MKL_NUM_THREADS=1` env vars
>   (`code/event_value_fpca.py`). A local re-check at n_jobs=8 did not clearly confirm the fix (interactive node
>   is itself shared with other users' jobs, an unreliable testbed) -- not conclusive, abandoned rather than keep
>   burning time on a side diagnostic. `full_run()` also gained a `resume=True` default: a transition whose
>   `results_<label>.csv` already exists is loaded from disk instead of recomputed.
> - **Resubmitted (job 12033617, same account/partition/resources, --time=03:00:00) to resume: 6->7 (full 200
>   perms, since none of its partial 112 were saved) + replication**, budgeted off the worst-case *measured*
>   rate (4.1s/question-fit) rather than the hoped-for fixed rate, since the fix wasn't locally confirmed. Not
>   back yet as of this note.
>
> **LLM mistake analysis + two data-pipeline fixes (2026-10-27).** Looked at where the five-country gpt-4.1-mini
> direction baseline (697 predictions, 53.7% accuracy) goes wrong, using the existing `reasoning_analysis/`
> tfidf/embedding clusters (no new LLM calls). Worst themes: civic/organization **membership questions, 29.3%
> accuracy** -- worse than chance; the model assumes "political upheaval -> people disengage" and is usually
> wrong. Military-coup narratives ~45-49%, also near/below chance. Best: internet/media-use questions, 86.1% --
> a safe global monotonic trend. Common pattern: the model writes one historical-event paragraph (revolution,
> coup, instability) per country-period and reuses it to answer unrelated questions, rather than reasoning about
> the specific question.
>
> - **Demographic splicing (new `code/demographic_splice.py`): does the LLM's prediction secretly align with one
>   demographic subgroup better than the population it was asked about? No.** Reused `change_detection.py`'s
>   existing significance protocol, run on data filtered to sex (X001), age (X003R, coarsened <45/45+ for power),
>   and settlement size (X049, coarsened smaller/larger; skipped for a country-wave with <50% coverage) -- no new
>   LLM calls. Every subgroup's accuracy against the 697 existing predictions (48.8-50.5%, near chance) was
>   *lower* than the population accuracy (53.7%) the model was actually asked about; only 16/697 rows (2.3%) were
>   "wrong vs. population, right vs. one subgroup," spread thin across groups -- not a real pattern. Output:
>   `gdelt_plots/demographic_splice/`.
> - **Two question-format bugs found and fixed while auditing format coverage, both verified to change only
>   labels, not any significance decision or effect size (checked: identical p-values/effects before and after):**
>   - `change_detection.py`'s `detect_country_changes()` was mislabeling every `numeric_scale` question's result
>     row as `format="ordinal_scale"` (a wrong hardcoded string passed into `_test_scale`/`_row`); the actual
>     test behavior was already correct for numeric_scale (same branch as ordinal_scale), so this cost nothing
>     scientifically, but it meant the real format was lost downstream. Fixed: pass the question's own `fmt`
>     through instead of overwriting it.
>   - `direction_baseline.py`'s `FAMILY` dict (maps a question format to its LLM prompt style) was missing a
>     `"numeric_scale"` entry; before the mislabeling fix this didn't matter (those questions were silently
>     riding along disguised as `ordinal_scale`, which has the same entry), but once the mislabeling is fixed
>     this filter would have DROPPED all ~90 numeric_scale questions (confirmed some, e.g. A170 "satisfaction
>     with life", A168A, A173, E033, were already in the existing 697-row results, just mislabeled). Fixed:
>     added `"numeric_scale": "ordered"` (same prompt family as ordinal_scale -- confirmed its prompt builder,
>     `_describe_scale()`, already handles numeric_scale's "only endpoints labeled" shape correctly).
>   - Net effect of both fixes together: the existing 697-row experiment's results are unaffected (same questions,
>     same truth, same effect sizes); future runs now report numeric_scale questions under their correct format.
> - **Not done yet (format audit originally scoped as three steps; did (1) demographics and found/fixed (2) this
>   one concrete bug; have not done (3) a full sweep of the other 7 formats for similar issues):** `categorical`
>   (54 codes), `forced_choice` (22), `ranked_choice` (20), `count` (1), `knowledge` (3) were spot-checked only
>   (E003-E006 "aims of respondent, first/second choice" parsed cleanly) -- not exhaustively audited. Also still
>   true: `event_value_inputs.py`'s `anchors()`/`rescaled_means()` (used by the GDELT-correlation work above, a
>   separate pipeline from direction_baseline.py) still only cover `{ordinal_scale, binary, mention_flag}` =
>   711/905 question codes; the other 194 (now correctly including numeric_scale's 90) are out of scope for that
>   pipeline unless explicitly extended -- not requested yet.
>
> **Dashboard: new "Demographics" tab (2026-10-27), added to the existing WVS artifact (link above).** New
> `code/export_dashboard_demographics.py` (writes `results/direction_baseline/five_country_set/demographics_data.js`,
> `window.DEMOGRAPHICS`) + `code/dashboard/demographics.js` (view), wired into `code/dashboard/index.html` as a
> 5th tab alongside Browse/Compare/LLM baseline/Model reasoning. Shows the 1647 significant cross-sectional
> demographic differences from the demographic-splice work above (sex/age/settlement, within one wave, not across
> waves), filterable by wave/country/demographic/search. Iterated live with the user: fixed an effect-vs-difference
> column confusion (they were accidentally the same number; now `effect` = the test statistic a significance
> decision is based on, `diff` = the groups' plain raw difference -- these coincide for binary-style questions but
> not for scales), added literal group averages (mean code or % per group, not just a delta), added click-to-group
> table headers (Country/Wave/Question/Split, hierarchical, click order = nesting order), fixed wave ordering
> within a group (was sorted by effect strength, not wave), and added a small per-group trend sparkline (two
> lines, solid/dashed, hollow dot = not significant that wave) shown whenever grouping narrows to one
> country+question+demographic with multiple waves of data. The sparkline needed the export script extended to
> compute every tested question's group means/shares regardless of significance (not just the 1647 significant
> ones, since a non-significant wave would otherwise just be a hole in the line) -- `demographics_data.js` now
> also carries a `trend` dict, 7549 points, mean/share only (scale and binary/mention/knowledge formats; skips
> categorical/forced_choice/ranked_choice, whose "tracked option" isn't fixed across waves).

> **New data-acquisition brief (2026-11-04), Phase 0-2 done:** a separate brief ("Data Acquisition Brief WVS Value-Shift
> Project.md", kept outside the repo) scopes collecting news text, citizen text and aggregate indicators for 6 WVS
> countries (US, EG, UA, TH, HK, UY), 2003-2023, as supporting signal -- not yet an analysis plan. `wvs_data/` scaffolded
> (`WVS_DATA_DIR` in `code/config.py`, default `<repo>/data/wvs_data`, same pattern as `GDELT_DIR`/`NYT_ARCHIVE_DIR`;
> `news/`, `citizen/`, `raw/` gitignored as large/regenerable, `meta/` and `indicators/` tracked). `meta/wvs_fieldwork.csv`
> filled in (18 rows, country x wave 5-7) from IHSN/GHDx catalog pages -- **resolves the brief's open question: Uruguay
> wave 7 fieldwork was 2022-01-27 to 2022-03-22, not 2023.** 13/18 rows have exact day-level dates; US W7, UA W5, TH W6,
> TH W7, HK W6 only have year-level precision (worldvaluessurvey.org's own methodology pages gave no per-country dates
> and sometimes hit bot-detection; IHSN catalog pages for these had no stated fieldwork period). Flagged in
> `meta/collection_log.csv`: Egypt wave 6's stated fieldwork period (2012-03 to 2012-04) disagrees by exactly one year
> with its own interview-date variable V261 (2013-02 to 2013-04) -- used the stated period, discrepancy unexplained.
> `meta/gaps.csv` pre-populated with the brief's own known-gaps table (Mada Masr block, HK forum platform switch,
> VKontakte block, consumer-confidence start gaps, Pushshift/X API end, pre-2010 citizen-text thinness). Noted but not
> yet acted on: **GDELT events (all 6 countries, full range) and NYT Archive metadata (US, full range) are already
> pulled from earlier work** (see the two entries above) and satisfy two of the brief's cross-country news routes
> without new collection. Five items drafted for Tania to act on (NYT key's Article Search/Books API scopes, Meta
> Content Library application text, Reddit for Researchers application text, a Purdue-library question re:
> Nexis/ProQuest/Factiva, whether new Telegram collection is wanted) -- not yet sent/submitted by her. Phases 3-5
> (indicators, news collection, citizen text collection) not started.

> **Data-acquisition brief, Phase 3 done (2026-11-04): aggregate indicators, all 6 countries.** New scripts
> `code/wvs_data_pull_wdi.py` and `code/wvs_data_pull_wikipedia_pageviews.py` (both dry-run by default, `--run` to
> execute); the rest of this phase was manual research (one subagent) and transcription, not scripted. Written to
> `data/wvs_data/indicators/{country}/{series_id}.csv`, logged in `meta/{sources,gaps,collection_log}.csv`.
> - **World Bank WDI** (GDP growth, CPI inflation, unemployment -- `gdp_wdi`/`cpi_wdi`/`unemp_wdi`): 18/18
>   country x series pulled clean, 2003-2023, free unauthenticated API. Only gap: Ukraine unemployment has no
>   figure for 2022-2023 (wartime suspension of labor-force surveys, as the brief predicted).
> - **Consumer confidence** (one series per country except Egypt and Hong Kong, which have two each): US
>   (`cci_umich`) pulled directly from FRED's UMCSENT series (252 monthly points, 2003-2023 -- used FRED's
>   redistribution rather than umich.edu directly, since UMich's own usage terms restrict redistribution without
>   written consent). The other five countries needed manual research across press releases/PDFs (a subagent did
>   this, with an explicit no-fabrication rule -- every value has a source URL); results are thin and uneven:
>   Ukraine (`cci_infosapiens`) is the strongest at 63 monthly points 2018-2023 (cross-checked against individual
>   PDF press releases); Egypt Nielsen 9 points (2010-2016), Egypt IDSC only 3 (ahram.org.eg, its main outlet,
>   blocks automated fetches with HTTP 403); Thailand UTCC only 9 scattered points across the full 20 years (no
>   public historical table exists; Trading Economics/CEIC gate full history behind a paid plan); Uruguay UCU only
>   3 points, 2022-2023 (pre-2022 sits behind a discontinued paid Alphacast dataset). **Flagged for a spot-check:**
>   Hong Kong's two series (CUHK 9 points 2010-2018, CityU 3 points 2019-2021) came from Google-indexed snippets of
>   the source PDFs/press releases, not from reopening the documents directly -- both source sites (a CUHK web
>   archive, CityU's JS-rendered press pages) returned bot-check/empty responses to every automated fetch, retried
>   after a rate-limit reset with the same result. One cleanup after the subagent finished: its sources happened to
>   include data past 2023 (Thailand, Uruguay, Ukraine all had rows into 2025-2026); trimmed back to the brief's
>   2003-2023 scope (dropped 4/13, 8/11, 24/87 rows respectively) and fixed `sources.csv` coverage_end to match.
> - **Wikipedia pageviews**: daily country totals from Wikimedia's differentially-private historical release
>   (`analytics.wikimedia.org/published/datasets/country_project_page_historical/`, one file per day, summed
>   `private_count` per country). **Egypt and Thailand never appear in this dataset** (checked multiple sample days
>   across the full range) -- consistent with Wikimedia's privacy "country protection list" (hides ~35 countries),
>   not substituted. US/UA/HK/UY: 2183/2188 days fetched (2017-02-09 to 2023-02-05, the dataset's own full range at
>   country granularity; nothing finer exists after that date, only global totals) -- 5 genuine 404s in Wikimedia's
>   own release, not a pipeline bug. **Sanity check: Ukraine's daily total jumps from ~800K to 3M+ exactly at
>   2022-02-24**, the day of the full-scale invasion -- a real signal, not noise. Needed a custom `User-Agent`
>   header (Wikimedia blocks the default `python-requests` UA per their API etiquette policy); the ~3-7MB daily
>   files themselves are not kept, only the aggregated per-country sums and a resume marker.
> - **Not collected, logged as gaps instead of substituted:** Box Office Mojo (robots.txt disallows automated
>   access for everyone except named search-engine crawlers -- IMDb/Amazon terms, not scraped per the brief's
>   never-do rule). Spotify via kworb.net (its per-country pages are a live today-only snapshot plus a lifetime
>   per-track totals page -- no reconstructible historical daily/weekly archive; the brief itself already called
>   Spotify low-value for most of these windows, so this wasn't pursued further via charts.spotify.com's own
>   archive). Quarterly national GDP from each country's own statistical agency (BEA/CBE/Derzhstat/NESDC/Census and
>   Statistics Dept HK/BCU) -- not started, lower priority than the annual WDI series.
> - **New "needs Tania" item, 6th alongside the five from Phase 2:** Media Cloud's API requires a new account/key
>   (no self-serve-and-anonymous path found) -- per the brief's rule against using a credential she hasn't given,
>   not signed up for.

> **`fast_llm.py` now also calls Claude models (2026-11-03), via the Anthropic SDK (`anthropic==0.125.0`, added to
> `environment.yml`).** `run_llm`/`run_llm_async` pick the provider from the model name (`claude-...` -> Anthropic,
> everything else -> OpenAI as before); same call signature, same on-disk cache, same adaptive-concurrency/retry
> logic either way -- only the network call and response parsing (`client.messages.create`, system prompt split out
> of `messages`, `usage.input_tokens`/`output_tokens`, `stop_reason`) differ per provider. Claude calls require
> `max_tokens` explicitly (raises `ValueError` up front otherwise -- the Anthropic API has no default) and
> `ANTHROPIC_API_KEY` in the environment. Verified offline only, with a mocked Anthropic client standing in for the
> network (dispatch, response parsing, the `max_tokens` guard, and that caching/`plan_llm` behave the same as the
> existing OpenAI path) -- no real Claude call has been made yet; that needs an explicit ask per the project rule.

## Data (`data/`)

- `WVS_Time_Series_1981-2022_csv_v5_0.csv` — full raw World Values Survey time-series microdata
  (waves ~1-7, 1981-2022). One row per respondent. Key columns: `S002VS` (wave number), `COUNTRY_ALPHA`
  (country code), plus hundreds of question columns (e.g. `A027`, `E179`) and demographic columns
  (`X001` sex, `X002` age, `X007` marital status, `X011` num children, `X025R` education, `X026`
  lives with parents, `X028` employment, `X036` occupation, `X040` union member, `X044` savings,
  `X045` subjective class, `X047R_WVS` income bracket, `X049` town size). Negative values are WVS
  missing/non-response codes (e.g. -1 don't know, -2 no answer, -4 not asked, -5 missing).
- `WVS_Time_Series_Vars.csv` — codebook: variable names, labels, and raw answer-choice text for the
  above.
- `averages_cached.csv` — precomputed cache of average answer value per (wave, country, question),
  produced by `WVSDataLoader.calculate_averages()`.
- `parsed_survey_data.jsonl` / `parsed_survey_data_cleaned.jsonl` — per-question metadata
  (`parsed_question` text + `choices` list) rewritten from the raw codebook into a cleaner
  natural-language form via an LLM pass (see `code/prompts/system/codebook_parsing/create_survey.txt`). This is what
  the prediction pipelines use to build prompts. `_cleaned` is a later-pass version; not yet clear
  what changed between the two or which is canonical.
- `significant_changes/` — precomputed "ground truth" of what changed between WVS waves, split by
  three statistical tests (`welch/`, `ks/`, `wasserstein/` subfolders). Each subfolder has one CSV
  per (country-group letter A-G, wave-pair) combo, e.g. `welch/C_wave5_to_wave7.csv`, covering
  wave pairs 5→6, 6→7, 5→7. Produced by `WVSDataLoader.get_significant_changes()`. The three tests
  have not yet been checked against each other for agreement.
- `gdelt/` — early, unfinished attempt to bring in real-world news events (GDELT) as a signal.
  Contains one BigQuery export (`bq-results-...csv`: 349,961 raw Egypt events, 2008-2009, 8 columns) and one
  test pull scoped to Egypt (`egypt_test_results.csv`: monthly counts per CAMEO root code, 2008-2018).
  Also `ledger.jsonl` and `cache/` (written by `code/gdelt_bq.py`, gitignored, see the GDELT section below).
  Not wired into any pipeline yet.

## Code (`code/`)

### Data loading / significance testing
- `WVS_dataloader.py` and `dataloader.py` — two versions of essentially the same `WVSDataLoader`
  class (load WVS CSV, cache per-wave/country/question averages, compute significant changes
  between waves). `WVS_dataloader.py` is the newer/more complete one (adds Wasserstein distance,
  save/load of results to `significant_changes/`, per-country and all-country significance lookup
  helpers). `dataloader.py` only supports Welch's t-test and is used by the older
  `run_survey_on_llm.py` baseline runner. These should probably be consolidated at some point.
- Both compute: which countries/questions are common across two waves, per-country-per-question
  averages, and significance of the change in average between two waves (Welch t-test, KS test,
  Wasserstein distance).

### LLM pipelines (all use OpenAI `gpt-4.1-mini` via `thread_gpt_suite`, a local threaded/multiprocess
API-call handler in `code/thread_gpt_suite/`)

1. **Direct baseline shift prediction** — `run_survey_on_llm.py` (`WVSLLMBaselineRunner`).
   Given a survey question + country + two waves, prompts the LLM (system prompt
   `prompts/system/llm_baseline_predict_value_changes.txt`) to reason about relevant real-world
   events and predict, in one sentence, how the country's average answer shifted between the two
   waves. No persona, no explicit event feed — relies entirely on the model's own background
   knowledge. Has `compute_accuracy()` for scoring against actual per-question mean shifts.
   Run so far for: USA and IND, wave 6 (see `results/accuracy_results_{USA,IND}_6.json` and
   `results/llm_baseline_predict_value_changes_*_{USA,IND}_6.json`).

2. **Persona-based individual prediction** — `persona_based_prediction_llm_runner.py`
   (`PersonaBasedPredictionLLMRunner`). For a given country + wave, builds one prompt per real
   respondent using their actual demographic profile (age, sex, marital status, education, class,
   income, town size, etc.), batches 10 survey questions per prompt, and asks the LLM
   (system prompt `prompts/system/persona_based_prediction.txt`) to answer as that person. Then
   parses the LLM's per-question answers and compares them to the respondent's real recorded
   answer. Deduplicates identical demographic+question-batch prompts to cut down on LLM calls.
   This is the most heavily run pipeline: raw + parsed results exist for USA, EGY, TUR, UKR across
   waves 5, 6, 7 (`results/results_new/persona_based_prediction_{raw,parsed}_{country}_{wave}.json`).
   No aggregate accuracy summary has been computed from these yet (unlike the baseline pipeline,
   which has `accuracy_results_*.json`).

3. **Chain-of-thought prediction** (uncommitted, in progress) — `prompts/system/CoT_prediction.txt`
   and `prompts/user/CoT_prediction.txt`, exercised in `code/CoT_prompt_prediction.ipynb`. Not yet
   clear how this differs in purpose from the direct baseline above (possibly a more structured
   reasoning-chain variant) — needs follow-up.

### Prompt construction helper
- `prompts/{system,user}/codebook_parsing/create_survey.txt` — one-off LLM prompt used to turn raw WVS codebook question
  text + choices into cleaner natural-language questions (feeding `parsed_survey_data*.jsonl`).

### Notebooks (exploratory, not scripts)
- `analysis.ipynb`, `clean_up_questions.ipynb`, `embedding_tests.ipynb`, `parse_codebok.ipynb`,
  `starter.ipynb`, `quick_test.ipynb`, `see_changes_for_countries.ipynb`,
  `run_llm_baseline.ipynb`, `run_llm_basline_changes.ipynb`, `temp_post_process_WVS.ipynb`,
  `persona_based_prediction.ipynb` (+ an uncommitted `persona_based_prediction copy.ipynb`),
  `CoT_prompt_prediction.ipynb`. Contents not yet individually audited — likely where a lot of the
  actual exploratory analysis and one-off scripts live, worth going through before drawing
  conclusions from `results/`.

### Uncommitted local changes (as of this snapshot)
- `WVS_dataloader.py`: adds Wasserstein distance test, per-country significance lookup, and
  save/load of `significant_changes/` results (not yet in the git-tracked version's history).
- `persona_based_prediction_llm_runner.py`: filters out negative (missing/non-response) answer
  choices before showing them to the LLM as options.
- `prompts/system/persona_based_prediction.txt`: updated from single-question to 10-question
  batched format to match the batched prompt construction in the runner.

## Results (`results/`)

- `accuracy_results_{USA,IND}_6.json` — overall/per-question/per-person accuracy from the direct
  baseline pipeline.
- `llm_baseline_predict_value_changes_{results,parsed}_{USA,IND}_6.json` — raw LLM outputs and
  parsed predictions from the direct baseline pipeline.
- `results_new/persona_based_prediction_{raw,parsed}_{EGY,TUR,UKR,USA}_{5,6,7}.json` — raw and
  parsed persona-simulation results (plus one duplicate `UKR_6_` run). No accuracy has been
  computed from these yet.
- `plot.png` — untracked, not yet identified what it plots.

## Findings (2026-09-23)

- **Correction:** letters A-G in `significant_changes/*/` are WVS *question-prefix* groups (A-questions, B-questions, ...),
  not country groups. Each file covers all countries. `change_magnitude` is a signed mean difference, and is blown up
  by `F025_WVS` (values ~1e7) — exclude that question before using it.
- **Significance tests:** at p<.05, Welch flags 72% and KS 68% of country-question-pairs as "significant" (too permissive
  to be useful as ground truth). Welch/KS agree on 84%; ranking correlation |t| vs KS = 0.84, KS vs Wasserstein = 0.84.
  Top-5% overlap between Welch and Wasserstein is only ~35%. Wasserstein has no p-value, so the loader treats every row as significant.
- **Persona accuracy (USA/EGY/TUR/UKR x waves 5-7, `UKR_6_` duplicate excluded):** exact-match 57% overall vs 69% for a
  per-question modal-answer baseline (in-sample). Predicted shifts between waves do NOT track actual shifts
  (mean Pearson ~0.05, sign agreement ~54%). Multi-select question (A027 batch) is never answered by the LLM (100% empty).

## Question categories and change-detection protocol (decided 2026-09-24)

- `code/categorize_questions.py` -> `data/question_categories.csv`: all 905 questions labeled by answer format
  (ordinal 474, mention_flag 152, numeric 90, binary 85, categorical 54, forced_choice 22, ranked_choice 20,
  identifier 4, knowledge 3, count 1), hand-swept until clean. Mention flags carry a `group` (battery).
- A change is "significant" only if it is reliable (test, FDR-corrected across questions) AND larger than a
  minimum size tau. Effect size per format (chosen by user):
  - ordinal / numeric: Wasserstein distance on answers rescaled to 0-1 (plus signed mean shift for direction)
  - binary / knowledge: percentage-point change in share
  - categorical / forced choice: BOTH total shift in shares (permutation test) AND Cramer's V (chi-square)
  - ranked choice: first+second choice combined, total shift in shares
  - mention flags: per-item pp change plus battery-level average absolute change
- Implemented in `code/change_detection.py` (`ChangeConfig` holds all thresholds; loader method
  `WVSDataLoader.detect_significant_changes(wave1, wave2)`). Starting thresholds are moderate guesses, not yet
  calibrated; plan is to calibrate with a split-half noise floor. Test run USA waves 5->6: 145 tests, 8% of ordinal
  questions significant (vs ~70% with plain p<.05). The old file-based filters in the loader are unchanged/superseded.

## Open threads / things worth checking before picking a direction

- Persona-based results: an aggregate accuracy was computed (see Findings 2026-09-23); those results are archived and untrusted.
- The three significance tests (Welch, KS, Wasserstein) were compared (Findings 2026-09-23); the project now uses the change-detection protocol instead.
- GDELT: daily statistics for 29 countries (2003-11 to 2023-07) are pulled and explorable, and event shares were correlated with WVS answers (two pairs
  survive correction). Step 2 (row-level event detail, same range/countries) is now also pulled and verified, 497.6M rows, ~26 GB.
  Still open: whether to feed events to the LLM prompts, a mean-answer version of the survey side, and any use of the row-level detail
  (actor names/types, source URLs) beyond what the aggregated daily stats already cover.
- NYT Archive API: article metadata (no full text) pulled for the same full range, 237 months, not yet filtered or used for anything.
- LLM direction baseline: gpt-4.1-mini, gpt-5-mini and gpt-5 (reasoning on/off) are done in both option orders; Claude and Qwen are not run (need an
  ANTHROPIC_API_KEY, Claude support in `fast_llm.py`, and a decision on how to run Qwen).
- `dataloader.py` vs `WVS_dataloader.py`, and `parsed_survey_data.jsonl` vs `parsed_survey_data_cleaned.jsonl`, are duplicate/divergent — should be reconciled.
- `.gitignore` and the GDELT wrapper files (`code/gdelt_bq.py`, `code/test_gdelt_bq.py`) are untracked in main; the wrapper also lives on branch `worktree-gdelt-wrapper`.
- `environment.yml` (for a `wvs-shifts` env) was never built and is unused; work runs in the existing conda env `wvs`.
