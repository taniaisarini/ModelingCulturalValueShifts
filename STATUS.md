# Status

Inventory of what exists in this repo; the dated notes below are the running log (last updated 2026-10-26). No research question is
settled yet — this is a snapshot to work from, not a plan. Update this doc as code/data/results change.

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
  survive correction). Still open: whether to feed events to the LLM prompts, a mean-answer version of the survey side, Step 2 event-level detail.
- LLM direction baseline: gpt-4.1-mini, gpt-5-mini and gpt-5 (reasoning on/off) are done in both option orders; Claude and Qwen are not run (need an
  ANTHROPIC_API_KEY, Claude support in `fast_llm.py`, and a decision on how to run Qwen).
- `dataloader.py` vs `WVS_dataloader.py`, and `parsed_survey_data.jsonl` vs `parsed_survey_data_cleaned.jsonl`, are duplicate/divergent — should be reconciled.
- `.gitignore` and the GDELT wrapper files (`code/gdelt_bq.py`, `code/test_gdelt_bq.py`) are untracked in main; the wrapper also lives on branch `worktree-gdelt-wrapper`.
- `environment.yml` (for a `wvs-shifts` env) was never built and is unused; work runs in the existing conda env `wvs`.
