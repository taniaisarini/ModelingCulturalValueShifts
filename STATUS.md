# Status

Inventory of what exists in this repo; the dated notes below are the running log (last updated 2026-10-29). No research question is
settled yet — this is a snapshot to work from, not a plan. Update this doc as code/data/results change.

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
