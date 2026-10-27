# ModelingCulturalValueShifts

This is an exploratory NLP research project involving the World Values Survey (WVS). There is no fixed research question yet — the current phase is running exploratory experiments with existing code/data to find one.

See [STATUS.md](STATUS.md) for a running log of what code, data, and results exist (dated notes in the blockquotes near the top in chronological order, newest just above the "Data" heading; older inventory sections below). Keep that file updated as work progresses.

## Working rules (from the user; follow them)

- **Concise replies.** Short answers, one point at a time, no walls of text, no long option menus unless asked. Ask at most one short question at the end.
- **Never make design decisions or assume what the user wants.** Offer options and costs, give an opinion only when marked as one, and let the user choose. Open-ended requests ("more detail", "try several ways") still need the scope stated back.
- **Prompts need approval first.** Never write, change or use an LLM prompt until the user has seen the exact text (system and user, with a filled-in example) and confirmed it.
- **Never tell any model that the data comes from the World Values Survey / WVS** (no "WVS", survey name, wave or variable codes like `E069`, or verbatim codebook wording in any prompt). The goal is reasoning about how values change, not recall. Code, docs and file names may say WVS.
- **Do not run experiments unasked.** "Write the code" means write it and test it offline (build prompts, mock clients, dry runs), then stop. Anything that calls a paid API or BigQuery for real needs an explicit ask.
- **All LLM calls go through `code/fast_llm.py`** (`run_llm`). Never use the archived `archive/llm_calling_v1/` code or write a new calling loop. Calls are cached in `llm_cache/` (never send the same call twice; `plan_llm` shows hits and misses without calling).
- **Dashboards and artifacts are private** and contain full prompts and model replies. Keep them private.
- Commit or push only when asked. Update `STATUS.md` after meaningful work.

## Set up on a new machine

1. `git clone git@github.com:taniaisarini/ModelingCulturalValueShifts.git`, then `conda env create -f environment.yml` and `conda activate wvs` (python 3.13; the file has not been tested on a fresh machine, so fix it if the solve fails).
2. Copy the raw survey file by hand (not in git, 1.4 GB) to `data/WVS_Time_Series_1981-2022_csv_v5_0.csv`. Everything else the code reads from `data/` is in git.
3. Set machine-specific folders: copy `code/config.example.toml` to `code/config.local.toml` (gitignored) and set `gdelt_dir` (GDELT cache, pulled data and explorer files; defaults to `data/gdelt`) and optionally `plots_dir` (defaults to `gdelt_plots/`). Environment variables `GDELT_DIR` and `PLOTS_DIR` override the file. The GDELT daily file (`pulls/events_daily_2003-11-01_2023-07-31.parquet`, 29 countries) lives in that folder and is not in git; copy it from the old machine if needed.
4. API keys come from the environment, never from files: `export OPENAI_API_KEY=...`. Google credentials (`GOOGLE_APPLICATION_CREDENTIALS`, a service-account key file) are only needed to run new GDELT BigQuery queries; never commit the key (it is gitignored).
5. Check: `python code/test_gdelt_bq.py` (offline), and `python code/run_direction_baseline_set.py` (dry run: builds prompts and shows cache hits, sends nothing).

## Where things are

- **Survey data and change detection:** `change_detection.py` (what counts as a significant change between two waves; `ChangeConfig` holds thresholds), `categorize_questions.py` (answer format of each question, `data/question_categories.csv`), `WVS_dataloader.py` (older loader), `data/significant_changes/`.
- **LLM direction baseline** (predict which way a country's answers moved between two waves): `direction_baseline.py` (prompts are in `code/prompts/{system,user}/direction_baseline/`; `flip=True` reverses answer order), `run_direction_baseline_set.py` (five countries, `--model`, `--reasoning off|on`, `--flip`, `--run`; dry run by default), results in `results/direction_baseline/`. Models run so far: gpt-4.1-mini, gpt-5-mini and gpt-5 with reasoning off and on, both option orders. Claude and Qwen are not run.
- **Analysis of the model replies:** `reasoning_tfidf.py`, `reasoning_embed.py`, `reasoning_common.py`; flipped-order consistency and the LLM judge: `reasoning_flip_consistency.py`, `reasoning_flip_judge.py`, prompt in `code/prompts/user/flip_judge/`.
- **GDELT:** read [GDELT_BIGQUERY.md](GDELT_BIGQUERY.md) first (how BigQuery cost works, which tables, how to avoid being charged, setup). `gdelt_bq.py` (cost-guarded BigQuery access: no raw SQL, per-query and monthly byte caps, local ledger, cache; dry run by default), `gdelt_pull.py` and `gdelt_sweep.py` (daily statistics per day, country and CAMEO code for the 29 countries in WVS waves 5-7, 2003-11 to 2023-07), `gdelt_plot_monthly.py`, `export_gdelt_explorer.py` and `code/explorer/index.html` (interactive explorer artifact), `cameo_codes.json` (official CAMEO names). GDELT 1.0 (before 2015-02-18) and 2.0 (after) are not comparable in counts or tone.
- **Event shares vs survey answers:** `event_value_inputs.py` -> `event_value_correlations.py` (levels), `event_value_changes.py` (changes over time), `event_value_significance.py` (permutation tests). Output in `gdelt_plots/`.
- **Dashboards (Claude artifacts):** `export_dashboard_baseline.py`, `export_dashboard_reasoning.py`, sources in `code/dashboard/`. Links are in `STATUS.md`.
- **Archived:** `archive/llm_prediction_v1/` (first-round LLM results, not trusted) and `archive/llm_calling_v1/` (old calling code, do not use).

## Conventions

- Paths inside the repo are found from file locations; only data outside the repo goes through `code/config.py`. Do not hard-code absolute paths.
- Expensive or paid steps default to a dry run and take `--run`. Cached calls cost nothing to repeat.
- Scripts have a docstring at the top saying what they read, write and how to run them.
