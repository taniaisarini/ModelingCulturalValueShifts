# GDELT on BigQuery: how to query it without being charged

Notes from the session (2026-10-19) that researched this and built `code/gdelt_bq.py`. Read this before running any GDELT query.
What the code does is in `STATUS.md` and the docstring of `code/gdelt_bq.py`; this file is the *why*. Items marked **unverified** were not checked.

## Why GDELT
Real-world events as a signal for why country answers shift between WVS waves. Typical query: one country, a few months.

## How cost works (what we found)
- BigQuery bills **bytes scanned**, not rows returned. The free tier (sandbox) is **1 TB of queries per month**, 10 GB storage;
  sandbox tables expire after 60 days (so save results as CSV/parquet, not BigQuery tables).
- **Use the `_partitioned` tables and filter `_PARTITIONTIME`** (GDELT's own test: a GKG query went from 423 GB to 15 GB).
  Partitions are by **day only**: a country filter (`ActionGeo_CountryCode`) does *not* reduce bytes; the **date range and the number of
  columns named** do. `SELECT *` and `LIMIT` do not help. A **dry run is free** and reports bytes before anything is billed.
- `_PARTITIONTIME` is ingestion time and can lag `SQLDATE`; filter both (the wrapper pads the partition range by 3 days).
- Sources: https://blog.gdeltproject.org/announcing-partitioned-gdelt-bigquery-tables/ , https://cloud.google.com/bigquery/docs/sandbox

## Tables (checked with free metadata calls and dry runs)
| | table | size | covers |
|---|---|---|---|
| GDELT 1.0 | `gdelt-bq.full.events_partitioned` | 251 GB, day-partitioned | before 2015-02-18 (partitions seen to at least 2019, none in Jan 2022) |
| GDELT 2.0 | `gdelt-bq.gdeltv2.events_partitioned` | 398 GB, day-partitioned | 2015-02-18 on (2011 returned ~0 GB) |
The unpartitioned `full.events`, `gdeltv2.events` and the `eventmentions` tables also exist: do not use them. 1.0 and 2.0 counts and
tone are **not comparable**. Rough cost for Egypt, one quarter, few columns: 0.03-0.25 GB (1.0), 0.7-1.4 GB (2.0).

## Layers that stop accidental charges (strongest first)
1. **A project with no billing account** (BigQuery sandbox): over-quota queries fail instead of charging. Only guarantee that does not
   depend on our code. The project `gdelt-sandbox-512120` got a billing account linked automatically (the owner had other billed
   projects) and was told to disable it; **unverified that this was ever done**. Check the Billing page.
2. A custom daily "Query usage" quota in the Cloud console (e.g. 50 GB/day). **Unverified that one is set.**
3. In `gdelt_bq.py`: `maximum_bytes_billed` on every job (BigQuery rejects, no charge), mandatory dry run, 20 GB per-query cap,
   800 GB monthly local ledger (only counts queries made through this wrapper), result cache, no raw SQL, dry run unless `--run`.

## Setting it up on a new machine
1. Cloud console -> new project, **do not attach billing**; note the project ID (`--project`).
2. IAM & Admin -> Service Accounts -> create one with role **BigQuery Job User** -> add a JSON key. Keep the key **outside the repo**
   (a copy in the repo root is gitignored, but do not rely on that).
3. `pip install google-cloud-bigquery` in env `wvs`; `export GOOGLE_APPLICATION_CREDENTIALS=<key path>`.
4. `python code/gdelt_bq.py --project <id> --country EGY --start 2016-01-01 --end 2016-03-31` (dry run), then add `--run`.

## Caveats
- The first wrapper (`git show 425baf3:code/gdelt_wrapper.py`) hit the unpartitioned table with `SELECT *` and no date filter, and
  an old service-account key is in git history (`code/archived/deleted_wvs-project-...json`): **rotate it** if it is still active.
- `data/gdelt/egypt_test_results.csv` (made before the wrapper, query not saved) has far smaller counts than an unfiltered pull
  (Jan 2008 code 02: 7 vs 841), probably a filtered subset. Do not mix them.
- Countries are FIPS in GDELT (EG), ISO3 in WVS (EGY); `ISO3_TO_FIPS` in `gdelt_bq.py`.
