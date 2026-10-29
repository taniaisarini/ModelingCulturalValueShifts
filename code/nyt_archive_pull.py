"""Pull NYT Archive API data (article metadata, not full text) for the full WVS wave 5-7 range by default,
one calendar month per call, and cache the raw JSON locally.

    python code/nyt_archive_pull.py                 # dry run: call plan, nothing sent
    python code/nyt_archive_pull.py --run            # run it

Needs NYT_API_KEY in the environment (never in a file; sign up at developer.nytimes.com). Endpoint:
https://api.nytimes.com/svc/archive/v1/{year}/{month}.json -- one call returns EVERY NYT article for that
month (headline, abstract, snippet, keywords, section, byline, word_count, etc. -- confirmed via NYT's own
OpenAPI spec that there is no full article body text field). There is no per-country filter on this endpoint;
it returns all NYT content for the month, undifferentiated -- filtering to what's relevant to a given WVS
country happens locally, after pulling (not implemented here yet -- this script only pulls and caches).

Output: one raw JSON file per month under NYT_ARCHIVE_DIR (see config.py: $NYT_ARCHIVE_DIR,
code/config.local.toml, or <repo>/data/nyt_archive):
    <NYT_ARCHIVE_DIR>/raw/<year>-<month>.json
and a resume marker under <NYT_ARCHIVE_DIR>/_pulled/<year>-<month>.done (a month already pulled is skipped
unless --no-resume).

Rate limiting: the exact current limit is NOT confirmed (NYT's developer portal was not reachable to verify;
third-party sources disagree, from 5/min+500/day to 10/min+4000/day) -- check developer.nytimes.com after
signing up and pass the real number via --calls-per-minute. Defaults to the most conservative figure found
(5/min) until that is confirmed. Retries on HTTP 429 with exponential backoff, honoring a Retry-After header
if present.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NYT_ARCHIVE_DIR  # noqa: E402

D = dt.date
BASE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
WVS_RANGE = (D(2003, 11, 1), D(2023, 7, 31))  # one year before wave 5 fieldwork start, through wave 7 fieldwork end


def month_list(start, end):
    out, cur = [], D(start.year, start.month, 1)
    while cur <= end:
        out.append((cur.year, cur.month))
        cur = D(cur.year + 1, 1, 1) if cur.month == 12 else D(cur.year, cur.month + 1, 1)
    return out


def marker_path(outdir, year, month):
    return outdir / "_pulled" / f"{year}-{month:02d}.done"


def raw_path(outdir, year, month):
    return outdir / "raw" / f"{year}-{month:02d}.json"


def fetch_month(session, api_key, year, month, max_retries=5):
    url = BASE_URL.format(year=year, month=month)
    for attempt in range(max_retries):
        resp = session.get(url, params={"api-key": api_key}, timeout=60)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
            print(f"  429 rate-limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"{year}-{month:02d}: exceeded {max_retries} retries on 429")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=D.fromisoformat, help="range start (default: full WVS range, 2003-11-01)")
    ap.add_argument("--end", type=D.fromisoformat, help="range end (default: full WVS range, 2023-07-31)")
    ap.add_argument("--run", action="store_true", help="actually call the API (default: dry run / plan only)")
    ap.add_argument("--no-resume", action="store_true", help="re-pull months that already have a _pulled marker")
    ap.add_argument("--calls-per-minute", type=float, default=5.0,
                     help="pacing; default 5/min is the most conservative figure found -- confirm the real "
                          "limit at developer.nytimes.com and pass it here")
    a = ap.parse_args()

    start, end = a.start or WVS_RANGE[0], a.end or WVS_RANGE[1]
    months = month_list(start, end)
    outdir = NYT_ARCHIVE_DIR
    plan = [(y, m) for y, m in months if a.no_resume or not marker_path(outdir, y, m).exists()]
    skipped = len(months) - len(plan)

    print(f"range {start} .. {end}: {len(months)} month(s), {skipped} already pulled (skipping), "
          f"{len(plan)} to fetch\n")
    for y, m in plan[:5]:
        print(f"  {y}-{m:02d}")
    if len(plan) > 5:
        print(f"  ... and {len(plan) - 5} more")
    est_minutes = len(plan) / a.calls_per_minute
    print(f"\nat {a.calls_per_minute}/min: ~{est_minutes:.1f} minutes to complete")
    if not a.run:
        print("\ndry run only. Add --run to execute.")
        return

    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        raise RuntimeError("NYT_API_KEY not set in the environment")

    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    (outdir / "_pulled").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    delay = 60.0 / a.calls_per_minute
    done, errors = 0, []
    for y, m in plan:
        try:
            data = fetch_month(session, api_key, y, m)
        except Exception as exc:
            print(f"{y}-{m:02d}: FAILED -- {exc}", flush=True)
            errors.append((y, m, str(exc)))
            time.sleep(delay)
            continue
        n_docs = len(data.get("response", {}).get("docs", []))
        with open(raw_path(outdir, y, m), "w", encoding="utf-8") as f:
            json.dump(data, f)
        marker_path(outdir, y, m).touch()
        done += 1
        print(f"{y}-{m:02d}: {n_docs:,} articles written", flush=True)
        time.sleep(delay)

    print(f"\n{done}/{len(plan)} months fetched, {len(errors)} errors")
    if errors:
        print("failed months (retry with a future run):", errors)


if __name__ == "__main__":
    main()
