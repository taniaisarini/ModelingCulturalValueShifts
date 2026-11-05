"""Pull daily Wikipedia country-level pageview totals for the data-acquisition-brief countries,
from Wikimedia's differentially-private historical release (one file per day, country x project x
page, 2017-02-09 to 2023-02-05 -- this is the dataset's own full range; nothing later exists at
country granularity).

    python code/wvs_data_pull_wikipedia_pageviews.py          # dry run: show the plan
    python code/wvs_data_pull_wikipedia_pageviews.py --run    # actually download and aggregate

Source: https://analytics.wikimedia.org/published/datasets/country_project_page_historical/
(one <date>.tsv per day, ~3-7MB, columns: country_name, country_code, project, page_id, page_title,
item_id, private_count -- no header row). Egypt and Thailand never appear in this dataset (checked
2017-02-09 and 2019-06-15; consistent with Wikimedia's privacy "country protection list" that hides
~35 countries) -- only US, UA, HK, UY are pulled; EG/TH are logged as a gap, not substituted.

For each day, sums private_count per target country across all released pages (an undercount of true
total traffic, since only pages above a release threshold are included) and appends one row to
<WVS_DATA_DIR>/indicators/{country}/wiki_pageviews.csv (date,value,unit,frequency,source_url). The
~3-7MB daily files themselves are not kept (public aggregate counts, not licensed text) -- only a
resume marker under <WVS_DATA_DIR>/_cache/wiki_pageviews/_pulled/<date>.done.

Pacing: no documented rate limit for this static-file host; defaults to 1 request/second per the
brief's fallback rule.
"""
import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import WVS_DATA_DIR  # noqa: E402

D = dt.date
BASE_URL = "https://analytics.wikimedia.org/published/datasets/country_project_page_historical/{date}.tsv"
# Wikimedia blocks the default python-requests User-Agent; their API etiquette policy asks for one
# that identifies the tool (see https://meta.wikimedia.org/wiki/User-Agent_policy).
USER_AGENT = ("ModelingCulturalValueShifts-research/1.0 "
              "(https://github.com/taniaisarini/ModelingCulturalValueShifts)")
RANGE = (D(2017, 2, 9), D(2023, 2, 5))  # dataset's full available range at country granularity
TARGET_COUNTRIES = {"US", "UA", "HK", "UY"}  # EG, TH never appear in this dataset -- see docstring


def day_list(start, end):
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur = cur + dt.timedelta(days=1)
    return out


def marker_path(outdir, day):
    return outdir / "_cache" / "wiki_pageviews" / "_pulled" / f"{day.isoformat()}.done"


def out_csv_path(outdir, country):
    p = outdir / "indicators" / country / "wiki_pageviews.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_header(path):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "value", "unit", "frequency", "source_url"])


def fetch_day_totals(session, day, max_retries=5):
    url = BASE_URL.format(date=day.isoformat())
    for attempt in range(max_retries):
        resp = session.get(url, timeout=120)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt * 5))
            print(f"  429 rate-limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"{day}: exceeded {max_retries} retries on 429")

    totals = {c: 0 for c in TARGET_COUNTRIES}
    for line in resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        code = parts[1]
        if code in TARGET_COUNTRIES:
            try:
                totals[code] += int(parts[6])
            except ValueError:
                continue
    return totals, url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=D.fromisoformat, help="range start (default: 2017-02-09, dataset start)")
    ap.add_argument("--end", type=D.fromisoformat, help="range end (default: 2023-02-05, dataset end)")
    ap.add_argument("--run", action="store_true", help="actually download (default: dry run / plan only)")
    ap.add_argument("--no-resume", action="store_true", help="re-pull days that already have a _pulled marker")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1.0/sec)")
    a = ap.parse_args()

    start, end = a.start or RANGE[0], a.end or RANGE[1]
    days = day_list(start, end)
    outdir = WVS_DATA_DIR
    plan = [d for d in days if a.no_resume or not marker_path(outdir, d).exists()]
    skipped = len(days) - len(plan)

    print(f"range {start} .. {end}: {len(days)} day(s), {skipped} already pulled (skipping), "
          f"{len(plan)} to fetch, countries {sorted(TARGET_COUNTRIES)}\n")
    est_minutes = len(plan) * a.delay / 60
    print(f"at {a.delay}s/request: ~{est_minutes:.1f} minutes minimum (plus download time)")
    if not a.run:
        print("\ndry run only. Add --run to execute.")
        return

    for c in TARGET_COUNTRIES:
        ensure_header(out_csv_path(outdir, c))
    marker_path(outdir, start).parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    done, errors = 0, []
    for day in plan:
        try:
            totals, url = fetch_day_totals(session, day)
        except Exception as exc:
            print(f"{day}: FAILED -- {exc}", flush=True)
            errors.append((str(day), str(exc)))
            time.sleep(a.delay)
            continue
        for c, total in totals.items():
            with open(out_csv_path(outdir, c), "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([day.isoformat(), total, "pageviews (differentially private, "
                                         "summed over released pages)", "daily", url])
        marker_path(outdir, day).touch()
        done += 1
        if done % 100 == 0 or day == plan[-1]:
            print(f"{day}: done {done}/{len(plan)}", flush=True)
        time.sleep(a.delay)

    print(f"\n{done}/{len(plan)} days fetched, {len(errors)} errors")
    if errors:
        print("failed days (retry with a future run):", errors[:10],
              "... and more" if len(errors) > 10 else "")


if __name__ == "__main__":
    main()
