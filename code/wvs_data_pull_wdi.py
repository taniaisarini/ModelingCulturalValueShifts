"""Pull World Bank WDI indicators (GDP growth, CPI inflation, unemployment) for the 6
data-acquisition-brief countries, full range 2003-2023. Free public API, no key needed.

    python code/wvs_data_pull_wdi.py          # dry run: show the plan
    python code/wvs_data_pull_wdi.py --run    # actually fetch and write CSVs

Endpoint: https://api.worldbank.org/v2/country/{ISO3}/indicator/{CODE}?format=json&date=2003:2023
Writes one CSV per country x series to <WVS_DATA_DIR>/indicators/{country}/{series_id}.csv
(columns: date,value,unit,frequency,source_url), per the brief's output spec. Annual frequency,
World Bank's own units (growth rate %, etc.) -- not rescaled.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import WVS_DATA_DIR  # noqa: E402

COUNTRIES = {"US": "USA", "EG": "EGY", "UA": "UKR", "TH": "THA", "HK": "HKG", "UY": "URY"}
SERIES = {
    "gdp_wdi": ("NY.GDP.MKTP.KD.ZG", "% annual growth"),
    "cpi_wdi": ("FP.CPI.TOTL.ZG", "% annual inflation"),
    "unemp_wdi": ("SL.UEM.TOTL.ZS", "% of total labor force"),
}
START, END = 2003, 2023
BASE_URL = "https://api.worldbank.org/v2/country/{iso3}/indicator/{code}"


def fetch_series(session, iso3, code):
    url = BASE_URL.format(iso3=iso3, code=code)
    params = {"format": "json", "date": f"{START}:{END}", "per_page": 1000}
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"unexpected response shape for {iso3}/{code}: {payload}")
    return payload[1] or []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="actually call the API (default: dry run / plan only)")
    a = ap.parse_args()

    plan = [(country, iso3, series_id, code, unit)
            for country, iso3 in COUNTRIES.items()
            for series_id, (code, unit) in SERIES.items()]
    print(f"{len(plan)} country x series pulls planned ({len(COUNTRIES)} countries x {len(SERIES)} series), "
          f"{START}-{END}\n")
    for country, iso3, series_id, code, unit in plan[:5]:
        print(f"  {country} ({iso3}) {series_id} [{code}]")
    if len(plan) > 5:
        print(f"  ... and {len(plan) - 5} more")
    if not a.run:
        print("\ndry run only. Add --run to execute.")
        return

    session = requests.Session()
    done, errors = 0, []
    for country, iso3, series_id, code, unit in plan:
        outdir = WVS_DATA_DIR / "indicators" / country
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{series_id}.csv"
        source_url = BASE_URL.format(iso3=iso3, code=code) + f"?date={START}:{END}"
        try:
            rows = fetch_series(session, iso3, code)
        except Exception as exc:
            print(f"{country} {series_id}: FAILED -- {exc}", flush=True)
            errors.append((country, series_id, str(exc)))
            continue
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "value", "unit", "frequency", "source_url"])
            for row in sorted(rows, key=lambda r: r["date"]):
                if row["value"] is None:
                    continue
                w.writerow([row["date"], row["value"], unit, "annual", source_url])
        n_written = sum(1 for r in rows if r["value"] is not None)
        print(f"{country} {series_id}: {n_written}/{len(rows)} years written -> {outpath}", flush=True)
        done += 1
        time.sleep(0.3)

    print(f"\n{done}/{len(plan)} series fetched, {len(errors)} errors")
    if errors:
        print("failed:", errors)


if __name__ == "__main__":
    main()
