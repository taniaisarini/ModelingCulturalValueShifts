"""Pull daily GDELT event statistics (per day, country and full CAMEO event code) and store them locally.

    python code/gdelt_pull.py --start 2016-01-01 --end 2016-06-30              # dry run: SQL and estimated GB, nothing billed
    python code/gdelt_pull.py --start 2016-01-01 --end 2016-06-30 --run        # run it

Everything is written under the GDELT directory (see config.py: $GDELT_DIR, code/config.local.toml, or <repo>/data/gdelt): the query cache and ledger
(see gdelt_bq.py) and `pulls/`, which gets the daily table and a weekly table rolled up from it (weeks start on Monday), each as
parquet and csv. The weekly table is computed locally and is exact, because counts and sums add up; the means and standard deviations
are recomputed from n, sum and sum of squares. Needs GOOGLE_APPLICATION_CREDENTIALS and google-cloud-bigquery for the real run.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402

SUMS = ["n_events", "sum_mentions", "sum_mentions_sq", "sum_sources", "sum_articles", "sum_tone", "sum_tone_sq"]


def to_frame(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d")
    df["goldstein"] = df["goldstein"].round(4)         # constant per event code; BigQuery's float noise is ~1e-7
    return df.drop(columns="SQLDATE")[["date", "country", "event_code"] + SUMS + ["min_tone", "max_tone", "goldstein"]]


def to_weekly(daily):
    d = daily.assign(week_start=daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D"))
    w = d.groupby(["week_start", "country", "event_code"], as_index=False).agg(
        **{c: (c, "sum") for c in SUMS}, min_tone=("min_tone", "min"), max_tone=("max_tone", "max"), goldstein=("goldstein", "first"))
    assert (d.groupby("event_code")["goldstein"].nunique() <= 1).all()       # constant per event code
    return w


def add_means(df):
    """Mean and standard deviation of tone and mentions from the stored sums (population variance over events)."""
    n = df["n_events"]
    out = df.copy()
    for k, s, sq in [("tone", "sum_tone", "sum_tone_sq"), ("mentions", "sum_mentions", "sum_mentions_sq")]:
        out[f"mean_{k}"] = df[s] / n
        out[f"sd_{k}"] = np.sqrt(np.maximum(df[sq] / n - (df[s] / n) ** 2, 0))
    return out


def save(df, stem, outdir):
    paths = {"parquet": outdir / f"{stem}.parquet", "csv": outdir / f"{stem}.csv"}
    df.to_parquet(paths["parquet"], index=False, compression="zstd")
    df.to_csv(paths["csv"], index=False)
    return {k: p.stat().st_size for k, p in paths.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--end", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--countries", nargs="*", default=g.WVS_COMMON_COUNTRIES, help="ISO3 codes (default: the 29 in WVS waves 5-7)")
    ap.add_argument("--run", action="store_true", help="actually execute (default: dry run only)")
    a = ap.parse_args()

    client = g.GDELTClient()
    q = g.events_daily_stats(a.countries, a.start, a.end)
    print(q.sql, "\nparams:", {k: v for k, v in q.params.items() if k == "countries"})
    res = client.fetch(q, run=a.run)
    if isinstance(res, g.DryRun):
        print(f"\nDRY RUN: would scan ~{res.estimated_gb:.3f} GB (month so far: {client.month_bytes() / g.GB:.2f} GB). Add --run to execute.")
        return

    daily = to_frame(res)
    weekly = to_weekly(daily)
    outdir = client.dir / "pulls"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.start}_{a.end}"
    sd = save(add_means(daily), f"events_daily_{tag}", outdir)
    sw = save(add_means(weekly), f"events_weekly_{tag}", outdir)
    print(f"\nrows: daily {len(daily):,}  weekly {len(weekly):,}  ({len(daily) / len(weekly):.1f}x)")
    mb = lambda b: f"{b / 1e6:.1f} MB"  # noqa: E731
    print(f"daily : parquet {mb(sd['parquet'])}, csv {mb(sd['csv'])}\nweekly: parquet {mb(sw['parquet'])}, csv {mb(sw['csv'])}")
    print(f"events in the pull: {int(daily['n_events'].sum()):,}; distinct event codes {daily['event_code'].nunique()}; month total billed {client.month_bytes() / g.GB:.3f} GB")
    per = daily.groupby("country")["n_events"].sum()
    missing = [c for c in a.countries if g.to_fips(c) not in per.index]
    print("events per country (FIPS):", per.sort_values(ascending=False).to_dict())
    print("countries with NO rows (check the code mapping):", missing or "none")
    print(f"written to {outdir}")


if __name__ == "__main__":
    main()
