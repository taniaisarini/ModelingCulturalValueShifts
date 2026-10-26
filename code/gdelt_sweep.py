"""Sweep: daily GDELT event statistics for the 29 WVS wave 5-7 countries over a long date range, one query per chunk.

    python code/gdelt_sweep.py                    # dry run: estimated GB per chunk and in total, nothing billed
    python code/gdelt_sweep.py --run              # run every chunk (cached chunks are free) and write the combined files

Default range 2003-11-01 .. 2023-07-31 (one year before the first wave 5 fieldwork month, through the last wave 7 fieldwork month).
Chunks are calendar years, split at 2015-02-18 (GDELT 1.0 table before, 2.0 after; see gdelt_bq.py) and 2016 in two halves so the
Jan-Jun 2016 test pull is reused from the cache. Per-chunk parquet files go to <gdelt dir>/pulls/chunks/, the combined table to
<gdelt dir>/pulls/events_daily_<start>_<end>.parquet and .csv. Columns: date, country (FIPS), event_code, n_events, sum_mentions,
sum_mentions_sq, sum_sources, sum_articles, sum_tone, sum_tone_sq, min_tone, max_tone, goldstein.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402
from gdelt_pull import to_frame  # noqa: E402

D = dt.date


def chunks(start, end):
    cuts = [D(start.year, 1, 1)]
    out = []
    y = start.year
    while D(y, 1, 1) <= end:
        a, b = D(y, 1, 1), D(y, 12, 31)
        pieces = [(a, b)]
        if y == 2015:
            pieces = [(a, D(2015, 2, 17)), (D(2015, 2, 18), b)]
        if y == 2016:
            pieces = [(a, D(2016, 6, 30)), (D(2016, 7, 1), b)]
        out += pieces
        y += 1
    out = [(max(a, start), min(b, end)) for a, b in out if b >= start and a <= end]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=D(2003, 11, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=D(2023, 7, 31))
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    client = g.GDELTClient()
    outdir = client.dir / "pulls"
    (outdir / "chunks").mkdir(parents=True, exist_ok=True)
    plan, total = [], 0
    for s, e in chunks(a.start, a.end):
        q = g.events_daily_stats(g.WVS_COMMON_COUNTRIES, s, e)
        res = client.fetch(q, run=False)                  # cached rows, or a free DryRun
        if isinstance(res, g.DryRun):
            plan.append((s, e, q, res.estimated_gb)); total += res.estimated_bytes
            print(f"{s} .. {e}  {q.table.split('.')[1]:6s} would scan {res.estimated_gb:6.2f} GB")
        else:
            plan.append((s, e, q, 0.0)); print(f"{s} .. {e}  cached ({len(res):,} rows)")
    print(f"\n{len(plan)} chunks; to be billed {total / g.GB:.1f} GB; month so far {client.month_bytes() / g.GB:.2f} GB; "
          f"after this {(client.month_bytes() + total) / g.GB:.1f} GB of 1,100 GB free")
    if not a.run:
        print("dry run only. Add --run to execute.")
        return

    parts = []
    for s, e, q, _ in plan:
        rows = client.fetch(q, run=True)
        df = to_frame(rows)
        del rows
        f = outdir / "chunks" / f"events_daily_{s}_{e}.parquet"
        df.to_parquet(f, index=False, compression="zstd")
        parts.append(f)
        print(f"{s} .. {e}: {len(df):,} rows, {int(df['n_events'].sum()):,} events; month billed {client.month_bytes() / g.GB:.2f} GB", flush=True)

    df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True).sort_values(["date", "country", "event_code"], ignore_index=True)
    dup = df.duplicated(["date", "country", "event_code"]).sum()
    tag = f"{a.start}_{a.end}"
    df.to_parquet(outdir / f"events_daily_{tag}.parquet", index=False, compression="zstd")
    df.to_csv(outdir / f"events_daily_{tag}.csv", index=False)
    print(f"\ncombined: {len(df):,} rows, {int(df['n_events'].sum()):,} events, {df['date'].min().date()} .. {df['date'].max().date()}, duplicate keys: {dup}")
    print("files:", {p.name: f"{p.stat().st_size / 1e6:.0f} MB" for p in sorted(outdir.glob(f'events_daily_{tag}.*'))})
    per_year = df.groupby(df["date"].dt.year).agg(rows=("n_events", "size"), events=("n_events", "sum"))
    print(per_year.to_string())
    miss = [c for c in g.WVS_COMMON_COUNTRIES if g.to_fips(c) not in set(df["country"])]
    print("countries with no rows:", miss or "none")


if __name__ == "__main__":
    main()
