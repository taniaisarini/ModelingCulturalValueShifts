"""Pull row-level GDELT events (not aggregated), all 29 WVS wave 5-7 countries, and store them locally
partitioned by country/year/month for easy lookup.

    python code/gdelt_pull_raw_events.py --year 2011              # one year, dry run: SQL and estimated GB, nothing billed
    python code/gdelt_pull_raw_events.py --year 2011 --run         # one year, run it
    python code/gdelt_pull_raw_events.py --start 2003-11-01 --end 2023-07-31 --run   # arbitrary range; with
                                                                                       # no --start/--end and no
                                                                                       # --year, defaults to the
                                                                                       # full WVS wave 5-7 range

Columns pulled (18 of the 23 in gdelt_bq.ALLOWED_COLUMNS; drops MonthYear, Year, EventBaseCode, EventRootCode,
QuadClass, which are exact derivations of SQLDATE/EventCode and add nothing): GLOBALEVENTID, SQLDATE, IsRootEvent,
EventCode, GoldsteinScale, NumMentions, NumSources, NumArticles, AvgTone, Actor1Name, Actor1CountryCode,
Actor1Type1Code, Actor2Name, Actor2CountryCode, Actor2Type1Code, ActionGeo_CountryCode, ActionGeo_FullName,
SOURCEURL.

A year that straddles the GDELT 1.0/2.0 boundary (2015-02-18) is split into two queries automatically (gdelt_bq
refuses to query across it; the two tables are not comparable). Output goes under RAW_EVENTS_DIR (see config.py:
$RAW_EVENTS_DIR, code/config.local.toml, or <GDELT_DIR>/raw_events), one parquet file per country x month:
    <RAW_EVENTS_DIR>/country=<FIPS>/year=<YYYY>/month=<MM>.parquet
Needs GOOGLE_APPLICATION_CREDENTIALS and google-cloud-bigquery for the real run; see GDELT_BIGQUERY.md.

One full year of row-level events (wide text columns: actor names, URLs) is too big to materialize in memory at
once -- a first attempt pulling 2019 in a single query hit an OOM kill on this machine's cgroup limit (~9.5 GB
RSS) partway through turning BigQuery's results into a Python list of dicts, *after* BigQuery had already run
and billed the query (confirmed via the BigQuery job history: billed, not a cache hit -- that 18.66 GB was
wasted and had to be added back into the ledger by hand). Two independent fixes, since one month's worth alone
is not a guarantee: (1) queried and written one *calendar month* at a time, writing and releasing each month's
data before starting the next, so peak memory does not grow with the year's total size; (2) results are now
converted with BigQuery's own to_dataframe() (gdelt_bq.GDELTClient.fetch_dataframe), which builds typed
columnar arrays directly instead of a Python dict per row -- meaningfully less overhead for a wide, string-heavy
schema. A soft RLIMIT_AS memory cap (see MEMORY_CAP_GB below) is also set so a run that still goes over raises a
catchable MemoryError instead of a hard, silent cgroup SIGKILL. Resumable: a month whose output already exists
(a marker under <RAW_EVENTS_DIR>/_pulled/) is skipped unless --no-resume, in case a run is interrupted partway.
"""
import argparse
import datetime as dt
import resource
import sys
from itertools import groupby
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402
from config import RAW_EVENTS_DIR  # noqa: E402
from gdelt_sweep import chunks  # noqa: E402

D = dt.date

FULL_COLS = set(g.ALLOWED_COLUMNS)
DROP_COLS = {"MonthYear", "Year", "EventBaseCode", "EventRootCode", "QuadClass"}  # exact derivations, see docstring
COLUMNS = sorted(FULL_COLS - DROP_COLS)

MEMORY_CAP_GB = 7  # default soft cap, below the ~9.5 GB cgroup limit seen in the original interactive-session
                   # crash; override with --mem-cap-gb when running under a Slurm allocation with more headroom
                   # (e.g. 12 for a 16G job) -- a 2015 chunk with ~3M rows still hit this default under Slurm


def to_frame(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # NOTE: repeated GLOBALEVENTID rows were seen scattered across several months and once looked like a data
    # quality issue, but dropping them (an earlier version of this function did) was WRONG -- cross-checking
    # against the independent aggregated sweep's COUNT(1) showed the "duplicate" rows are part of the true,
    # expected count (GDELT's own table legitimately has more than one row per event ID in some cases; COUNT(1)
    # on the raw table counts them too). Keep them. The in-place-deduped files from that mistake were restored.
    return df.rename(columns={"ActionGeo_CountryCode": "country"}).drop(columns="SQLDATE")


def save_partitioned(df, outdir):
    """One parquet file per (country, year, month); overwrites any file already there for that partition."""
    written = []
    for (country, year, month), part in df.groupby(["country", "year", "month"]):
        d = outdir / f"country={country}" / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"month={month:02d}.parquet"
        part.drop(columns=["country", "year", "month"]).to_parquet(f, index=False, compression="zstd")
        written.append((country, year, month, len(part), f))
    return written


def month_pieces(start, end):
    """Split a (start, end) range into calendar-month pieces, so each query/materialization stays small."""
    out, cur = [], start
    while cur <= end:
        next_month = D(cur.year + 1, 1, 1) if cur.month == 12 else D(cur.year, cur.month + 1, 1)
        out.append((cur, min(end, next_month - dt.timedelta(days=1))))
        cur = next_month
    return out


def marker_path(outdir, year, month):
    return outdir / "_pulled" / f"{year}-{month:02d}.done"


WVS_RANGE = (D(2003, 11, 1), D(2023, 7, 31))  # one year before wave 5 fieldwork start, through wave 7 fieldwork end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, help="pull just this one calendar year (mutually exclusive with --start/--end)")
    ap.add_argument("--month", type=int, choices=range(1, 13), help="with --year, restrict to one calendar month (for testing)")
    ap.add_argument("--start", type=D.fromisoformat, help="range start (default: full WVS range, 2003-11-01)")
    ap.add_argument("--end", type=D.fromisoformat, help="range end (default: full WVS range, 2023-07-31)")
    ap.add_argument("--countries", nargs="*", default=g.WVS_COMMON_COUNTRIES, help="ISO3 codes (default: the 29 in WVS waves 5-7)")
    ap.add_argument("--run", action="store_true", help="actually execute (default: dry run only)")
    ap.add_argument("--no-resume", action="store_true", help="re-pull months that already have a _pulled marker")
    ap.add_argument("--mem-cap-gb", type=float, default=MEMORY_CAP_GB, help=f"soft RLIMIT_AS cap (default {MEMORY_CAP_GB})")
    a = ap.parse_args()

    if a.year is not None:
        range_start, range_end = D(a.year, 1, 1), D(a.year, 12, 31)
    else:
        range_start = a.start or WVS_RANGE[0]
        range_end = a.end or WVS_RANGE[1]

    soft_cap = int(a.mem_cap_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (soft_cap, soft_cap))
    print(f"soft memory cap set: {a.mem_cap_gb} GB (raises MemoryError instead of a hard cgroup kill if exceeded)\n")

    client = g.GDELTClient()
    outdir = RAW_EVENTS_DIR
    range_pieces = chunks(range_start, range_end)  # splits at the 2015-02-18 1.0/2.0 boundary
    pieces = [mp for yp in range_pieces for mp in month_pieces(*yp)]
    if a.month is not None:
        pieces = [(s, e) for s, e in pieces if s.month == a.month or e.month == a.month]
    print(f"range {range_start} .. {range_end}{f' month {a.month:02d}' if a.month else ''}: "
          f"{len(pieces)} monthly chunk(s), columns: {COLUMNS}\n")

    plan, total = [], 0
    for s, e in pieces:
        done = marker_path(outdir, s.year, s.month).exists()
        if done and not a.no_resume:
            plan.append((s, e, None, 0.0)); print(f"{s} .. {e}  already pulled, skipping")
            continue
        q = g.events_raw(a.countries, s, e, COLUMNS)
        res = client.fetch_dataframe(q, run=False)
        if isinstance(res, g.DryRun):
            plan.append((s, e, q, res.estimated_gb)); total += res.estimated_bytes
            print(f"{s} .. {e}  {q.table.split('.')[1]:6s} would scan {res.estimated_gb:6.2f} GB")
        else:
            plan.append((s, e, q, 0.0)); print(f"{s} .. {e}  cached ({len(res):,} rows)")
    print(f"\n{len(plan)} chunk(s); to be billed {total / g.GB:.2f} GB; month so far {client.month_bytes() / g.GB:.2f} GB; "
          f"after this {(client.month_bytes() + total) / g.GB:.2f} GB of 1,100 GB free")
    if not a.run:
        print("dry run only. Add --run to execute.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    per_country_total = {}
    failed_chunks = []
    # Group by calendar month before writing: chunks() already splits one month (Feb 2015, the GDELT 1.0/2.0
    # boundary) into two pieces on two different tables. A first version wrote each piece separately to the
    # same output file, so the second piece silently overwrote the first's data instead of combining -- lost
    # roughly half of Feb 2015 for all 29 countries (caught by cross-checking against the independent
    # aggregated sweep). Fix: fetch every piece of a month, concatenate, THEN write once.
    month_groups = [(key, list(grp)) for key, grp in groupby(plan, key=lambda item: (item[0].year, item[0].month))]
    for (year, month), group in month_groups:
        if all(q is None for _, _, q, _ in group):
            continue  # resumed: already on disk
        s, e = group[0][0], group[-1][1]
        try:
            parts = []
            total_billed = 0.0
            for gs, ge, gq, _ in group:
                if gq is None:
                    continue  # one piece of this month already done (shouldn't happen alongside a fresh one, but safe)
                raw_df, billed = client.fetch_dataframe(gq, run=True)
                total_billed += billed
                parts.append(raw_df)
            mem_mb = sum(p.memory_usage(deep=True).sum() for p in parts) / 1e6
            df = to_frame(pd.concat(parts, ignore_index=True)) if len(parts) > 1 else to_frame(parts[0])
            del parts
            written = save_partitioned(df, outdir)
            n = len(df)
            for c, cnt in df.groupby("country").size().items():
                per_country_total[c] = per_country_total.get(c, 0) + cnt
            del df
            billed = total_billed
        except MemoryError:
            # Covers the whole chunk body, not just the fetch: a first version only wrapped fetch_dataframe()
            # and a later in-place pandas op (to_frame's rename/drop, which allocates new arrays) still hit the
            # cap uncaught and crashed the entire multi-year job over one chunk. The query is already billed
            # (fetch_dataframe updates the ledger before returning) and will be a free cache hit on retry, so
            # the right move is to skip this chunk and keep going -- not stop the whole run -- and let the
            # resume logic (no marker written) pick it up again next time, ideally with a smaller MEMORY_CAP_GB
            # headroom issue fixed (e.g. a higher cap, since this runs in a dedicated Slurm allocation now, not
            # the ~9.5 GB interactive-session cgroup limit this cap was originally tuned for).
            failed_chunks.append((s, e))
            print(f"\n{s} .. {e}: MemoryError at the {a.mem_cap_gb} GB soft cap -- SKIPPING this chunk and "
                  f"continuing. Billed already (cache hit on retry); no marker written, so a future run will "
                  f"retry it.", flush=True)
            continue
        marker_path(outdir, s.year, s.month).parent.mkdir(parents=True, exist_ok=True)
        marker_path(outdir, s.year, s.month).touch()
        total_rows += n
        print(f"{s} .. {e}: {n:,} rows ({mem_mb:,.0f} MB in memory), {len(written)} files written; "
              f"billed {billed / g.GB:.2f} GB; month total {client.month_bytes() / g.GB:.2f} GB", flush=True)

    print(f"\n{total_rows:,} total rows written to {outdir}")
    print("rows per country (FIPS):", dict(sorted(per_country_total.items(), key=lambda kv: -kv[1])))
    missing = [c for c in a.countries if g.to_fips(c) not in per_country_total]
    print("countries with no rows this run:", missing or "none")
    print("chunks skipped on MemoryError (retry with a future run):", failed_chunks or "none")


if __name__ == "__main__":
    main()
