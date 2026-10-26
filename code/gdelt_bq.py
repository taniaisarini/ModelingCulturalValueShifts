"""Cost-safe access to GDELT 2.0 Events on BigQuery.

Design goal: it should be impossible to be billed by accident. Layers, in the order a query meets them:

  1. No raw SQL. Callers ask for (country, start, end); SQL comes from a fixed template, values travel as
     query parameters, columns come from a whitelist (never `*`).
  2. `check_sql()` re-verifies the final text: only the partitioned events table, a bounded `_PARTITIONTIME`
     range, no `*`, a date span of at most `max_days`.
  3. Result cache: an identical query is never sent twice (`data/gdelt/cache/`).
  4. Free dry run; refused if the estimated bytes exceed `max_bytes`.
  5. Local monthly ledger of bytes billed; refused if this query would push the month over `monthly_budget`.
  6. Only with `run=True` is a real job created, and it carries `maximum_bytes_billed=max_bytes`, so BigQuery
     itself rejects (without charging) anything bigger than the dry run promised.

`run` defaults to False everywhere: by default you only get the SQL and the estimated size.

The real backstops are outside this file and should be set once in the Google Cloud console: use a *sandbox*
project (no billing account linked) and a custom daily "Query usage" quota. The local ledger only knows about
queries made through this module.

Usage (dry run, prints SQL and estimated GB; add --run to execute):
    python code/gdelt_bq.py --country EGY --start 2011-01-01 --end 2011-03-31
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Cache, ledger and pulled data live here (set in code/config.py: GDELT_DIR environment variable, code/config.local.toml, or <repo>/data/gdelt).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GDELT_DIR as DEFAULT_DIR  # noqa: E402

# Both tables are day-partitioned (verified with metadata calls). v1 is the GDELT 1.0 history, v2 starts 2015-02-18.
# Event counts are NOT comparable across the two (v2 has many more sources); compare shares/tone, not raw counts.
TABLES = {"v1": "gdelt-bq.full.events_partitioned", "v2": "gdelt-bq.gdeltv2.events_partitioned"}
EVENTS_TABLE = TABLES["v2"]
V2_START = dt.date(2015, 2, 18)

GB = 10 ** 9
DEFAULT_MAX_BYTES = 20 * GB          # per query
DEFAULT_MONTHLY_BUDGET = 800 * GB    # 80% of the 1 TB/month free tier
DEFAULT_MAX_DAYS = 366
PARTITION_SLACK_DAYS = 3             # events are ingested a little after their SQLDATE

# GDELT 2.0 Events columns we allow to be selected in a raw pull.
ALLOWED_COLUMNS = {
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "IsRootEvent",
    "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Name", "Actor1CountryCode", "Actor1Type1Code",
    "Actor2Name", "Actor2CountryCode", "Actor2Type1Code",
    "ActionGeo_CountryCode", "ActionGeo_FullName", "SOURCEURL",
}

# WVS uses ISO alpha-3, GDELT uses FIPS 10-4. Add countries as needed.
ISO3_TO_FIPS = {"EGY": "EG", "THA": "TH", "UKR": "UP", "URY": "UY", "HKG": "HK", "USA": "US", "IND": "IN", "TUR": "TU",
                # the other countries present in WVS waves 5, 6 and 7
                "ARG": "AR", "AUS": "AS", "BRA": "BR", "CHL": "CI", "CHN": "CH", "COL": "CO", "CYP": "CY", "DEU": "GM",
                "IRQ": "IZ", "JOR": "JO", "JPN": "JA", "KOR": "KS", "MAR": "MO", "MEX": "MX", "MYS": "MY", "NLD": "NL",
                "NZL": "NZ", "PER": "PE", "ROU": "RO", "RUS": "RS", "TWN": "TW"}
WVS_COMMON_COUNTRIES = ["ARG", "AUS", "BRA", "CHL", "CHN", "COL", "CYP", "DEU", "EGY", "HKG", "IND", "IRQ", "JOR", "JPN", "KOR",
                        "MAR", "MEX", "MYS", "NLD", "NZL", "PER", "ROU", "RUS", "THA", "TUR", "TWN", "UKR", "URY", "USA"]


class CostGuardError(RuntimeError):
    """Raised instead of sending anything that could cost money."""


@dataclass
class Query:
    sql: str
    params: dict = field(default_factory=dict)   # name -> (bigquery type, value)
    start: dt.date = None
    end: dt.date = None
    table: str = EVENTS_TABLE


@dataclass
class DryRun:
    sql: str
    params: dict
    estimated_bytes: int

    @property
    def estimated_gb(self):
        return self.estimated_bytes / GB


# ---------------------------------------------------------------- query building

def to_fips(country):
    c = country.strip().upper()
    if len(c) == 3:
        if c not in ISO3_TO_FIPS:
            raise ValueError(f"No FIPS code known for {c}; add it to ISO3_TO_FIPS")
        return ISO3_TO_FIPS[c]
    if not re.fullmatch(r"[A-Z]{2}", c):
        raise ValueError(f"Country must be an ISO3 or 2-letter FIPS code, got {country!r}")
    return c


def _check_dates(start, end, max_days):
    if not (isinstance(start, dt.date) and isinstance(end, dt.date)):
        raise ValueError("start and end must be datetime.date")
    if end < start:
        raise ValueError("end is before start")
    if (end - start).days + 1 > max_days:
        raise CostGuardError(f"Range is {(end - start).days + 1} days; the limit is {max_days}")


def pick_table(start, end, table=None):
    """'v1' before 2015-02-18, 'v2' from then on. A range that straddles the boundary is refused: the two
    tables are not comparable, so make two queries instead."""
    if table is not None:
        if table not in TABLES:
            raise ValueError(f"table must be one of {sorted(TABLES)}")
        return TABLES[table]
    if end < V2_START:
        return TABLES["v1"]
    if start >= V2_START:
        return TABLES["v2"]
    raise CostGuardError(f"Range {start}..{end} straddles the GDELT 1.0/2.0 boundary ({V2_START}); split it")


def _build(select_sql, tail_sql, country, start, end, max_days, table=None):
    """`country` is one code (ISO3 or FIPS) or a list of codes. The country filter does not reduce the bytes scanned."""
    _check_dates(start, end, max_days)
    tbl = pick_table(start, end, table)
    many = isinstance(country, (list, tuple))
    p_start = dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc)
    p_end = dt.datetime.combine(end + dt.timedelta(days=PARTITION_SLACK_DAYS + 1), dt.time.min, tzinfo=dt.timezone.utc)
    sql = (
        f"SELECT {select_sql}\n"
        f"FROM `{tbl}`\n"
        "WHERE _PARTITIONTIME >= @p_start AND _PARTITIONTIME < @p_end\n"
        "  AND SQLDATE >= @d_start AND SQLDATE <= @d_end\n"
        f"  AND ActionGeo_CountryCode {'IN UNNEST(@countries)' if many else '= @country'}\n"
        f"{tail_sql}"
    )
    params = {
        "p_start": ("TIMESTAMP", p_start), "p_end": ("TIMESTAMP", p_end),
        "d_start": ("INT64", int(start.strftime("%Y%m%d"))), "d_end": ("INT64", int(end.strftime("%Y%m%d"))),
    }
    if many:
        params["countries"] = ("ARRAY<STRING>", [to_fips(c) for c in country])
    else:
        params["country"] = ("STRING", to_fips(country))
    q = Query(sql=sql, params=params, start=start, end=end, table=tbl)
    check_sql(q, max_days)
    return q


def events_by_month(country, start, end, max_days=DEFAULT_MAX_DAYS, table=None):
    """Monthly counts per CAMEO root code (the same shape as data/gdelt/egypt_test_results.csv), plus tone."""
    return _build(
        "MonthYear, EventRootCode, COUNT(*) AS n_events, SUM(NumMentions) AS n_mentions, "
        "AVG(AvgTone) AS avg_tone, AVG(GoldsteinScale) AS avg_goldstein",
        "GROUP BY MonthYear, EventRootCode\nORDER BY MonthYear, EventRootCode",
        country, start, end, max_days, table)


def events_daily_stats(countries, start, end, max_days=DEFAULT_MAX_DAYS, table=None):
    """One row per day, country and full CAMEO event code, with the sufficient statistics (count, sums, sums of squares,
    min/max) of mentions and tone. Means and standard deviations can be recomputed exactly after rolling up to weeks or
    months, or from full codes to base and root codes. (POW instead of x*x: check_sql forbids `*`.)"""
    return _build(
        "SQLDATE, ActionGeo_CountryCode AS country, EventCode AS event_code, COUNT(1) AS n_events, "
        "SUM(NumMentions) AS sum_mentions, SUM(POW(NumMentions, 2)) AS sum_mentions_sq, "
        "SUM(NumSources) AS sum_sources, SUM(NumArticles) AS sum_articles, "
        "SUM(AvgTone) AS sum_tone, SUM(POW(AvgTone, 2)) AS sum_tone_sq, MIN(AvgTone) AS min_tone, MAX(AvgTone) AS max_tone, "
        "AVG(GoldsteinScale) AS goldstein",
        "GROUP BY SQLDATE, ActionGeo_CountryCode, EventCode\nORDER BY SQLDATE, ActionGeo_CountryCode, EventCode",
        list(countries), start, end, max_days, table)


def events_raw(country, start, end, columns, max_days=DEFAULT_MAX_DAYS, table=None):
    """Row-level events with an explicit column list. Prefer events_by_month; raw pulls can be large."""
    columns = list(columns)
    bad = [c for c in columns if c not in ALLOWED_COLUMNS]
    if not columns or bad:
        raise ValueError(f"Columns must be a non-empty subset of ALLOWED_COLUMNS; bad: {bad}")
    return _build(", ".join(columns), "", country, start, end, max_days, table)


def check_sql(q, max_days=DEFAULT_MAX_DAYS):
    """Last line of defence: refuse any query text that is not a bounded, partition-filtered events query."""
    sql = q.sql
    if re.search(r"\*", re.sub(r"COUNT\(\*\)", "", sql)):
        raise CostGuardError("`*` is not allowed")
    tables = set(re.findall(r"`([^`]+)`", sql))
    if q.table not in TABLES.values() or tables != {q.table}:
        raise CostGuardError(f"Only {sorted(TABLES.values())} may be queried, found {tables}")
    if "_PARTITIONTIME >= @p_start" not in sql or "_PARTITIONTIME < @p_end" not in sql:
        raise CostGuardError("Missing bounded _PARTITIONTIME filter")
    if re.search(r";|--", sql):
        raise CostGuardError("Multiple statements or comments are not allowed")
    p0, p1 = q.params["p_start"][1], q.params["p_end"][1]
    if (p1 - p0).days > max_days + PARTITION_SLACK_DAYS + 1:
        raise CostGuardError("Partition range too wide")


# ---------------------------------------------------------------- backends

class BigQueryBackend:
    """Thin wrapper over google-cloud-bigquery. Imported lazily so the rest works (and is testable) without it."""

    def __init__(self, project=None):
        from google.cloud import bigquery   # pip install google-cloud-bigquery
        self._bq = bigquery
        self.client = bigquery.Client(project=project)

    def _params(self, params):
        out = []
        for n, (t, v) in params.items():
            if t.startswith("ARRAY<"):
                out.append(self._bq.ArrayQueryParameter(n, t[6:-1], v))
            else:
                out.append(self._bq.ScalarQueryParameter(n, t, v))
        return out

    def dry_run(self, sql, params):
        cfg = self._bq.QueryJobConfig(query_parameters=self._params(params), dry_run=True, use_query_cache=False)
        return self.client.query(sql, job_config=cfg).total_bytes_processed

    def run(self, sql, params, max_bytes):
        cfg = self._bq.QueryJobConfig(query_parameters=self._params(params), maximum_bytes_billed=max_bytes,
                                      use_query_cache=True)
        job = self.client.query(sql, job_config=cfg)
        rows = [dict(r) for r in job.result()]
        return rows, int(job.total_bytes_billed or 0)

    def table_info(self, table):
        t = self.client.get_table(table)   # metadata call, free
        return {"time_partitioning": str(t.time_partitioning), "num_bytes": t.num_bytes, "num_rows": t.num_rows}


# ---------------------------------------------------------------- client

def _key(q):
    payload = {"sql": q.sql, "params": {k: [t, str(v)] for k, (t, v) in sorted(q.params.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_jsonl(path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue   # truncated line from a crash mid-write


class GDELTClient:
    def __init__(self, backend=None, directory=DEFAULT_DIR, max_bytes=DEFAULT_MAX_BYTES,
                 monthly_budget=DEFAULT_MONTHLY_BUDGET, max_days=DEFAULT_MAX_DAYS, project=None):
        self._backend = backend
        self._project = project
        self.dir = Path(directory)
        self.max_bytes = max_bytes
        self.monthly_budget = monthly_budget
        self.max_days = max_days

    @property
    def backend(self):
        if self._backend is None:
            self._backend = BigQueryBackend(self._project)
        return self._backend

    # -- cache / ledger (append-only JSONL; NFS-safe, no SQLite)
    def _cache_path(self, key):
        return self.dir / "cache" / f"{key[:2]}.jsonl"

    def _cache_get(self, key):
        for rec in _read_jsonl(self._cache_path(key)):
            if rec.get("key") == key:
                return rec
        return None

    def _cache_put(self, key, q, rows, billed):
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"key": key, "sql": q.sql, "params": {k: [t, str(v)] for k, (t, v) in q.params.items()},
               "rows": rows, "bytes_billed": billed, "ts": time.time()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    @property
    def ledger_path(self):
        return self.dir / "ledger.jsonl"

    def month_bytes(self, month=None):
        month = month or dt.date.today().strftime("%Y-%m")
        return sum(r.get("bytes", 0) for r in _read_jsonl(self.ledger_path) if r.get("month") == month)

    def _ledger_add(self, q, billed):
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {"month": dt.date.today().strftime("%Y-%m"), "bytes": billed, "ts": time.time(), "key": _key(q)}
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # -- main entry point
    def fetch(self, q, run=False):
        """Return cached rows (list of dicts) if available, else a DryRun, else run the query and return rows."""
        check_sql(q, self.max_days)
        key = _key(q)
        cached = self._cache_get(key)
        if cached is not None:
            return cached["rows"]

        est = self.backend.dry_run(q.sql, q.params)
        if est > self.max_bytes:
            raise CostGuardError(f"Estimated {est / GB:.2f} GB exceeds per-query cap {self.max_bytes / GB:.2f} GB")
        used = self.month_bytes()
        if used + est > self.monthly_budget:
            raise CostGuardError(f"Would bring this month to {(used + est) / GB:.1f} GB, over the "
                                 f"{self.monthly_budget / GB:.0f} GB budget ({used / GB:.1f} GB used)")
        if not run:
            return DryRun(q.sql, q.params, est)

        rows, billed = self.backend.run(q.sql, q.params, self.max_bytes)
        self._ledger_add(q, billed)
        self._cache_put(key, q, rows, billed)
        return rows

    def check_table(self, table="v2"):
        """Free metadata call: confirm the table exists and is time-partitioned."""
        return self.backend.table_info(TABLES[table])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, help="ISO3 (EGY) or FIPS (EG)")
    ap.add_argument("--start", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--end", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--project", default=None)
    ap.add_argument("--run", action="store_true", help="actually execute (default: dry run only)")
    args = ap.parse_args()
    client = GDELTClient(project=args.project)
    q = events_by_month(args.country, args.start, args.end)
    print(q.sql)
    res = client.fetch(q, run=args.run)
    if isinstance(res, DryRun):
        print(f"\nDRY RUN: would scan ~{res.estimated_gb:.3f} GB (month so far: {client.month_bytes() / GB:.2f} GB). "
              "Re-run with --run to execute.")
    else:
        print(f"\n{len(res)} rows; month total {client.month_bytes() / GB:.3f} GB billed")


if __name__ == "__main__":
    main()
