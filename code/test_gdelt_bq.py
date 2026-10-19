"""Offline tests for gdelt_bq.py: a fake backend stands in for BigQuery, nothing touches the network.

Run: python -m pytest code/test_gdelt_bq.py   (or: python code/test_gdelt_bq.py)
"""
import datetime as dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g

D = dt.date


class FakeBackend:
    def __init__(self, est=1 * g.GB, billed=1 * g.GB):
        self.est, self.billed = est, billed
        self.dry_calls = self.run_calls = 0
        self.last_max_bytes = None

    def dry_run(self, sql, params):
        self.dry_calls += 1
        return self.est

    def run(self, sql, params, max_bytes):
        self.run_calls += 1
        self.last_max_bytes = max_bytes
        return [{"MonthYear": 201101, "n_events": 5}], self.billed


def client(tmp, **kw):
    be = kw.pop("backend", FakeBackend())
    return g.GDELTClient(backend=be, directory=tmp, **kw), be


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f"{exc.__name__} not raised")


def q3():
    return g.events_by_month("EGY", D(2011, 1, 1), D(2011, 3, 31))


def test_default_is_dry_run_and_creates_no_job():
    with tempfile.TemporaryDirectory() as t:
        c, be = client(t)
        res = c.fetch(q3())
        assert isinstance(res, g.DryRun) and res.estimated_gb == 1.0
        assert be.run_calls == 0 and c.month_bytes() == 0


def test_run_sets_byte_cap_records_ledger_and_caches():
    with tempfile.TemporaryDirectory() as t:
        c, be = client(t)
        rows = c.fetch(q3(), run=True)
        assert rows[0]["n_events"] == 5 and be.last_max_bytes == g.DEFAULT_MAX_BYTES
        assert c.month_bytes() == g.GB
        again = c.fetch(q3(), run=True)            # cache hit: no new dry run, no new job
        assert again == rows and be.run_calls == 1 and be.dry_calls == 1


def test_over_cap_estimate_is_refused():
    with tempfile.TemporaryDirectory() as t:
        c, be = client(t, backend=FakeBackend(est=50 * g.GB))
        raises(g.CostGuardError, c.fetch, q3(), run=True)
        assert be.run_calls == 0


def test_monthly_budget_blocks():
    with tempfile.TemporaryDirectory() as t:
        c, be = client(t, monthly_budget=int(1.5 * g.GB))
        c.fetch(q3(), run=True)                                           # 1 GB used
        other = g.events_by_month("EGY", D(2012, 1, 1), D(2012, 3, 31))   # not cached; would reach 2 GB
        raises(g.CostGuardError, c.fetch, other, run=True)
        assert be.run_calls == 1


def test_range_too_long_refused():
    raises(g.CostGuardError, g.events_by_month, "EGY", D(2010, 1, 1), D(2012, 1, 1))
    raises(ValueError, g.events_by_month, "EGY", D(2011, 3, 1), D(2011, 1, 1))


def test_columns_whitelist_and_no_star():
    raises(ValueError, g.events_raw, "EGY", D(2011, 1, 1), D(2011, 1, 31), ["*"])
    raises(ValueError, g.events_raw, "EGY", D(2011, 1, 1), D(2011, 1, 31), ["Actor1Name; DROP"])
    raises(ValueError, g.events_raw, "EGY", D(2011, 1, 1), D(2011, 1, 31), [])
    assert "SQLDATE" in g.events_raw("EGY", D(2011, 1, 1), D(2011, 1, 31), ["SQLDATE"]).sql


def test_check_sql_rejects_tampered_queries():
    q = q3()
    for bad in (q.sql.replace("_PARTITIONTIME >= @p_start AND _PARTITIONTIME < @p_end", "1=1"),
                q.sql.replace(g.EVENTS_TABLE, "gdelt-bq.gdeltv2.events"),
                q.sql.replace("MonthYear, EventRootCode,", "*,", 1),
                q.sql + ";"):
        raises(g.CostGuardError, g.check_sql, g.Query(bad, q.params, q.start, q.end))
    c, be = client(tempfile.mkdtemp())
    raises(g.CostGuardError, c.fetch, g.Query(q.sql.replace(g.EVENTS_TABLE, "x.y.z"), q.params))
    assert be.dry_calls == 0


def test_country_codes():
    assert g.to_fips("EGY") == "EG" and g.to_fips("eg") == "EG"
    raises(ValueError, g.to_fips, "ZZZ")
    raises(ValueError, g.to_fips, "E'; --")
    assert "EG'" not in q3().sql and q3().params["country"] == ("STRING", "EG")   # value is a parameter, not in the SQL


def test_corrupt_lines_are_skipped():
    with tempfile.TemporaryDirectory() as t:
        c, be = client(t)
        c.fetch(q3(), run=True)
        for p in [c.ledger_path] + list((Path(t) / "cache").glob("*.jsonl")):
            with open(p, "a") as f:
                f.write('{"truncated": \n')
        assert c.month_bytes() == g.GB
        assert c.fetch(q3(), run=True)[0]["n_events"] == 5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")
