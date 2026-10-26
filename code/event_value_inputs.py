"""Inputs for the event-share vs survey-answer correlations (stage 1: event levels, stage 2: survey anchor shares).

    python code/event_value_inputs.py        # writes event_levels.csv and anchors.parquet to gdelt_plots/event_value_correlations/

Event level of a country in a year = mean over the year's months of (events of that CAMEO root category / all events recorded for the country
that month), in percent (months with no events are skipped; 2023 has Jan-Jul only). A country-wave is tied to its survey year (the year with the
most respondents). Survey anchor of a question in a country-wave = percent of valid respondents on the HIGH side of the scale (code above the
midpoint of the scale; for yes/no and select-many items that is the higher code), for ordinal, binary and select-many questions; at least 100
valid respondents are required. Unweighted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402
from config import PLOTS_DIR  # noqa: E402
from change_detection import OFF_SCALE, load_question_info, scale_range  # noqa: E402
from direction_baseline import load_countries_waves  # noqa: E402

OUT = PLOTS_DIR / "event_value_correlations"
SRC = g.DEFAULT_DIR / "pulls" / "events_daily_2003-11-01_2023-07-31.parquet"
COUNTRIES = g.WVS_COMMON_COUNTRIES
FIPS2ISO = {g.to_fips(c): c for c in COUNTRIES}
ROOTS = [f"{i:02d}" for i in range(1, 21)]
FORMATS = {"ordinal_scale", "binary", "mention_flag"}


def event_levels():
    df = pd.read_parquet(SRC, columns=["date", "country", "event_code", "n_events"])
    df = df[df["country"].isin(FIPS2ISO)].copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    total = df.groupby(["country", "month"])["n_events"].sum().rename("total")
    ok = df[df["event_code"].str.fullmatch(r"(0[1-9]|1[0-9]|20)\d{1,2}")].assign(root=lambda x: x["event_code"].str[:2])
    m = ok.groupby(["country", "month", "root"])["n_events"].sum().unstack("root", fill_value=0).reindex(columns=ROOTS, fill_value=0)
    m = m.join(total, how="right").fillna(0)
    share = m[ROOTS].div(m["total"].replace(0, np.nan), axis=0) * 100
    share["year"] = share.index.get_level_values("month").year
    lv = share.groupby(["country", "year"])[ROOTS].mean()
    lv.index = pd.MultiIndex.from_tuples([(FIPS2ISO[c], y) for c, y in lv.index], names=["country", "year"])
    return lv


def anchors():
    cats, parsed = load_question_info()
    cats = cats.set_index("code")
    qs = [c for c in cats.index if cats.loc[c, "format"] in FORMATS and "+" not in c]
    df = load_countries_waves(COUNTRIES, [5, 6, 7])
    df = df[df["S002VS"].isin([5, 6, 7])].copy()
    df["cw"] = df["COUNTRY_ALPHA"] + "|" + df["S002VS"].astype(str)
    year = df.groupby("cw")["S020"].agg(lambda s: int(s.mode().iloc[0]))
    out = []
    for q in qs:
        if q not in df.columns:
            continue
        v = pd.to_numeric(df[q], errors="coerce")
        valid = v.notna() & (v >= 0)
        if q in OFF_SCALE:
            valid &= ~v.isin(OFF_SCALE[q])
        if valid.sum() < 500:
            continue
        rng = scale_range(q, parsed) or (v[valid].min(), v[valid].max())
        if rng[1] == rng[0]:
            continue
        high = valid & (v > (rng[0] + rng[1]) / 2)
        n = valid.groupby(df["cw"]).sum()
        k = high.groupby(df["cw"]).sum()
        r = pd.DataFrame({"n": n, "high": k})
        r = r[r["n"] >= 100]
        r["share_high"] = 100 * r["high"] / r["n"]
        r["code"] = q
        out.append(r[["code", "n", "share_high"]].reset_index())
    a = pd.concat(out, ignore_index=True)
    a["country"] = a["cw"].str.split("|").str[0]
    a["wave"] = a["cw"].str.split("|").str[1].astype(int)
    a["year"] = a["cw"].map(year)
    return a.drop(columns="cw")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lv = event_levels()
    lv.to_csv(OUT / "event_levels.csv")
    print("event levels:", lv.shape, "country-years;", lv.index.get_level_values("year").min(), "-", lv.index.get_level_values("year").max())
    a = anchors()
    a.to_parquet(OUT / "anchors.parquet", index=False)
    print("anchors:", len(a), "rows;", a["code"].nunique(), "questions;", a.groupby(["country", "wave"]).ngroups, "country-waves")
    print(a.groupby("code").size().describe().round(1).to_dict())


if __name__ == "__main__":
    main()
