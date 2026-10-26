"""Plots of the GDELT daily file: mean events per day in each month, per CAMEO root category, one figure per country.

    python code/gdelt_plot_monthly.py [--measure share|counts] [--window 3] [EGY THA UKR URY HKG]     # default measure: share

Measures: `share` = a category's events as a percent of ALL events recorded for that country in the same month (every code, including the
few malformed ones, is in the denominator), which removes overall changes in how much GDELT records about the country; `counts` = mean
events per day. Reads <gdelt dir>/pulls/events_daily_2003-11-01_2023-07-31.parquet. Monthly mean = events in the month / days in the month (days with
no row count as zero). One figure per country with 20 small panels (one per root category, each with its own y-axis), saved with its
monthly table to gdelt_plots/. Codes that are not real CAMEO codes ('---', 'X') are dropped. GDELT 1.0 -> 2.0 switch (2015-02-18) is marked.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdelt_bq as g  # noqa: E402
from config import PLOTS_DIR  # noqa: E402

SRC = g.DEFAULT_DIR / "pulls" / "events_daily_2003-11-01_2023-07-31.parquet"
OUT = PLOTS_DIR
ROOTS = {"01": "Make public statement", "02": "Appeal", "03": "Express intent to cooperate", "04": "Consult",
         "05": "Engage in diplomatic cooperation", "06": "Engage in material cooperation", "07": "Provide aid", "08": "Yield",
         "09": "Investigate", "10": "Demand", "11": "Disapprove", "12": "Reject", "13": "Threaten", "14": "Protest",
         "15": "Exhibit force posture", "16": "Reduce relations", "17": "Coerce", "18": "Assault", "19": "Fight",
         "20": "Use unconventional mass violence"}
NAMES = {"EGY": "Egypt", "THA": "Thailand", "UKR": "Ukraine", "URY": "Uruguay", "HKG": "Hong Kong"}
# tokens from the dataviz reference palette (light surface); one series per panel, so the series colour is slot 1
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
START, END = pd.Timestamp("2003-11-01"), pd.Timestamp("2023-07-31")
V2 = pd.Timestamp("2015-02-18")


def monthly_means(df, fips):
    d = df[(df["country"] == fips) & df["event_code"].str.fullmatch(r"(0[1-9]|1[0-9]|20)\d{1,2}")].copy()
    d["root"] = d["event_code"].str[:2]
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    m = d.groupby(["month", "root"])["n_events"].sum().unstack(fill_value=0).reindex(columns=list(ROOTS), fill_value=0)
    months = pd.date_range(START.to_period("M").to_timestamp(), END.to_period("M").to_timestamp(), freq="MS")
    m = m.reindex(months, fill_value=0)
    return m.div(m.index.days_in_month, axis=0)          # events per day, averaged over the month


def monthly_shares(df, fips, window=1):
    """Percent of all events. window>1: centred rolling window over that many months, pooled (events in the window / all events in the
    window); the first and last months use the months available."""
    d = df[df["country"] == fips].copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    total = d.groupby("month")["n_events"].sum()
    ok = d[d["event_code"].str.fullmatch(r"(0[1-9]|1[0-9]|20)\d{1,2}")].assign(root=lambda x: x["event_code"].str[:2])
    m = ok.groupby(["month", "root"])["n_events"].sum().unstack(fill_value=0).reindex(columns=list(ROOTS), fill_value=0)
    months = pd.date_range(START.to_period("M").to_timestamp(), END.to_period("M").to_timestamp(), freq="MS")
    m = m.reindex(months, fill_value=0)
    total = total.reindex(months).fillna(0)
    if window > 1:
        m = m.rolling(window, center=True, min_periods=1).sum()
        total = total.rolling(window, center=True, min_periods=1).sum()
    return m.div(total.replace(0, float("nan")), axis=0) * 100


def compact(v, _):
    return f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"


def pct(v, _):
    return f"{v:g}%"


def plot(m, name, path, measure="counts", window=1):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharex=True, facecolor=SURFACE)
    for ax, (code, label) in zip(axes.flat, ROOTS.items()):
        ax.set_facecolor(SURFACE)
        ax.plot(m.index, m[code], color=SERIES, linewidth=1.6)
        ax.axvline(V2, color=MUTED, linewidth=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{code}  {label}", fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, labelsize=8.5)
        ax.yaxis.set_major_formatter(mt.FuncFormatter(pct if measure == "share" else compact))
        ax.yaxis.set_major_locator(mt.MaxNLocator(4, integer=False))
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5, month=1, day=1))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    if measure == "share":
        title = f"{name}: what share of recorded GDELT events were of each kind"
        span = "that month" if window == 1 else f"a rolling {window}-month window (centred on each month)"
        sub = (f"Each category's events as a percent of all events recorded for the country over {span}, Nov 2003 to Jul 2023. One panel per CAMEO root category, "
               "each with its own y-axis.\nThe vertical line marks the switch from GDELT 1.0 to 2.0 (2015-02-18); the mix of sources changed there, so shares can shift too.")
    else:
        title = f"{name}: how often each kind of GDELT event was recorded"
        sub = ("Mean events per day, averaged over each month, Nov 2003 to Jul 2023. One panel per CAMEO root category, each with its own y-axis.\n"
               "The vertical line marks the switch from GDELT 1.0 to 2.0 (2015-02-18); counts jump there, so compare shapes, not levels, across it.")
    fig.suptitle(title, fontsize=15, x=0.01, ha="left", color=INK, y=0.995)
    fig.text(0.01, 0.945, sub, fontsize=9.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    measure = "share"
    if "--measure" in args:
        i = args.index("--measure")
        measure = args[i + 1]
        args = args[:i] + args[i + 2:]
    window = 1
    if "--window" in args:
        i = args.index("--window")
        window = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    assert measure in ("share", "counts") and (window == 1 or measure == "share")
    codes = args or ["EGY", "THA", "UKR", "URY", "HKG"]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC, columns=["date", "country", "event_code", "n_events"])
    for c in codes:
        stem = ("monthly_share_of_events" + (f"_rolling{window}" if window > 1 else "")) if measure == "share" else "monthly_mean_events_per_day"
        m = monthly_shares(df, g.to_fips(c), window) if measure == "share" else monthly_means(df, g.to_fips(c))
        m.rename(columns=lambda k: f"{k} {ROOTS[k]}").rename_axis("month").round(4).to_csv(OUT / f"{stem}_{c}.csv")
        plot(m, NAMES.get(c, c), OUT / f"{stem}_{c}.png", measure, window)
        print(c, "ok", measure, f"(shares sum to {m.sum(axis=1).mean():.1f}% on average)" if measure == "share" else "")


if __name__ == "__main__":
    main()
