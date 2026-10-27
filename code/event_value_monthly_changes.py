"""First look at whether the SHAPE of a country's monthly GDELT event-share trajectory (not just its two wave-year
endpoints, see event_value_changes.py) relates to how survey answers changed, framed as correlation + permutation
test (not a trained predictive model -- with only 29 countries, a per-question predictive accuracy claim would not
be trustworthy; see STATUS.md).

    python code/event_value_inputs.py             # first: event levels, monthly shares and survey anchors
    python code/event_value_monthly_changes.py

For each country and each wave-to-wave transition (5->6, 6->7), the monthly event share (% of the country's events)
of each of the 20 CAMEO root categories, over all months from January of the first wave's survey year through
December of the second wave's survey year, is reduced to 4 fixed-size summary features (independent of the
transition's length in months, which varies 4-11 years across countries):
  level      mean monthly share over the window
  trend      slope of share vs. time, percentage points per year
  volatility standard deviation of the monthly share over the window
  shift      mean of the last 6 months minus mean of the first 6 months
That is 80 features (20 categories x 4 stats) per country-transition, each correlated separately (Pearson, across
countries) against the matching change in each survey question's anchor (% of respondents on the high side, see
event_value_inputs.anchors), as a per-year rate (same Y definition as event_value_changes.py, for comparability).
Significance: a permutation null moves each country's whole feature vector to another country (200 shuffles),
giving a family-wise p-value per slice (chance of a |r| this large anywhere in that slice).
The 6->7 transition spans the GDELT 1.0/2.0 switch (2015-02-18) for every country (see STATUS.md); 5->6 does not --
so this slice should be read as a replication check on 5->6, not a like-for-like second sample.

Output: gdelt_plots/event_value_monthly_changes/correlations_long.csv (every finite pair) and top_pairs.csv (top 25
per slice, by |r|). No plots -- this is a first look, not a dashboard.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_value_changes as C  # noqa: E402
from event_value_correlations import corr_vec, load  # noqa: E402
from event_value_inputs import COUNTRIES, OUT as BASE_OUT, ROOTS, event_shares_monthly  # noqa: E402

OUT = BASE_OUT.parent / "event_value_monthly_changes"
STATS = ["level", "trend", "volatility", "shift"]
SLICES = ["rate of change 5→6", "rate of change 6→7"]
N_PERM = 200
MIN_MONTHS = 24


def monthly_features(years):
    """(len(COUNTRIES), 2, 20, 4) array: feature[c, t, i, k] for transition t in (5->6, 6->7)."""
    share = event_shares_monthly()
    F = np.full((len(COUNTRIES), 2, 20, len(STATS)), np.nan)
    for ci, c in enumerate(COUNTRIES):
        if c not in share.index.get_level_values("country"):
            continue
        s = share.loc[c].sort_index()  # index: month timestamp, columns: ROOTS
        for t, (wa, wb) in enumerate([(5, 6), (6, 7)]):
            ya, yb = years.get((c, wa)), years.get((c, wb))
            if pd.isna(ya) or pd.isna(yb):
                continue
            win = s[(s.index >= pd.Timestamp(int(ya), 1, 1)) & (s.index < pd.Timestamp(int(yb) + 1, 1, 1))]
            if len(win) < MIN_MONTHS:
                continue
            x = np.arange(len(win), dtype=float) / 12.0  # years, for the trend's units (points per year)
            for i, root in enumerate(ROOTS):
                v = win[root].values
                F[ci, t, i, 0] = v.mean()
                F[ci, t, i, 1] = np.polyfit(x, v, 1)[0]
                F[ci, t, i, 2] = v.std()
                F[ci, t, i, 3] = v[-6:].mean() - v[:6].mean()
    return F


def slice_r(F2, DY, t):
    """R (20, 4, Q), N (Q,) for one transition's features F2 (C, 20, 4) against DY['r12' or 'r23'] (C, Q)."""
    key = "r12" if t == 0 else "r23"
    Y = DY[key]
    Q = Y.shape[1]
    R = np.full((20, len(STATS), Q), np.nan)
    N = np.zeros(Q, dtype=int)
    X = F2.reshape(F2.shape[0], -1)  # (C, 80)
    for j in range(Q):
        m = ~np.isnan(Y[:, j]) & ~np.isnan(X).all(axis=1)
        if m.sum() >= 15:
            r = corr_vec(X[m], Y[m, j])
            R[:, :, j] = r.reshape(20, len(STATS))
            N[j] = m.sum()
    return R, N


def null_max(F2, DY, t, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    mx = np.zeros(n_perm)
    for p in range(n_perm):
        Rp, _ = slice_r(F2[rng.permutation(F2.shape[0])], DY, t)
        mx[p] = np.nanmax(np.abs(Rp))
        print(f"\r  slice {SLICES[t]}: permutation {p + 1}/{n_perm}", end="", flush=True)
    print()
    return mx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X3, Y3, codes, years = load()
    yrs = np.full((len(COUNTRIES), 3), np.nan)
    for (c, w), y in years.items():
        yrs[COUNTRIES.index(c), w - 5] = y
    DY = C.rates(Y3, yrs)
    cats = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "question_categories.csv").fillna("").set_index("code")
    label = {q: cats.loc[q, "label"] if q in cats.index else q for q in codes}

    print("computing monthly features...")
    F = monthly_features(years)
    for t in range(2):
        have = np.isfinite(F[:, t]).any(axis=(1, 2)).sum()
        print(f"  {SLICES[t]}: {have}/{len(COUNTRIES)} countries with >= {MIN_MONTHS} months of GDELT data in window")

    rows = []
    for t in range(2):
        R, N = slice_r(F[:, t], DY, t)
        print(f"{SLICES[t]}: {int(np.isfinite(R).sum())} finite (feature, question) pairs, {int((N > 0).sum())} questions")
        mx = null_max(F[:, t], DY, t, N_PERM)
        for i, root in enumerate(ROOTS):
            for k, stat in enumerate(STATS):
                for j, code in enumerate(codes):
                    r = R[i, k, j]
                    if np.isfinite(r):
                        rows.append((SLICES[t], root, stat, code, label[code], code[0], r, N[j], (1 + (mx >= abs(r)).sum()) / (1 + N_PERM)))
    long = pd.DataFrame(rows, columns=["slice", "event", "feature", "question", "question_label", "section", "r", "n", "p_familywise"])
    long.to_csv(OUT / "correlations_long.csv", index=False)
    top = long.assign(absr=long["r"].abs()).sort_values(["slice", "absr"], ascending=[True, False]).groupby("slice").head(25).drop(columns="absr")
    top.to_csv(OUT / "top_pairs.csv", index=False)
    print()
    print(top.to_string(index=False, max_colwidth=40))


if __name__ == "__main__":
    main()
