"""Same analysis as event_value_correlations.py, but on CHANGE over time (a derivative) instead of levels.

    python code/event_value_inputs.py          # first (shared inputs)
    python code/event_value_changes.py

For each country, the change in an event category's level (percentage points of the country's events) and in a survey anchor (percentage points of
respondents on the high side) between two waves, divided by the years between the two survey years (so a rate: points per year). Six slices:
  rate 5->6, rate 6->7, rate 5->7     across countries
  consecutive rates pooled            5->6 and 6->7 together, each pair centred (pair effects removed)
  acceleration                        rate 6->7 minus rate 5->6, countries with all three waves
  consecutive rates pooled, Spearman  same as the pooled slice but on ranks within each pair (robust to outliers)
Permutation null: a country's whole event trajectory (levels and survey years) is moved to another country, 100 times; family-wise p as before.
Output in gdelt_plots/event_value_changes/: six PNG sheets, correlations_long.csv, top_pairs.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_value_correlations as ev  # noqa: E402
from event_value_correlations import (AQUA, BLUE, DIV, GRID, INK, MUTED, ORANGE, ROOT_NAMES, SECTIONS, SEQ, SURFACE, center_groups,  # noqa: E402
                                      corr_vec, load, sheet, short, style)
from event_value_inputs import COUNTRIES, OUT as BASE_OUT, ROOTS  # noqa: E402

OUT = BASE_OUT.parent / "event_value_changes"
SL = ["rate of change 5→6\n(points per year)", "rate of change 6→7\n(points per year)", "rate of change 5→7\n(points per year)",
      "consecutive rates pooled\n(pair effects removed)", "acceleration\n(rate 6→7 minus rate 5→6)",
      "consecutive rates pooled,\nrank correlation (Spearman)"]
N_PERM = 100


def rates(V3, yrs):
    """Dict of change-per-year arrays for a (C, 3, K) array and (C, 3) survey years."""
    g = lambda a, b: (V3[:, b] - V3[:, a]) / (yrs[:, b] - yrs[:, a])[:, None]   # noqa: E731
    return {"r12": g(0, 1), "r23": g(1, 2), "r13": g(0, 2)}


def slices(DX, DY):
    """R (6, 20, Q), N (6, Q) from the change-per-year dicts of X (C, 20) and Y (C, Q)."""
    Q = DY["r12"].shape[1]
    R = np.full((6, 20, Q), np.nan)
    N = np.zeros((6, Q), dtype=int)
    acc_x = DX["r23"] - DX["r12"]
    for j in range(Q):
        for s, key in enumerate(("r12", "r23", "r13")):
            m = ~np.isnan(DY[key][:, j])
            if m.sum() >= 20:
                R[s, :, j] = corr_vec(DX[key][m], DY[key][m, j]); N[s, j] = m.sum()
        rows = []
        for p, key in enumerate(("r12", "r23")):
            m = ~np.isnan(DY[key][:, j])
            if m.sum() >= 10:
                rows.append((DX[key][m], DY[key][m, j], np.full(m.sum(), p)))
        if rows and sum(len(r[1]) for r in rows) >= 25:
            xs, ys, gs = (np.concatenate([r[k] for r in rows]) for k in range(3))
            xc, yc = center_groups(xs, ys, gs)
            R[3, :, j] = corr_vec(xc, yc); N[3, j] = len(yc)
            xr = np.concatenate([np.apply_along_axis(rankdata, 0, r[0]) for r in rows])
            yr = np.concatenate([rankdata(r[1]) for r in rows])
            xc, yc = center_groups(xr, yr, gs)
            R[5, :, j] = corr_vec(xc, yc); N[5, j] = len(yc)
        acc_y = DY["r23"][:, j] - DY["r12"][:, j]
        m = ~np.isnan(acc_y)
        if m.sum() >= 12:
            R[4, :, j] = corr_vec(acc_x[m], acc_y[m]); N[4, j] = m.sum()
    return R, N


def run_null(X3, yrs, DY, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    mx = np.zeros((n_perm, 6))
    allabs = []
    for p in range(n_perm):
        perm = rng.permutation(X3.shape[0])
        Rp, _ = slices(rates(X3[perm], yrs[perm]), DY)
        mx[p] = np.nanmax(np.abs(Rp).reshape(6, -1), axis=1)
        allabs.append(np.abs(Rp[:, :, ::5]))
        print(f"\rpermutation {p + 1}/{n_perm}", end="", flush=True)
    print()
    return mx, np.stack(allabs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X3, Y3, codes, years = load()
    yrs = np.full((len(COUNTRIES), 3), np.nan)
    for (c, w), y in years.items():
        yrs[COUNTRIES.index(c), w - 5] = y
    cats = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "question_categories.csv").fillna("").set_index("code")
    label = {q: cats.loc[q, "label"] if q in cats.index else q for q in codes}
    DX, DY = rates(X3, yrs), rates(Y3, yrs)
    R, N = slices(DX, DY)
    print("pairs per slice:", [int(np.isfinite(R[s]).sum()) for s in range(6)], "| questions:", [int((N[s] > 0).sum()) for s in range(6)])
    mx, nullabs = run_null(X3, yrs, DY, N_PERM)

    rows = []
    for s in range(6):
        for i in range(20):
            for j in range(len(codes)):
                r = R[s, i, j]
                if np.isfinite(r):
                    rows.append((SL[s].replace("\n", " "), ROOTS[i], ROOT_NAMES[ROOTS[i]], codes[j], label[codes[j]], codes[j][0], r, N[s, j],
                                 (1 + (mx[:, s] >= abs(r)).sum()) / (1 + N_PERM)))
    long = pd.DataFrame(rows, columns=["slice", "event", "event_name", "question", "question_label", "section", "r", "n", "p_familywise"])
    long.to_csv(OUT / "correlations_long.csv", index=False)
    long.assign(absr=long["r"].abs()).sort_values(["slice", "absr"], ascending=[True, False]).groupby("slice").head(25).drop(columns="absr").to_csv(OUT / "top_pairs.csv", index=False)
    name = lambda s: SL[s].replace("\n", " ")   # noqa: E731

    # sheet 1: heatmaps by section
    fig, axes = sheet(2, 3, (20, 12.5), "Where changes in event shares and changes in survey answers go together",
                      "Mean absolute correlation between the change per year in each event category's share and the change per year in the survey questions of each WVS section. "
                      "Darker = stronger. Sections are WVS code prefixes (names approximate).")
    secs = list(SECTIONS)
    mats, vmax = [], 0
    for s in range(6):
        M = np.full((20, len(secs)), np.nan)
        for k, sec in enumerate(secs):
            idx = [j for j, q in enumerate(codes) if q[0] == sec and np.isfinite(R[s][:, j]).any()]
            if idx:
                M[:, k] = np.nanmean(np.abs(R[s][:, idx]), axis=1)
        mats.append(M); vmax = max(vmax, np.nanmax(M))
    for s, ax in enumerate(axes.flat):
        style(ax)
        ax.imshow(mats[s], cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(secs)), [SECTIONS[x] for x in secs], rotation=45, ha="right")
        ax.set_yticks(range(20), [f"{r} {ROOT_NAMES[r]}" for r in ROOTS])
        ax.set_title(f"{name(s)}   ({int((N[s] > 0).sum())} questions)", fontsize=9.5, loc="left")
        for (i, k), v in np.ndenumerate(mats[s]):
            if np.isfinite(v):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 * vmax else INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet1_heatmaps_event_by_question_section.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # sheet 2: strongest pairs
    fig, axes = sheet(3, 2, (20, 15), "The strongest correlations between changes, in each slice",
                      "Top 12 (event category, question) pairs by |r|; both sides are change per year in percentage points. Bar = correlation (blue positive, red negative); n = countries or country-pairs; "
                      "* = beats chance across the whole slice (family-wise permutation p < 0.05).")
    for s, ax in enumerate(axes.flat):
        style(ax)
        sub = long[long["slice"] == name(s)].assign(absr=lambda d: d["r"].abs()).sort_values("absr", ascending=False).head(12).iloc[::-1]
        ax.barh(range(len(sub)), sub["r"], color=[BLUE if r > 0 else ev.RED for r in sub["r"]], height=0.6)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_yticks(range(len(sub)), [f"{short(q, 52)}  ×  {e} {n}  (n={k}){' *' if p < 0.05 else ''}" for q, e, n, k, p in
                                         zip(sub["question_label"], sub["event"], sub["event_name"], sub["n"], sub["p_familywise"])], fontsize=7.5)
        ax.set_xlim(-1, 1)
        ax.set_title(name(s), fontsize=10.5, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet2_strongest_pairs.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # sheet 3: observed vs chance
    fig, axes = sheet(2, 3, (20, 10), "Are the correlations between changes bigger than chance?",
                      f"Distribution of |r| over all pairs in each slice (blue) against the same slice with each country's event trajectory moved to a random country ({N_PERM} shuffles, outline). "
                      "Vertical line = largest |r| seen.")
    for s, ax in enumerate(axes.flat):
        style(ax)
        obs = np.abs(R[s][np.isfinite(R[s])])
        nul = nullabs[:, s][np.isfinite(nullabs[:, s])]
        bins = np.linspace(0, 1, 41)
        ax.hist(obs, bins=bins, color=BLUE, alpha=0.85, density=True)
        ax.hist(nul, bins=bins, histtype="step", color=INK, linewidth=1.4, density=True)
        ax.axvline(obs.max(), color=ORANGE, linewidth=1.4)
        ax.set_title(f"{name(s)}   (max |r| {obs.max():.2f}; null max typically {np.median(mx[:, s]):.2f})", fontsize=9, loc="left")
        ax.set_xlabel("|r|", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "sheet3_observed_vs_chance.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # sheet 4: scatter of the strongest pooled consecutive-change pairs
    qi = {q: j for j, q in enumerate(codes)}
    top = long[long["slice"] == name(3)].assign(absr=lambda d: d["r"].abs()).sort_values("absr", ascending=False).head(12)
    fig, axes = sheet(3, 4, (20, 14), "The strongest pooled pairs, one dot per country and wave interval",
                      "Change per year in the event share (x, points) against change per year in the survey anchor (y, points). Blue = wave 5→6, orange = wave 6→7. Labels = country codes.")
    for ax, row in zip(axes.flat, top.itertuples()):
        style(ax)
        i, j = ROOTS.index(row.event), qi[row.question]
        for key, col, lab in (("r12", BLUE, "5→6"), ("r23", ORANGE, "6→7")):
            m = ~np.isnan(DY[key][:, j])
            ax.scatter(DX[key][m, i], DY[key][m, j], s=26, color=col, edgecolor=SURFACE, linewidth=1, label=lab, zorder=3)
            for c in np.where(m)[0]:
                ax.annotate(COUNTRIES[c], (DX[key][c, i], DY[key][c, j]), fontsize=5.5, color=MUTED, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
        ax.set_title(f"{row.event} {row.event_name}  ×  {short(row.question_label, 44)}\nr = {row.r:+.2f} (pair effects removed), n = {row.n}", fontsize=8.5, loc="left")
        ax.set_xlabel("event share change, points/yr", fontsize=8)
        ax.set_ylabel("answer change, points/yr", fontsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet4_scatter_strongest_pooled.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # sheet 5: country panels for the strongest pooled pair (change in each interval, both sides in units of their own spread)
    best = top.iloc[0]
    i, j = ROOTS.index(best["event"]), qi[best["question"]]
    fig, axes = sheet(6, 5, (20, 16), f"Country by country, strongest pooled pair: {best['event']} {best['event_name']}  ×  {short(best['question_label'], 60)}",
                      f"Change per year in each wave interval (r = {best['r']:+.2f}, n = {int(best['n'])}), each series divided by its own standard deviation across countries and intervals: "
                      "blue = event share, orange = survey anchor. Missing bars = question not asked in both waves.")
    sx = np.nanstd(np.concatenate([DX["r12"][:, i], DX["r23"][:, i]]))
    sy = np.nanstd(np.concatenate([DY["r12"][:, j], DY["r23"][:, j]]))
    for ax, c in zip(axes.flat, range(len(COUNTRIES))):
        style(ax)
        ax.set_title(COUNTRIES[c], fontsize=9, loc="left")
        for k, key in enumerate(("r12", "r23")):
            ax.bar(k - 0.18, DX[key][c, i] / sx, width=0.34, color=BLUE)
            v = DY[key][c, j] / sy
            if np.isfinite(v):
                ax.bar(k + 0.18, v, width=0.34, color=ORANGE)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_xticks([0, 1], ["5→6", "6→7"], fontsize=8)
        ax.set_ylim(-3, 3)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(COUNTRIES):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet5_country_changes_strongest_pair.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # sheet 6: the event change rates themselves
    fig, axes = sheet(1, 3, (22, 10), "Event changes: change per year in each category's share of a country's events, between waves",
                      "Rows = the 29 countries, columns = CAMEO root categories, cells = percentage points per year between the two survey years (red = fell, blue = rose). "
                      "6→7 and 5→7 span the GDELT 1.0 to 2.0 switch (2015-02-18), which shifts shares for every country, so compare countries within an interval.")
    lim = np.nanpercentile(np.abs(np.concatenate([DX[k].ravel() for k in ("r12", "r23", "r13")])), 98)
    for ax, (key, lab) in zip(axes.flat, (("r12", "wave 5→6"), ("r23", "wave 6→7"), ("r13", "wave 5→7"))):
        style(ax)
        ax.imshow(DX[key], cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(20), ROOTS, fontsize=8)
        ax.set_yticks(range(len(COUNTRIES)), COUNTRIES, fontsize=8)
        ax.set_title(lab, fontsize=11, loc="left")
        ax.set_xlabel("CAMEO root category", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "sheet6_event_change_rates_by_country.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)
    print("done:", sorted(p.name for p in OUT.glob("sheet*.png")))
    print(long.assign(absr=long["r"].abs()).sort_values(["slice", "absr"], ascending=[True, False]).groupby("slice").head(3)
          .drop(columns="absr").to_string(index=False, max_colwidth=40))


if __name__ == "__main__":
    main()
