"""Correlate yearly GDELT event shares with survey answers across countries, waves, and several ways of slicing.

    python code/event_value_inputs.py          # first: event levels and survey anchors
    python code/event_value_correlations.py    # then: correlations, a permutation null, and the plot sheets

X = a country's event level (mean monthly share of one CAMEO root category, in the wave's survey year); Y = the survey anchor (percent of
respondents on the high side of a question; see event_value_inputs.py). Pearson correlations for every (root category, question) pair, in six slices:
  wave 5 / wave 6 / wave 7   across the countries in that wave
  waves pooled               wave effects removed (X and Y centred within each wave) -> compares countries, net of the GDELT 1.0/2.0 shift
  within country             country and wave effects removed (needs the country in all 3 waves) -> co-movement over time inside countries
  changes between waves      wave-to-wave changes (5->6, 6->7), centred within each pair
A permutation null (country event levels shuffled across countries, 100 times) gives, for every pair, the chance of a maximum |r| at least that large
anywhere in the slice (family-wise p). Output: correlations_long.csv, top_pairs.csv and six PNG sheets in gdelt_plots/event_value_correlations/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_value_inputs import COUNTRIES, OUT, ROOTS  # noqa: E402

ROOT_NAMES = {"01": "Public statement", "02": "Appeal", "03": "Intent to cooperate", "04": "Consult", "05": "Diplomatic coop.",
              "06": "Material coop.", "07": "Provide aid", "08": "Yield", "09": "Investigate", "10": "Demand", "11": "Disapprove",
              "12": "Reject", "13": "Threaten", "14": "Protest", "15": "Force posture", "16": "Reduce relations", "17": "Coerce",
              "18": "Assault", "19": "Fight", "20": "Mass violence"}
SECTIONS = {"A": "A life", "B": "B environment", "C": "C work", "D": "D family", "E": "E politics", "F": "F religion/morals",
            "G": "G national identity", "H": "H security", "I": "I science"}     # WVS question-code prefixes; names approximate
SLICES = ["wave 5", "wave 6", "wave 7", "waves pooled\n(wave effects removed)", "within country\n(country + wave effects removed)",
          "changes between waves\n(pair effects removed)"]
N_PERM = 100
# palette tokens (dataviz reference palette, light surface)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
SEQ = mc.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = mc.LinearSegmentedColormap.from_list("div", [RED, GRAY, BLUE])


# ------------------------------------------------------------------ data

def load():
    lv = pd.read_csv(OUT / "event_levels.csv", index_col=[0, 1])
    a = pd.read_parquet(OUT / "anchors.parquet")
    codes = sorted(a["code"].unique())
    years = a.groupby(["country", "wave"])["year"].first()
    X3 = np.full((len(COUNTRIES), 3, 20), np.nan)
    Y3 = np.full((len(COUNTRIES), 3, len(codes)), np.nan)
    ci = {c: i for i, c in enumerate(COUNTRIES)}
    qi = {q: j for j, q in enumerate(codes)}
    for (c, w), y in years.items():
        X3[ci[c], w - 5] = lv.loc[(c, y)].values
    for r in a.itertuples():
        Y3[ci[r.country], r.wave - 5, qi[r.code]] = r.share_high
    return X3, Y3, codes, years


# ------------------------------------------------------------------ correlations

def corr_vec(xs, ys):
    yc = ys - ys.mean()
    xc = xs - xs.mean(0)
    sy, sx = np.sqrt((yc ** 2).sum()), np.sqrt((xc ** 2).sum(0))
    r = np.full(xs.shape[1], np.nan)
    ok = (sx > 1e-12) & (sy > 1e-12)
    r[ok] = (xc[:, ok].T @ yc) / (sx[ok] * sy)
    return r


def center_groups(xs, ys, groups):
    xs, ys = xs.copy(), ys.copy()
    for g in np.unique(groups):
        m = groups == g
        xs[m] -= xs[m].mean(0)
        ys[m] -= ys[m].mean()
    return xs, ys


def slice_r(X3, Y3):
    """R (6, 20, Q) and N (6, Q) for the six slices."""
    Q = Y3.shape[2]
    R = np.full((6, 20, Q), np.nan)
    N = np.zeros((6, Q), dtype=int)
    for j in range(Q):
        y = Y3[:, :, j]
        for w in range(3):                                           # slices 0-2: one wave
            m = ~np.isnan(y[:, w])
            if m.sum() >= 20:
                R[w, :, j] = corr_vec(X3[m, w], y[m, w]); N[w, j] = m.sum()
        # slice 3: pooled, centred within wave (waves with >= 10 countries, at least two such waves)
        rows = [(X3[m, w], y[m, w], np.full(m.sum(), w)) for w in range(3) if (m := ~np.isnan(y[:, w])).sum() >= 10]
        if len(rows) >= 2 and sum(len(r[1]) for r in rows) >= 40:
            xs, ys, gs = (np.concatenate([r[k] for r in rows]) for k in range(3))
            xs, ys = center_groups(xs, ys, gs)
            R[3, :, j] = corr_vec(xs, ys); N[3, j] = len(ys)
        # slice 4: countries observed in all three waves, country and wave effects removed
        full = ~np.isnan(y).any(axis=1)
        if full.sum() >= 12:
            xx, yy = X3[full], y[full]                              # (m, 3, 20), (m, 3)
            xx = xx - xx.mean(1, keepdims=True) - xx.mean(0, keepdims=True) + xx.mean((0, 1), keepdims=True)
            yy = yy - yy.mean(1, keepdims=True) - yy.mean(0, keepdims=True) + yy.mean()
            R[4, :, j] = corr_vec(xx.reshape(-1, 20), yy.ravel()); N[4, j] = yy.size
        # slice 5: changes between consecutive waves, centred within each pair
        rows = []
        for p, (a, b) in enumerate([(0, 1), (1, 2)]):
            m = ~np.isnan(y[:, a]) & ~np.isnan(y[:, b])
            if m.sum() >= 10:
                rows.append((X3[m, b] - X3[m, a], y[m, b] - y[m, a], np.full(m.sum(), p)))
        if rows and sum(len(r[1]) for r in rows) >= 25:
            xs, ys, gs = (np.concatenate([r[k] for r in rows]) for k in range(3))
            xs, ys = center_groups(xs, ys, gs)
            R[5, :, j] = corr_vec(xs, ys); N[5, j] = len(ys)
    return R, N


def null_max(X3, Y3, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    mx = np.zeros((n_perm, 6))
    allabs = []
    for p in range(n_perm):
        Rp, _ = slice_r(X3[rng.permutation(X3.shape[0])], Y3)     # whole country trajectories of X moved to another country
        mx[p] = np.nanmax(np.abs(Rp).reshape(6, -1), axis=1)
        allabs.append(np.abs(Rp[:, :, ::5]))                       # a subsample of the null, for the histogram
        print(f"\rpermutation {p + 1}/{n_perm}", end="", flush=True)
    print()
    return mx, np.stack(allabs)


# ------------------------------------------------------------------ plots

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def sheet(nrows, ncols, size, title, sub):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED})
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, squeeze=False)
    fig.suptitle(title, fontsize=16, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.962, sub, fontsize=10, color=MUTED, ha="left", va="top")
    return fig, axes


def short(s, n=64):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    X3, Y3, codes, years = load()
    cats = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "question_categories.csv").fillna("").set_index("code")
    label = {q: cats.loc[q, "label"] if q in cats.index else q for q in codes}
    R, N = slice_r(X3, Y3)
    print("pairs per slice:", [int(np.isfinite(R[s]).sum()) for s in range(6)], "| questions per slice:", [int((N[s] > 0).sum()) for s in range(6)])
    mx, nullabs = null_max(X3, Y3, N_PERM)

    rows = []
    for s in range(6):
        for i in range(20):
            for j in range(len(codes)):
                r = R[s, i, j]
                if np.isfinite(r):
                    rows.append((SLICES[s].replace("\n", " "), ROOTS[i], ROOT_NAMES[ROOTS[i]], codes[j], label[codes[j]], codes[j][0], r, N[s, j],
                                 (1 + (mx[:, s] >= abs(r)).sum()) / (1 + N_PERM)))
    long = pd.DataFrame(rows, columns=["slice", "event", "event_name", "question", "question_label", "section", "r", "n", "p_familywise"])
    long.to_csv(OUT / "correlations_long.csv", index=False)
    top = long.assign(absr=long["r"].abs()).sort_values(["slice", "absr"], ascending=[True, False]).groupby("slice").head(25).drop(columns="absr")
    top.to_csv(OUT / "top_pairs.csv", index=False)

    # ---- sheet 1: where the correlations are (mean |r| by event x question section)
    fig, axes = sheet(2, 3, (20, 12.5), "Where event shares and survey answers go together",
                      "Mean absolute correlation between each event category's share and the survey questions in each WVS section (cell = average over the questions). "
                      "Darker = stronger. Sections are the WVS code prefixes (names approximate). Six ways of slicing, see the title of each panel.")
    secs = list(SECTIONS)
    vmax = 0
    mats = []
    for s in range(6):
        M = np.full((20, len(secs)), np.nan)
        for k, sec in enumerate(secs):
            idx = [j for j, q in enumerate(codes) if q[0] == sec]
            if idx:
                M[:, k] = np.nanmean(np.abs(R[s][:, idx]), axis=1) if np.isfinite(R[s][:, idx]).any() else np.nan
        mats.append(M); vmax = max(vmax, np.nanmax(M))
    for s, ax in enumerate(axes.flat):
        style(ax)
        im = ax.imshow(mats[s], cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(secs)), [SECTIONS[x] for x in secs], rotation=45, ha="right")
        ax.set_yticks(range(20), [f"{r} {ROOT_NAMES[r]}" for r in ROOTS])
        ax.set_title(f"{SLICES[s]}   ({int((N[s] > 0).sum())} questions)", fontsize=10, loc="left")
        for (i, k), v in np.ndenumerate(mats[s]):
            if np.isfinite(v):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 * vmax else INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet1_heatmaps_event_by_question_section.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # ---- sheet 2: strongest pairs per slice
    fig, axes = sheet(3, 2, (20, 15), "The strongest individual correlations in each slice",
                      "Top 12 (event category, question) pairs by |r|. Bar = correlation (blue positive, red negative); n = countries or country-waves used; "
                      "* = beats chance across the whole slice (family-wise permutation p < 0.05). Question = share on the high side of the scale.")
    for s, ax in enumerate(axes.flat):
        style(ax)
        sub = long[long["slice"] == SLICES[s].replace("\n", " ")].assign(absr=lambda d: d["r"].abs()).sort_values("absr", ascending=False).head(12).iloc[::-1]
        cols = [BLUE if r > 0 else RED for r in sub["r"]]
        ax.barh(range(len(sub)), sub["r"], color=cols, height=0.6)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_yticks(range(len(sub)), [f"{short(q, 52)}  ×  {e} {n}  (n={k}){' *' if p < 0.05 else ''}" for q, e, n, k, p in
                                         zip(sub["question_label"], sub["event"], sub["event_name"], sub["n"], sub["p_familywise"])], fontsize=7.5)
        ax.set_xlim(-1, 1)
        ax.set_title(SLICES[s].replace("\n", " "), fontsize=10.5, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet2_strongest_pairs.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # ---- sheet 3: observed vs chance
    fig, axes = sheet(2, 3, (20, 10), "Are the correlations bigger than chance?",
                      f"Distribution of |r| over all (event, question) pairs in each slice (blue) against the same slice after shuffling which country each event profile belongs to "
                      f"({N_PERM} shuffles, outline). A real signal would put the blue to the right of the outline. Vertical line = largest |r| seen.")
    for s, ax in enumerate(axes.flat):
        style(ax)
        obs = np.abs(R[s][np.isfinite(R[s])])
        nul = nullabs[:, s][np.isfinite(nullabs[:, s])]
        bins = np.linspace(0, 1, 41)
        ax.hist(obs, bins=bins, color=BLUE, alpha=0.85, density=True)
        ax.hist(nul, bins=bins, histtype="step", color=INK, linewidth=1.4, density=True)
        ax.axvline(obs.max(), color=ORANGE, linewidth=1.4)
        ax.set_title(f"{SLICES[s].replace(chr(10), ' ')}   (max |r| {obs.max():.2f}; null max typically {np.median(mx[:, s]):.2f})", fontsize=9.5, loc="left")
        ax.set_xlabel("|r|", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "sheet3_observed_vs_chance.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # ---- sheet 4: scatter of the strongest pooled pairs (raw values; colour = wave)
    s = 3
    sub = long[long["slice"] == SLICES[s].replace("\n", " ")].assign(absr=lambda d: d["r"].abs()).sort_values("absr", ascending=False).head(12)
    fig, axes = sheet(3, 4, (20, 14), "The strongest pooled pairs, one dot per country and wave",
                      "Event share (x, mean monthly % of the country's events in the survey year) against the survey anchor (y, % on the high side). Ranking used wave effects removed; "
                      "dots show raw values. Colour = wave. Labels = country codes.")
    qi = {q: j for j, q in enumerate(codes)}
    for ax, row in zip(axes.flat, sub.itertuples()):
        style(ax)
        i, j = ROOTS.index(row.event), qi[row.question]
        for w, col in zip(range(3), (BLUE, ORANGE, AQUA)):
            m = ~np.isnan(Y3[:, w, j])
            ax.scatter(X3[m, w, i], Y3[m, w, j], s=26, color=col, edgecolor=SURFACE, linewidth=1, label=f"wave {w + 5}", zorder=3)
            for c in np.where(m)[0]:
                ax.annotate(COUNTRIES[c], (X3[c, w, i], Y3[c, w, j]), fontsize=5.5, color=MUTED, xytext=(2, 2), textcoords="offset points")
        ax.set_title(f"{row.event} {row.event_name}  ×  {short(row.question_label, 44)}\nr = {row.r:+.2f} (wave effects removed), n = {row.n}", fontsize=8.5, loc="left")
        ax.set_xlabel("event share, %", fontsize=8)
        ax.set_ylabel("% on high side", fontsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet4_scatter_strongest_pooled.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # ---- sheet 5: country trajectories for the strongest within-country pair
    s = 4
    best = long[long["slice"] == SLICES[s].replace("\n", " ")].assign(absr=lambda d: d["r"].abs()).sort_values("absr", ascending=False).iloc[0]
    i, j = ROOTS.index(best["event"]), qi[best["question"]]
    fig, axes = sheet(6, 5, (20, 16), f"Within-country co-movement for the strongest pair: {best['event']} {best['event_name']}  ×  {short(best['question_label'], 70)}",
                      f"Each panel is one country across waves 5, 6, 7 (r = {best['r']:+.2f}, within-country slice, n = {int(best['n'])}). Both series are rescaled to 0 (lowest of the three waves) to 1 (highest) so shapes can be compared: "
                      "blue = event share, orange = survey anchor. Panels without 3 waves of answers are left blank.")
    for ax, c in zip(axes.flat, range(len(COUNTRIES))):
        style(ax)
        ax.set_title(COUNTRIES[c], fontsize=9, loc="left")
        y = Y3[c, :, j]
        if np.isnan(y).any():
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "no answers in all 3 waves", ha="center", va="center", fontsize=7.5, color=MUTED, transform=ax.transAxes)
            continue
        x = X3[c, :, i]
        sc = lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.full_like(v, 0.5)   # noqa: E731
        ax.plot([5, 6, 7], sc(x), color=BLUE, linewidth=2, marker="o", markersize=4)
        ax.plot([5, 6, 7], sc(y), color=ORANGE, linewidth=2, marker="o", markersize=4)
        ax.set_xticks([5, 6, 7]); ax.set_yticks([0, 1]); ax.set_ylim(-0.1, 1.1)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    for ax in list(axes.flat)[len(COUNTRIES):]:
        ax.axis("off")
    for ax in list(axes.flat)[len(COUNTRIES):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "sheet5_within_country_trajectories.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)

    # ---- sheet 6: the event levels themselves
    fig, axes = sheet(1, 3, (22, 10), "Event levels: mean share of each country's events, in each wave's survey year",
                      "Rows = the 29 countries, columns = CAMEO root categories, cells = percent of the country's recorded events (mean of the monthly shares in the survey year). "
                      "Waves 5 and 6 come from GDELT 1.0, wave 7 from 2.0, so compare countries within a wave.")
    for w, ax in enumerate(axes.flat):
        style(ax)
        im = ax.imshow(X3[:, w, :], cmap=SEQ, aspect="auto", vmin=0, vmax=np.nanpercentile(X3, 99))
        ax.set_xticks(range(20), ROOTS, fontsize=8)
        ax.set_yticks(range(len(COUNTRIES)), [f"{c}  {years.get((c, w + 5), '')}" for c in COUNTRIES], fontsize=8)
        ax.set_title(f"wave {w + 5}", fontsize=11, loc="left")
        ax.set_xlabel("CAMEO root category", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "sheet6_event_levels_by_country_and_wave.png", dpi=105, facecolor=SURFACE)
    plt.close(fig)
    print("done:", sorted(p.name for p in OUT.glob("sheet*.png")))
    print(top.groupby("slice").head(3).to_string(index=False, max_colwidth=45))


if __name__ == "__main__":
    main()
