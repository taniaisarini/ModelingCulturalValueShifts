"""Predict WVS value shifts from the temporal SHAPE of a country's GDELT event-share trajectory (functional PCA +
per-question elastic net), replacing the univariate correlation tests in event_value_monthly_changes.py.

    python code/event_value_inputs.py          # first: monthly event shares and survey anchors (shared inputs)
    python code/event_value_fpca.py --dry-run   # 3 questions, prints array shapes + fPCA plots, fits nothing else

STEP 1  Fieldwork month per country-wave = median(S022, year*100+month of interview) across respondents. S022 has
        100% coverage (unlike the exact interview date S012, valid for only 46/87 country-waves); see
        fieldwork_months(). Cached in gdelt_plots/event_value_fpca/fieldwork_months.csv.
STEP 2  Per (country, category, transition): log(share + 0.1) each month of the interval [earlier wave's fieldwork
        month, month before later wave's fieldwork month], minus the mean of log(share + 0.1) over the 24 months
        before the earlier wave's fieldwork month (the baseline). Resampled to 12 points (np.array_split into 12
        chunks, mean of each) so every country has a length-12 curve regardless of the interval's length (48-132
        months across countries). See log_centered_curves() -> (29 countries, 20 categories, 12 points).
STEP 3  Functional PCA: per (category, transition), PCA across the 29 countries' 12-point curves (sklearn centers
        across countries internally). First 3 components kept -> 60 features/country; a 20-feature version keeps
        only component 1 per category. PCA is fit on X only, never Y, so fitting on all 29 countries (not just a
        training fold) does not leak Y information -- see fpca_per_category().
STEP 4  (not run by --dry-run beyond the 3 example questions) Per question with >= 20 countries: ElasticNetCV
        (l1_ratio in [0.2, 0.5, 0.8, 1.0], its own inner CV for alpha) on standardized fPCA scores + the country's
        earlier-wave mean on that question, leave-one-country-out CV; scaler and ElasticNetCV are refit inside each
        LOCO fold (on the other 28 countries only) to avoid leaking the held-out country into standardization.
        Reported against a mean-only and an earlier-wave-mean-only baseline. See fit_question().
STEPS 5-6 (permutation null, 5->6 -> 6->7 replication) -- not implemented yet; come back to this after the dry run
        is reviewed, since they reuse step 4's fit_question() as their inner loop.

Y = change in rescaled mean response (event_value_inputs.rescaled_means(), 0-1 per question) between waves, so
changes are comparable across questions regardless of each question's native scale.
"""
import os
# must be set before numpy/sklearn touch BLAS: the 2026-10-27 full run (16-worker ProcessPoolExecutor) took 4495s
# for 88 questions x 200 perms (~4.1s/fit vs. 1.6s measured single-process) -- each worker's BLAS calls were
# spawning their own threads on top of the 16 already-parallel processes, oversubscribing the allocated CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import threadpoolctl  # noqa: E402 -- belt-and-suspenders: env vars above can be set too late in a forked worker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import ElasticNetCV, LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_baseline import DATA_PATH  # noqa: E402
from event_value_correlations import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, sheet, style  # noqa: E402
from event_value_inputs import COUNTRIES, OUT as BASE_OUT, ROOTS, event_shares_monthly, rescaled_means  # noqa: E402

OUT = BASE_OUT.parent / "event_value_fpca"
BASELINE_MONTHS = 24
N_POINTS = 12
TRANSITIONS = [(5, 6), (6, 7)]


# ------------------------------------------------------------------ step 1: fieldwork months

def fieldwork_months():
    """country x wave -> Timestamp (1st of month), median interview month from S022 (= year*100 + month)."""
    cache = OUT / "fieldwork_months.csv"
    if cache.exists():
        tab = pd.read_csv(cache, index_col=0)
        tab.columns = tab.columns.astype(int)
        return tab.apply(pd.to_datetime)
    parts = []
    for chunk in pd.read_csv(DATA_PATH, usecols=["S002VS", "COUNTRY_ALPHA", "S022"], chunksize=200_000, low_memory=False):
        parts.append(chunk[chunk["COUNTRY_ALPHA"].isin(COUNTRIES) & chunk["S002VS"].isin([5, 6, 7])])
    df = pd.concat(parts, ignore_index=True)
    s22 = pd.to_numeric(df["S022"], errors="coerce")
    ok = (s22 > 190001) & (s22 < 300000)
    df = df[ok].copy()
    y, m = (s22[ok] // 100).astype(int), (s22[ok] % 100).astype(int)
    df = df[m.between(1, 12)]
    df["ord"] = y * 12 + m

    def to_ts(o):
        o = int(round(o))
        yy, mm = divmod(o - 1, 12)
        return pd.Timestamp(year=yy, month=mm + 1, day=1)

    med = df.groupby(["COUNTRY_ALPHA", "S002VS"])["ord"].median()
    tab = med.apply(to_ts).unstack().reindex(COUNTRIES)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(cache)
    return tab


# ------------------------------------------------------------------ step 2: log-centered, baseline-relative curves

def log_centered_curves(share, fw, wa, wb):
    """(29, 20, 12) array for transition wa->wb; months[ci] = (n baseline months, n interval months) for diagnostics."""
    curves = np.full((len(COUNTRIES), 20, N_POINTS), np.nan)
    months = {}
    for ci, c in enumerate(COUNTRIES):
        if c not in fw.index or pd.isna(fw.loc[c, wa]) or pd.isna(fw.loc[c, wb]):
            continue
        if c not in share.index.get_level_values("country"):
            continue
        s = share.loc[c].sort_index()
        bmonths = pd.date_range(fw.loc[c, wa] - pd.DateOffset(months=BASELINE_MONTHS), fw.loc[c, wa] - pd.DateOffset(months=1), freq="MS")
        imonths = pd.date_range(fw.loc[c, wa], fw.loc[c, wb] - pd.DateOffset(months=1), freq="MS")
        bl, iv = s.reindex(bmonths), s.reindex(imonths)
        months[c] = (bl[ROOTS[0]].notna().sum(), iv[ROOTS[0]].notna().sum())
        if months[c][1] < N_POINTS:
            continue
        for i, root in enumerate(ROOTS):
            baseline_mean = np.nanmean(np.log(bl[root].values + 0.1))
            li = np.log(iv[root].values + 0.1) - baseline_mean
            curves[ci, i] = [np.nanmean(ch) for ch in np.array_split(li, N_POINTS)]
    return curves, months


# ------------------------------------------------------------------ step 3: functional PCA

def fpca_per_category(curves, n_components=3):
    """curves (C, 20, 12) -> scores (C, 20, n_components), fitted PCA objects per category (for later projection)."""
    C, K, T = curves.shape
    scores = np.full((C, K, n_components), np.nan)
    pcas = {}
    for i in range(K):
        X = curves[:, i, :]
        ok = ~np.isnan(X).any(axis=1)
        if ok.sum() < n_components + 1:
            continue
        pca = PCA(n_components=n_components).fit(X[ok])
        scores[ok, i, :] = pca.transform(X[ok])
        pcas[i] = pca
    return scores, pcas


def project(curves, pcas, n_components=3):
    """Project curves (C, 20, 12) onto already-fitted PCA objects (e.g. the 5->6 ones, for step 6)."""
    C, K, T = curves.shape
    scores = np.full((C, K, n_components), np.nan)
    for i, pca in pcas.items():
        X = curves[:, i, :]
        ok = ~np.isnan(X).any(axis=1)
        scores[ok, i, :] = pca.transform(X[ok])
    return scores


# ------------------------------------------------------------------ step 4: per-question elastic net, LOCO-CV

L1_RATIOS = (0.5, 1.0)
INNER_CV = 5


def fit_question(scores, control, y, l1_ratios=L1_RATIOS, inner_cv=INNER_CV, min_countries=20):
    """scores (C, F) fPCA features, control (C,) earlier-wave mean, y (C,) change. LOCO-CV. Returns dict of metrics,
    or None if fewer than min_countries have everything finite.

    Control-only baseline is plain OLS (one feature, a CV grid search is overkill and slow). Reference for R² is
    the training-fold mean (R² = 0 by construction); "baseline" = max(0, r2_ctrl_only) -- the better of that trivial
    reference and the control-only OLS -- and gain = r2_oos - baseline. Sign accuracy is compared against the
    majority-class rate (also from LOCO: the training fold's majority sign, predicted for the held-out country)."""
    ok = ~(np.isnan(scores).any(axis=1) | np.isnan(control) | np.isnan(y))
    idx = np.where(ok)[0]
    n = len(idx)
    if n < min_countries:
        return None
    X = np.column_stack([scores[idx], control[idx]])
    yy = y[idx]
    pred_full, pred_ctrl_only, pred_majority, pred_mean, truth = [], [], [], [], []
    for h in range(n):
        tr = np.delete(np.arange(n), h)
        cv = min(inner_cv, n - 1)
        xs = StandardScaler().fit(X[tr])
        enet = ElasticNetCV(l1_ratio=list(l1_ratios), cv=cv, max_iter=20_000).fit(xs.transform(X[tr]), yy[tr])
        pred_full.append(enet.predict(xs.transform(X[h:h + 1]))[0])
        ctrl = LinearRegression().fit(X[tr, -1:], yy[tr])
        pred_ctrl_only.append(ctrl.predict(X[h:h + 1, -1:])[0])
        pred_majority.append(1.0 if (yy[tr] > 0).sum() >= (yy[tr] < 0).sum() else -1.0)
        pred_mean.append(yy[tr].mean())
        truth.append(yy[h])
    truth, pred_full, pred_ctrl_only, pred_majority, pred_mean = map(np.array, (truth, pred_full, pred_ctrl_only, pred_majority, pred_mean))
    ss_tot = ((truth - pred_mean) ** 2).sum()
    r2_oos = 1 - ((truth - pred_full) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    r2_ctrl_only = 1 - ((truth - pred_ctrl_only) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    baseline = max(0.0, r2_ctrl_only) if np.isfinite(r2_ctrl_only) else 0.0
    sign_acc = (np.sign(truth) == np.sign(pred_full)).mean()
    majority_rate = (np.sign(truth) == pred_majority).mean()
    enet_full = ElasticNetCV(l1_ratio=list(l1_ratios), cv=min(inner_cv, n - 1), max_iter=20_000).fit(StandardScaler().fit_transform(X), yy)
    return {"n": n, "r2_oos": r2_oos, "r2_ctrl_only": r2_ctrl_only, "baseline": baseline, "gain": r2_oos - baseline,
            "sign_accuracy": sign_acc, "majority_rate": majority_rate, "sign_acc_minus_majority": sign_acc - majority_rate,
            "coef": enet_full.coef_, "alpha": enet_full.alpha_, "l1_ratio": enet_full.l1_ratio_}


def run_all_questions(scores, fw, wa, wb, rm, questions, **kwargs):
    """{question: fit_question(...) result} for every question with enough countries in this transition."""
    results = {}
    for q in questions:
        X, control, y = build_xy(scores, fw, wa, wb, rm, q)
        res = fit_question(X, control, y, **kwargs)
        if res is not None:
            results[q] = res
    return results


def summary_stats(results):
    """The three Step 5 summary statistics, across questions, from a {question: fit_question() result} dict."""
    if not results:
        return {"n_questions": 0, "n_beat_baseline": 0, "mean_gain": np.nan, "mean_sign_acc_minus_majority": np.nan}
    gains = np.array([r["gain"] for r in results.values()])
    beat = np.array([r["r2_oos"] > r["baseline"] for r in results.values()])
    sam = np.array([r["sign_acc_minus_majority"] for r in results.values()])
    return {"n_questions": len(results), "n_beat_baseline": int(beat.sum()), "mean_gain": float(np.mean(gains)),
            "mean_sign_acc_minus_majority": float(np.mean(sam))}


# ------------------------------------------------------------------ assembling X, y for one transition

def build_xy(scores, fw, wa, wb, rm, question):
    """scores (C, 20, n_comp) for this transition -> X (C, 20*n_comp), control (C,) = earlier-wave mean_rescaled,
    y (C,) = change in mean_rescaled. Countries ordered as COUNTRIES; NaN where missing."""
    C, K, nc = scores.shape
    X = scores.reshape(C, K * nc)
    sub = rm[rm["code"] == question].set_index(["country", "wave"])["mean_rescaled"]
    control = np.array([sub.get((c, wa), np.nan) for c in COUNTRIES])
    later = np.array([sub.get((c, wb), np.nan) for c in COUNTRIES])
    y = later - control
    return X, control, y


# ------------------------------------------------------------------ dry run

DRY_RUN_QUESTIONS = ["G020", "H001", "A008"]
DRY_RUN_CATEGORIES = ["02", "11", "14"]  # Appeal, Disapprove, Protest -- flagged by the earlier correlation passes


def plot_components(pcas_by_cat, transition_label, path):
    fig, axes = sheet(1, len(pcas_by_cat), (6 * len(pcas_by_cat), 5.5), f"fPCA component shapes, {transition_label}",
                      "12-point loadings (x = chunk 1-12 of the interval, in order). Component 1 solid, 2 dashed, 3 dotted.")
    for ax, (root, pca) in zip(np.atleast_1d(axes.flat), pcas_by_cat.items()):
        style(ax)
        for k, (ls, col) in enumerate(zip(["-", "--", ":"], [BLUE, ORANGE, INK])):
            if k < pca.components_.shape[0]:
                ax.plot(range(1, N_POINTS + 1), pca.components_[k], linestyle=ls, color=col, linewidth=1.8,
                        label=f"comp {k + 1} ({pca.explained_variance_ratio_[k]:.0%} var)")
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(f"{root}", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel("chunk", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=105, facecolor=SURFACE)
    plt.close(fig)


def dry_run():
    OUT.mkdir(parents=True, exist_ok=True)
    fw = fieldwork_months()
    print("=== STEP 1: fieldwork months (median interview month, from S022) ===")
    print(fw.to_string())
    print()

    share = event_shares_monthly()
    rm = rescaled_means()

    for wa, wb in TRANSITIONS:
        label = f"{wa}→{wb}"
        curves, months = log_centered_curves(share, fw, wa, wb)
        n_cov = sum(1 for c in months.values() if c[1] >= N_POINTS)
        print(f"=== STEP 2: {label} log-centered curves ===")
        print(f"  curves shape: {curves.shape}  (countries, categories, points)")
        print(f"  countries with >= {N_POINTS} interval months: {n_cov}/{len(COUNTRIES)}")
        short = {c: v for c, v in list(months.items())[:5]}
        print(f"  (baseline_months, interval_months) sample: {short}")

        scores3, pcas3 = fpca_per_category(curves, n_components=3)
        scores1, pcas1 = fpca_per_category(curves, n_components=1)
        print(f"=== STEP 3: {label} functional PCA ===")
        print(f"  3-component scores shape: {scores3.shape} -> flattened feature count: {scores3.shape[1] * scores3.shape[2]}")
        print(f"  1-component scores shape: {scores1.shape} -> flattened feature count: {scores1.shape[1] * scores1.shape[2]}")
        for i in DRY_RUN_CATEGORIES:
            ri = ROOTS.index(i)
            if ri in pcas3:
                print(f"  category {i}: explained variance ratio (first 3) = {pcas3[ri].explained_variance_ratio_.round(3)}")

        pcas_by_cat = {i: pcas3[ROOTS.index(i)] for i in DRY_RUN_CATEGORIES if ROOTS.index(i) in pcas3}
        plot_path = OUT / f"dry_run_components_{wa}to{wb}.png"
        plot_components(pcas_by_cat, label, plot_path)
        print(f"  component plot saved: {plot_path}")
        print()

        print(f"=== STEP 4 (dry run, 3 questions only): {label} elastic net, LOCO-CV ===")
        for q in DRY_RUN_QUESTIONS:
            X, control, y = build_xy(scores3, fw, wa, wb, rm, q)
            print(f"  {q}: X shape {X.shape}, control shape {control.shape}, y shape {y.shape}, "
                  f"finite y count = {np.isfinite(y).sum()}")
            res = fit_question(X, control, y)
            if res is None:
                print(f"    -> skipped (fewer than 20 countries with complete data)")
                continue
            print(f"    n={res['n']}  r2_oos={res['r2_oos']:.3f}  r2_ctrl_only={res['r2_ctrl_only']:.3f}  "
                  f"baseline={res['baseline']:.3f}  gain={res['gain']:.3f}  sign_acc={res['sign_accuracy']:.3f}  "
                  f"majority_rate={res['majority_rate']:.3f}  (acc-maj)={res['sign_acc_minus_majority']:.3f}  "
                  f"alpha={res['alpha']:.4g} l1_ratio={res['l1_ratio']}")
        print()


# ------------------------------------------------------------------ step 5: permutation null (parallel)

state = {}


def _init_worker(scores, fw, wa, wb, rm, questions, fit_kwargs):
    threadpoolctl.threadpool_limits(1)  # one worker process = one CPU's worth of BLAS, no oversubscription
    state.update(scores=scores, fw=fw, wa=wa, wb=wb, rm=rm, questions=questions, fit_kwargs=fit_kwargs)


def _perm_worker(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(state["scores"].shape[0])
    res = run_all_questions(state["scores"][perm], state["fw"], state["wa"], state["wb"], state["rm"], state["questions"], **state["fit_kwargs"])
    return summary_stats(res)


def permutation_null(scores, fw, wa, wb, rm, questions, n_perm, seed0=0, n_jobs=1, fit_kwargs=None, progress=True):
    """Observed summary stats, a list of n_perm null summary-stat dicts (each country's GDELT feature row shuffled,
    same shuffle for every question, Y and control untouched), and per-statistic permutation p-values."""
    fit_kwargs = fit_kwargs or {}
    obs_results = run_all_questions(scores, fw, wa, wb, rm, questions, **fit_kwargs)
    obs_stats = summary_stats(obs_results)
    seeds = [seed0 + i for i in range(n_perm)]
    null_stats = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=(scores, fw, wa, wb, rm, questions, fit_kwargs)) as ex:
            for i, st in enumerate(ex.map(_perm_worker, seeds)):
                null_stats.append(st)
                if progress:
                    print(f"\r  permutation {i + 1}/{n_perm}", end="", flush=True)
    else:
        _init_worker(scores, fw, wa, wb, rm, questions, fit_kwargs)
        for i, s in enumerate(seeds):
            null_stats.append(_perm_worker(s))
            if progress:
                print(f"\r  permutation {i + 1}/{n_perm}", end="", flush=True)
    if progress:
        print()
    pvals = {}
    for key, better in [("n_beat_baseline", "higher"), ("mean_gain", "higher"), ("mean_sign_acc_minus_majority", "higher")]:
        obs_v = obs_stats[key]
        null_v = np.array([s[key] for s in null_stats])
        pvals[key] = (1 + (null_v >= obs_v).sum()) / (1 + len(null_v)) if better == "higher" else (1 + (null_v <= obs_v).sum()) / (1 + len(null_v))
    return obs_results, obs_stats, null_stats, pvals


# ------------------------------------------------------------------ step 6: fit 5->6, project and predict 6->7

def replicate(scores56, fw, rm, results56, questions):
    """For questions present in both transitions: freeze the 5->6 model (scaler + ElasticNetCV, already fit on all
    countries inside fit_question's returned object is NOT kept, so refit once here on the full 5->6 set), predict
    6->7 using 6->7 curves projected onto the 5->6 PCA components (not refit)."""
    share = event_shares_monthly()
    curves67, _ = log_centered_curves(share, fw, 6, 7)
    # refit the 5->6 fPCA (need the fitted PCA objects, not just scores, to project 6->7 onto them)
    curves56, _ = log_centered_curves(share, fw, 5, 6)
    _, pcas56 = fpca_per_category(curves56, n_components=3)
    scores67_proj = project(curves67, pcas56, n_components=3)

    out = []
    for q in questions:
        if q not in results56:
            continue
        X56, control56, y56 = build_xy(scores56, fw, 5, 6, rm, q)
        ok56 = ~(np.isnan(X56).any(axis=1) | np.isnan(control56) | np.isnan(y56))
        if ok56.sum() < 20:
            continue
        X56f = np.column_stack([X56[ok56], control56[ok56]])
        xs = StandardScaler().fit(X56f)
        enet = ElasticNetCV(l1_ratio=list(L1_RATIOS), cv=min(INNER_CV, ok56.sum() - 1), max_iter=20_000).fit(xs.transform(X56f), y56[ok56])

        X67, control67, y67 = build_xy(scores67_proj, fw, 6, 7, rm, q)
        ok67 = ~(np.isnan(X67).any(axis=1) | np.isnan(control67) | np.isnan(y67))
        if ok67.sum() < 5:
            continue
        X67f = np.column_stack([X67[ok67], control67[ok67]])
        pred = enet.predict(xs.transform(X67f))
        truth = y67[ok67]
        ss_tot = ((truth - truth.mean()) ** 2).sum()
        r2 = 1 - ((truth - pred) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
        sign_acc = (np.sign(truth) == np.sign(pred)).mean()
        out.append({"question": q, "n_67": int(ok67.sum()), "r2_replication": r2, "sign_acc_replication": sign_acc})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ full run (steps 4-6), for Slurm

def eligible_questions(fw, wa, wb, rm, min_countries=20):
    """Questions with >= min_countries countries that have a valid mean_rescaled in BOTH wa and wb (matched pairs,
    matching fit_question's own min_countries gate -- this is a prefilter so run_all_questions doesn't waste time
    building X/fitting models for questions that are obviously too sparse)."""
    sub = rm[rm["wave"].isin([wa, wb])]
    piv = sub.pivot_table(index=["code", "country"], columns="wave", values="mean_rescaled")
    if wa not in piv.columns or wb not in piv.columns:
        return []
    ok = piv[[wa, wb]].notna().all(axis=1)
    cnt = ok.groupby(level="code").sum()
    return sorted(cnt.index[cnt >= min_countries])


def full_run(n_perm=200, n_jobs=None, seed0=0, out=OUT, resume=True):
    """resume=True (default): a transition whose results_<label>.csv already exists on disk is loaded instead of
    recomputed (used to pick up after the 2026-10-27 run's 5->6 slice finished but 6->7 hit the Slurm time limit
    mid-permutation)."""
    n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    out.mkdir(parents=True, exist_ok=True)
    fw = fieldwork_months()
    share = event_shares_monthly()
    rm = rescaled_means()
    print(f"full_run: n_perm={n_perm} n_jobs={n_jobs} resume={resume}")

    results56 = None
    all_summary = {}
    for wa, wb in TRANSITIONS:
        label = f"{wa}to{wb}"
        curves, _ = log_centered_curves(share, fw, wa, wb)
        scores, _ = fpca_per_category(curves, n_components=3)
        questions = eligible_questions(fw, wa, wb, rm)

        cached = out / f"results_{label}.csv"
        if resume and cached.exists():
            print(f"[{label}] resuming from {cached} (already completed)")
            obs_results = {r["question"]: r for r in pd.read_csv(cached).to_dict("records")}
            obs_stats = summary_stats(obs_results)
            pvals = pd.read_csv(out / f"pvalues_{label}.csv", index_col=0).iloc[:, 0].to_dict()
        else:
            t0 = time.time()
            print(f"[{label}] {len(questions)} eligible questions, computing observed fit + {n_perm} permutations...")
            obs_results, obs_stats, null_stats, pvals = permutation_null(scores, fw, wa, wb, rm, questions, n_perm, seed0=seed0, n_jobs=n_jobs)
            print(f"[{label}] observed: {obs_stats}")
            print(f"[{label}] p-values: {pvals}")
            print(f"[{label}] wall time: {time.time() - t0:.0f}s")
            rows = [{"question": q, **r} for q, r in obs_results.items()]
            pd.DataFrame(rows).drop(columns="coef", errors="ignore").to_csv(cached, index=False)
            pd.DataFrame(null_stats).to_csv(out / f"null_stats_{label}.csv", index=False)
            pd.Series(pvals).to_csv(out / f"pvalues_{label}.csv")
        all_summary[label] = {"observed": obs_stats, "pvalues": pvals, "n_questions": len(questions)}
        if wa == 5:
            results56, scores56, questions56 = obs_results, scores, questions

    print("[replication] fitting 5->6, projecting 6->7...")
    rep = replicate(scores56, fw, rm, results56, questions56)
    rep.to_csv(out / "replication_5to6_on_6to7.csv", index=False)
    print(rep.to_string(index=False))
    pd.Series(all_summary).to_json(out / "summary.json", indent=2)
    print("done.")


if __name__ == "__main__":
    if "--dry-run" in sys.argv or len(sys.argv) == 1:
        dry_run()
    elif "--perm-check" in sys.argv:
        fw = fieldwork_months()
        share = event_shares_monthly()
        rm = rescaled_means()
        for wa, wb in TRANSITIONS:
            curves, _ = log_centered_curves(share, fw, wa, wb)
            scores, _ = fpca_per_category(curves, n_components=3)
            questions = eligible_questions(fw, wa, wb, rm)
            print(f"{wa}->{wb}: {len(questions)} eligible questions")
            obs_results, obs_stats, null_stats, pvals = permutation_null(scores, fw, wa, wb, rm, questions, n_perm=2, n_jobs=1)
            print("  observed:", obs_stats)
            print("  null (2 shuffles):", null_stats)
    elif "--full" in sys.argv:
        n_perm = 200
        for a in sys.argv:
            if a.startswith("--n-perm="):
                n_perm = int(a.split("=")[1])
        full_run(n_perm=n_perm)
    else:
        print("usage: --dry-run | --perm-check | --full [--n-perm=200]")
