"""Decide whether a country's answers changed significantly between two WVS waves.

A change counts as significant only if it is BOTH
  * reliable: permutation-test p-value, Benjamini-Hochberg corrected across all tests for that
    country and wave pair, is below `alpha`; and
  * big enough: the effect size is at least the threshold for the question's format.

Effect size by format (see data/question_categories.csv, built by categorize_questions.py):
  ordinal, numeric_scale : Wasserstein distance after rescaling answers to 0-1
  count                  : Wasserstein distance in original units
  binary, mention items  : change in the share giving the higher code (percentage points, 0-1)
  knowledge              : change in the share answering correctly
  categorical, forced_choice : total variation distance (half the sum of changes in option shares),
                           with Cramer's V + chi-square reported alongside
  ranked_choice pairs    : total variation distance on first+second choice mentions combined
  mention_flag batteries : per-item change, plus battery-level mean absolute change over its items

All thresholds live in `ChangeConfig` so they can be tuned.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES_PATH = ROOT / "data" / "question_categories.csv"
PARSED_PATH = ROOT / "data" / "parsed_survey_data_cleaned.jsonl"


@dataclass
class ChangeConfig:
    # --- reliability ---
    alpha: float = 0.05          # FDR level (Benjamini-Hochberg) for the permutation p-values
    min_n: int = 200             # min valid respondents per wave for a test to be run
    n_perm: int = 999            # permutations per test
    seed: int = 0
    # --- minimum effect sizes (a change must reach these to count) ---
    ordinal_w1: float = 0.05     # ordinal + numeric_scale: Wasserstein on 0-1 scale
    binary_pp: float = 0.05      # binary, knowledge, mention items: share change (0.05 = 5 points)
    categorical_tvd: float = 0.10    # categorical, forced_choice: total variation distance
    categorical_cramers_v: float = 0.10  # categorical, forced_choice: Cramer's V
    categorical_rule: str = "tvd"    # which one decides: "tvd", "cramers_v", "either", "both"
    ranked_tvd: float = 0.10     # ranked_choice pairs (first + second choice combined)
    battery_mean_pp: float = 0.03    # mention_flag battery: mean absolute item change
    count_w1: float = 0.5        # count questions: Wasserstein in raw units (children)


# Codes that sit outside the ordered scale (dropped before measuring change).
OFF_SCALE = {"E133": [4], "E134": [4], "E255": [4], "E275": [0]}
# Correct answer code for each knowledge question.
KNOWLEDGE_CORRECT = {"E279": 3, "E280": 1, "E281": 2}
# First-choice / second-choice question pairs, analysed together.
RANKED_PAIRS = [("A044", "A045"), ("C009", "C010"), ("E001", "E002"), ("E003", "E004"),
                ("E005", "E006"), ("E190", "E191"), ("E238", "E239"), ("E240", "E241"),
                ("G001", "G002"), ("G024", "G025")]
CHILD_QUALITY_COLS = ["A027", "A029", "A030", "A032", "A034", "A035", "A038", "A039", "A040",
                      "A041", "A042", "A043B"]
CHILD_QUALITY_CODE = "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"


# ------------------------------------------------------------------ helpers

def load_question_info(categories_path=CATEGORIES_PATH, parsed_path=PARSED_PATH):
    """Returns (categories_df, parsed) where parsed maps code -> {'choices': [...], ...}."""
    cats = pd.read_csv(categories_path).fillna("")
    parsed = {}
    with open(parsed_path) as f:
        for line in f:
            parsed.update(json.loads(line))
    return cats, parsed


def scale_range(code, parsed):
    """(lo, hi) of the on-scale answer codes listed in the codebook, or None if unknown."""
    codes = []
    for c in parsed.get(code, {}).get("choices", []):
        m = re.match(r"\s*(-?\d+)", str(c))
        if m and int(m.group(1)) >= 0 and int(m.group(1)) not in OFF_SCALE.get(code, []):
            codes.append(int(m.group(1)))
    return (min(codes), max(codes)) if len(codes) >= 2 else None


def _clean(values, code=None):
    """Drop missing / WVS negative codes / off-scale codes."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if code in OFF_SCALE:
        v = v[~np.isin(v, OFF_SCALE[code])]
    return v


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


# ---- statistics on count vectors (last axis = answer options; works on batches of permutations)

def _tvd(c1, c2, n1, n2, vals):
    return 0.5 * np.abs(c1 / n1 - c2 / n2).sum(-1)


def _w1(scale):
    def stat(c1, c2, n1, n2, vals):
        cdf_gap = np.abs(np.cumsum(c1 / n1, -1) - np.cumsum(c2 / n2, -1))[..., :-1]
        return (cdf_gap * (np.diff(vals) / scale)).sum(-1)
    return stat


def _perm_test(x1, x2, stat, n_perm, rng):
    """Exact-style permutation test for discrete answers: shuffling wave labels is the same as
    drawing wave-1 counts from a multivariate hypergeometric on the pooled counts."""
    vals, inv = np.unique(np.concatenate([x1, x2]), return_inverse=True)
    k, n1, n2 = len(vals), len(x1), len(x2)
    c1 = np.bincount(inv[:n1], minlength=k).astype(float)
    c2 = np.bincount(inv[n1:], minlength=k).astype(float)
    if k < 2:
        return 0.0, 1.0, vals, c1, c2
    obs = float(stat(c1, c2, n1, n2, vals))
    pooled = (c1 + c2).astype(np.int64)
    p1 = rng.multivariate_hypergeometric(pooled, n1, size=n_perm).astype(float)
    p2 = pooled - p1
    perm = stat(p1, p2, n1, n2, vals)
    p = (1 + np.sum(perm >= obs - 1e-12)) / (n_perm + 1)
    return obs, float(p), vals, c1, c2


def _chi_square(c1, c2):
    """Cramer's V and chi-square p-value for the 2 x k table of wave-by-option counts."""
    if len(c1) < 2:
        return 0.0, 1.0
    table = np.vstack([c1, c2])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(np.sqrt(chi2 / table.sum())), float(p)  # min(rows, cols) - 1 == 1 for 2 waves


# ------------------------------------------------------------------ per-format tests

def _row(country, w1, w2, code, level, fmt, group, n1, n2, effect_name, effect, p, **extra):
    r = dict(country=country, wave1=w1, wave2=w2, code=code, level=level, format=fmt, group=group,
             n1=n1, n2=n2, effect_name=effect_name, effect=effect, p_value=p,
             signed_change=np.nan, detail="", cramers_v=np.nan, p_chi2=np.nan, threshold=np.nan,
             top_gain_option=np.nan, top_gain=np.nan)
    r.update(extra)
    return r


def _test_scale(code, fmt, a, b, cfg, rng, parsed, ctx):
    x1, x2 = _clean(a, code), _clean(b, code)
    if min(len(x1), len(x2)) < cfg.min_n:
        return []
    if fmt == "count":
        scale, name, tau = 1.0, "wasserstein_raw", cfg.count_w1
    else:
        rng_ = scale_range(code, parsed) or (min(x1.min(), x2.min()), max(x1.max(), x2.max()))
        scale = max(rng_[1] - rng_[0], 1e-9)
        name, tau = "wasserstein_01", cfg.ordinal_w1
    eff, p, *_ = _perm_test(x1, x2, _w1(scale), cfg.n_perm, rng)
    return [_row(*ctx, code, "question", fmt, "", len(x1), len(x2), name, eff, p,
                 signed_change=(x2.mean() - x1.mean()) / scale, threshold=tau)]


def _test_share(code, fmt, a, b, cfg, rng, ctx, level="question", group="", recode=None):
    """Binary-style (two options) or multi-option categorical tests using total variation distance."""
    x1, x2 = _clean(a, code), _clean(b, code)
    if recode is not None:
        x1, x2 = recode(x1), recode(x2)
    if min(len(x1), len(x2)) < cfg.min_n:
        return []
    eff, p, vals, c1, c2 = _perm_test(x1, x2, _tvd, cfg.n_perm, rng)
    diff = c2 / len(x2) - c1 / len(x1)
    two_way = len(vals) <= 2
    if fmt in ("categorical", "forced_choice"):
        v, p_chi = _chi_square(c1, c2)
        top, gain = int(np.argmax(np.abs(diff))), int(np.argmax(diff))
        return [_row(*ctx, code, level, fmt, group, len(x1), len(x2), "tvd", eff, p,
                     detail=f"top mover: option {vals[top]:g} ({diff[top]:+.3f})",
                     cramers_v=v, p_chi2=p_chi, threshold=cfg.categorical_tvd,
                     top_gain_option=float(vals[gain]), top_gain=float(diff[gain]))]
    return [_row(*ctx, code, level, fmt, group, len(x1), len(x2), "share_change", eff, p,
                 signed_change=float(diff[-1]) if two_way else np.nan, threshold=cfg.binary_pp)]


def _test_ranked(first, second, a1, a2, b1, b2, cfg, rng, ctx):
    """First+second choice combined: each respondent contributes up to two mentions."""
    def mentions(f, s):
        f, s = np.asarray(f, float), np.asarray(s, float)
        ok = np.isfinite(f) & np.isfinite(s) & (f >= 0) & (s >= 0)
        return f[ok], s[ok]

    f1, s1 = mentions(a1, a2)
    f2, s2 = mentions(b1, b2)
    n1, n2 = len(f1), len(f2)
    if min(n1, n2) < cfg.min_n:
        return []
    opts = np.unique(np.concatenate([f1, s1, f2, s2]))
    if len(opts) < 2:
        return []
    idx = {o: i for i, o in enumerate(opts)}
    M = np.zeros((n1 + n2, len(opts)))
    for r, (f, s) in enumerate(zip(np.concatenate([f1, f2]), np.concatenate([s1, s2]))):
        M[r, idx[f]] += 1
        M[r, idx[s]] += 1
    tot = M.sum(0)
    stat = lambda m1, m2: 0.5 * np.abs(m1 / (2 * n1) - m2 / (2 * n2)).sum(-1)
    obs = float(stat(M[:n1].sum(0), M[n1:].sum(0)))
    perm = np.empty(cfg.n_perm)
    for i in range(cfg.n_perm):
        sel = rng.permutation(n1 + n2)[:n1]
        m1 = M[sel].sum(0)
        perm[i] = stat(m1, tot - m1)
    p = (1 + np.sum(perm >= obs - 1e-12)) / (cfg.n_perm + 1)
    diff = M[n1:].sum(0) / (2 * n2) - M[:n1].sum(0) / (2 * n1)
    top, gain = int(np.argmax(np.abs(diff))), int(np.argmax(diff))
    return [_row(*ctx, f"{first}+{second}", "question", "ranked_choice", "", n1, n2, "tvd", obs, p,
                 detail=f"top mover: option {opts[top]:g} ({diff[top]:+.3f})",
                 threshold=cfg.ranked_tvd,
                 top_gain_option=float(opts[gain]), top_gain=float(diff[gain]))]


def _test_battery(group, cols, get1, get2, cfg, rng, ctx):
    """Item-level share tests are run elsewhere; this adds one battery-level test:
    mean absolute change in share over the battery's items, permuting whole respondents."""
    def matrix(get):
        X = np.column_stack([np.asarray(get(c), float) for c in cols])
        valid = np.isfinite(X) & (X >= 0)
        return np.where(valid, (X >= 1).astype(float), 0.0), valid.astype(float)

    X1, V1 = matrix(get1)
    X2, V2 = matrix(get2)
    usable = (V1.sum(0) >= cfg.min_n) & (V2.sum(0) >= cfg.min_n)
    if usable.sum() < 2:
        return []
    X = np.vstack([X1, X2])[:, usable]
    V = np.vstack([V1, V2])[:, usable]
    n1, n2 = len(X1), len(X2)
    labels = np.zeros(n1 + n2, bool)
    labels[:n1] = True

    def stat(sel):  # sel: boolean matrix (perms x respondents), True = wave 1
        s1, v1 = sel @ X, sel @ V
        s2, v2 = (~sel) @ X, (~sel) @ V
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s2 / v2 - s1 / v1
        return np.nanmean(np.abs(d), axis=-1), np.nanmean(d, axis=-1)

    obs, signed = stat(labels[None, :])
    perm_sel = np.argsort(rng.random((cfg.n_perm, n1 + n2)), axis=1) < n1
    perm, _ = stat(perm_sel)
    p = (1 + np.sum(perm >= obs[0] - 1e-12)) / (cfg.n_perm + 1)
    return [_row(*ctx, group, "battery", "mention_flag", group, n1, n2, "mean_abs_share_change",
                 float(obs[0]), float(p), signed_change=float(signed[0]),
                 detail=f"{int(usable.sum())} items", threshold=cfg.battery_mean_pp)]


# ------------------------------------------------------------------ main entry points

def detect_country_changes(df, country, wave1, wave2, cfg=None, cats=None, parsed=None,
                           country_col="COUNTRY_ALPHA", wave_col="S002VS"):
    """All change tests for one country and wave pair. Returns a DataFrame, one row per test."""
    cfg = cfg or ChangeConfig()
    if cats is None or parsed is None:
        cats, parsed = load_question_info()
    rng = np.random.default_rng(cfg.seed)
    s1 = df[(df[wave_col] == wave1) & (df[country_col] == country)]
    s2 = df[(df[wave_col] == wave2) & (df[country_col] == country)]
    if s1.empty or s2.empty:
        return pd.DataFrame()
    have = set(df.columns)
    ctx = (country, wave1, wave2)
    get1 = lambda c: s1[c].to_numpy()
    get2 = lambda c: s2[c].to_numpy()
    rows = []

    ranked_codes = {c for pair in RANKED_PAIRS for c in pair}
    for _, r in cats.iterrows():
        code, fmt = r["code"], r["format"]
        if fmt == "identifier" or code in ranked_codes or code not in have:
            continue
        if fmt in ("ordinal_scale", "numeric_scale", "count"):
            rows += _test_scale(code, fmt, get1(code), get2(code), cfg, rng, parsed, ctx)
        elif fmt in ("binary", "categorical", "forced_choice"):
            if fmt == "categorical" and not parsed.get(code, {}).get("choices"):
                continue  # no listed answer options (country-specific codes): not interpretable
            rows += _test_share(code, fmt, get1(code), get2(code), cfg, rng, ctx)
        elif fmt == "knowledge":
            ok = KNOWLEDGE_CORRECT[code]
            rows += _test_share(code, fmt, get1(code), get2(code), cfg, rng, ctx,
                                recode=lambda v, ok=ok: (v == ok).astype(float))
        elif fmt == "mention_flag" and code != CHILD_QUALITY_CODE:
            rows += _test_share(code, fmt, get1(code), get2(code), cfg, rng, ctx,
                                level="item", group=r["group"])

    for first, second in RANKED_PAIRS:
        if first in have and second in have:
            rows += _test_ranked(first, second, get1(first), get1(second), get2(first),
                                 get2(second), cfg, rng, ctx)

    batteries = {g: list(gr["code"]) for g, gr in cats[cats["format"] == "mention_flag"].groupby("group")
                 if g}
    batteries["child_qualities"] = [c for c in CHILD_QUALITY_COLS if c in have]
    for g, cols in batteries.items():
        cols = [c for c in cols if c in have and c != CHILD_QUALITY_CODE]
        if g == "child_qualities":  # the 12 sub-columns are the items
            for c in cols:
                rows += _test_share(c, "mention_flag", get1(c), get2(c), cfg, rng, ctx,
                                    level="item", group=g)
        if len(cols) >= 2:
            rows += _test_battery(g, cols, get1, get2, cfg, rng, ctx)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return apply_significance(res, cfg)


def apply_significance(res, cfg):
    """FDR-correct within this country/wave pair, then require the minimum effect size."""
    res = res.copy()
    res["q_value"] = benjamini_hochberg(res["p_value"].to_numpy())
    reliable = (res["q_value"] < cfg.alpha) & (res["effect"] >= res["threshold"])
    res["significant"] = reliable

    cat = res["format"].isin(["categorical", "forced_choice"])
    if cat.any():
        res["q_chi2"] = np.nan
        res.loc[cat, "q_chi2"] = benjamini_hochberg(res.loc[cat, "p_chi2"].to_numpy())
        v_ok = (res["q_chi2"] < cfg.alpha) & (res["cramers_v"] >= cfg.categorical_cramers_v)
        rule = cfg.categorical_rule
        combined = {"tvd": reliable, "cramers_v": v_ok, "either": reliable | v_ok,
                    "both": reliable & v_ok}[rule]
        res.loc[cat, "significant"] = combined[cat]
        res["sig_tvd"] = np.where(cat, reliable, np.nan)
        res["sig_cramers_v"] = np.where(cat, v_ok, np.nan)
    return res


def detect_changes(df, wave1, wave2, countries=None, cfg=None, progress=True,
                   country_col="COUNTRY_ALPHA", wave_col="S002VS"):
    """Run detect_country_changes for every country present in both waves (or `countries`)."""
    cfg = cfg or ChangeConfig()
    cats, parsed = load_question_info()
    if countries is None:
        c1 = set(df.loc[df[wave_col] == wave1, country_col].dropna())
        c2 = set(df.loc[df[wave_col] == wave2, country_col].dropna())
        countries = sorted(c1 & c2)
    it = countries
    if progress:
        from tqdm import tqdm
        it = tqdm(countries, desc=f"waves {wave1}->{wave2}")
    out = [detect_country_changes(df, c, wave1, wave2, cfg, cats, parsed, country_col, wave_col)
           for c in it]
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
