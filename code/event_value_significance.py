"""Which event-vs-survey correlations are significant? 2,000 country shuffles for both analyses (levels and changes), pair-level and family-wise p,
Benjamini-Hochberg q within each slice, leave-one-country-out robustness, and the raw points of every flagged pair for the explorer's correlations tab.

    python code/event_value_significance.py       # writes gdelt_plots/event_value_significance/{all_pairs.csv, significant.csv, corr.json}

Significant  = family-wise permutation p < 0.05 within its slice (chance of a |r| this large anywhere in the slice).
Suggestive   = Benjamini-Hochberg q < 0.10 within its slice, but not significant.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import event_value_changes as C  # noqa: E402
import event_value_correlations as L  # noqa: E402
from change_detection import load_question_info, scale_range  # noqa: E402
from direction_baseline import parse_choices  # noqa: E402
from event_value_inputs import COUNTRIES, OUT as BASE, ROOTS  # noqa: E402

OUT = BASE.parent / "event_value_significance"
P = 2000
LV_SL = ["wave 5", "wave 6", "wave 7", "all waves pooled (wave effects removed)", "within country (country and wave effects removed)", "wave-to-wave changes (pair effects removed)"]
CH_SL = ["rate of change 5→6", "rate of change 6→7", "rate of change 5→7", "5→6 and 6→7 rates pooled (pair effects removed)", "acceleration (rate 6→7 minus rate 5→6)",
         "5→6 and 6→7 rates pooled, rank correlation"]
state = {}


def _init(X3, Y3, yrs):
    state.update(X3=X3, Y3=Y3, yrs=yrs, DY=C.rates(Y3, yrs))


def _perm(args):
    kind, seed = args
    X3, Y3, yrs = state["X3"], state["Y3"], state["yrs"]
    perm = np.random.default_rng(seed).permutation(X3.shape[0])
    if kind == "levels":
        R, _ = L.slice_r(X3[perm], Y3)
    else:
        R, _ = C.slices(C.rates(X3[perm], yrs[perm]), state["DY"])
    return R.astype(np.float32)


def pvalues(Robs, Rperm):
    """pair-level two-sided p, family-wise p, BH q (within each slice)."""
    ab, ap = np.abs(Robs), np.abs(Rperm)
    n = Rperm.shape[0]
    p_pair = np.full(Robs.shape, np.nan)
    for s in range(Robs.shape[0]):
        cnt = (np.nan_to_num(ap[:, s], nan=-1) >= np.nan_to_num(ab[s], nan=9)[None]).sum(axis=0)
        p_pair[s] = (1 + cnt) / (1 + n)
    p_pair[np.isnan(Robs)] = np.nan
    mx = np.nanmax(ap.reshape(n, Robs.shape[0], -1), axis=2)                     # (P, S)
    p_fw = np.full(Robs.shape, np.nan)
    q = np.full(Robs.shape, np.nan)
    for s in range(Robs.shape[0]):
        p_fw[s] = (1 + (mx[:, s][:, None, None] >= np.nan_to_num(ab[s], nan=9)[None]).sum(axis=0)) / (1 + n)
        ok = np.isfinite(Robs[s])
        p = p_pair[s][ok]
        order = np.argsort(p)
        ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
        qq = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(len(p)); out[order] = np.minimum(qq, 1.0)
        tmp = np.full(Robs[s].shape, np.nan); tmp[ok] = out; q[s] = tmp
    p_fw[np.isnan(Robs)] = np.nan
    return p_pair, p_fw, q


def high_side(code, parsed):
    ch = parse_choices(code, parsed)
    rng = scale_range(code, parsed) or ((min(c for c, _ in ch), max(c for c, _ in ch)) if ch else None)
    if not ch or not rng:
        return "", ""
    mid = (rng[0] + rng[1]) / 2
    return " / ".join(l for c, l in ch if c > mid), " / ".join(l for c, l in ch if c <= mid)


def points(kind, s, i, j, X3, Y3, DX, DY):
    out = []
    if kind == "levels":
        if s <= 2:
            for c in range(len(COUNTRIES)):
                if not np.isnan(Y3[c, s, j]):
                    out.append((COUNTRIES[c], f"wave {s + 5}", X3[c, s, i], Y3[c, s, j]))
        elif s in (3, 4):
            full = ~np.isnan(Y3[:, :, j]).any(axis=1)
            for c in range(len(COUNTRIES)):
                for w in range(3):
                    if not np.isnan(Y3[c, w, j]) and (s == 3 or full[c]):
                        out.append((COUNTRIES[c], f"wave {w + 5}", X3[c, w, i], Y3[c, w, j]))
        else:
            for a, b, lab in ((0, 1, "5→6"), (1, 2, "6→7")):
                for c in range(len(COUNTRIES)):
                    if not np.isnan(Y3[c, a, j]) and not np.isnan(Y3[c, b, j]):
                        out.append((COUNTRIES[c], lab, X3[c, b, i] - X3[c, a, i], Y3[c, b, j] - Y3[c, a, j]))
    else:
        keys = {0: [("r12", "5→6")], 1: [("r23", "6→7")], 2: [("r13", "5→7")], 3: [("r12", "5→6"), ("r23", "6→7")], 5: [("r12", "5→6"), ("r23", "6→7")]}
        if s == 4:
            for c in range(len(COUNTRIES)):
                y = DY["r23"][c, j] - DY["r12"][c, j]
                if not np.isnan(y):
                    out.append((COUNTRIES[c], "acceleration", DX["r23"][c, i] - DX["r12"][c, i], y))
        else:
            for k, lab in keys[s]:
                for c in range(len(COUNTRIES)):
                    if not np.isnan(DY[k][c, j]):
                        out.append((COUNTRIES[c], lab, DX[k][c, i], DY[k][c, j]))
    return [[c, g, round(float(x), 4), round(float(y), 4)] for c, g, x, y in out]


def loo(kind, s, i, j, X3, Y3, yrs):
    rs = []
    for k in range(len(COUNTRIES)):
        if kind == "levels":
            R, _ = L.slice_r(np.delete(X3, k, 0), np.delete(Y3, k, 0)[:, :, [j]])
        else:
            R, _ = C.slices(C.rates(np.delete(X3, k, 0), np.delete(yrs, k, 0)), C.rates(np.delete(Y3, k, 0)[:, :, [j]], np.delete(yrs, k, 0)))
        rs.append(R[s, i, 0])
    return np.array(rs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X3, Y3, codes, years = L.load()
    yrs = np.full((len(COUNTRIES), 3), np.nan)
    for (c, w), y in years.items():
        yrs[COUNTRIES.index(c), w - 5] = y
    cats, parsed = load_question_info()
    cats = cats.set_index("code")
    DX, DY = C.rates(X3, yrs), C.rates(Y3, yrs)
    obs = {"levels": L.slice_r(X3, Y3)[0], "changes": C.slices(DX, DY)[0]}
    rows = []
    for kind in ("levels", "changes"):
        with ProcessPoolExecutor(32, initializer=_init, initargs=(X3, Y3, yrs)) as ex:
            Rp = np.stack(list(ex.map(_perm, [(kind, 1000 + k) for k in range(P)], chunksize=20)))
        p_pair, p_fw, q = pvalues(obs[kind], Rp)
        names = LV_SL if kind == "levels" else CH_SL
        for s in range(6):
            for i in range(20):
                for j, code in enumerate(codes):
                    r = obs[kind][s, i, j]
                    if np.isfinite(r):
                        rows.append((kind, s, names[s], ROOTS[i], code, cats.loc[code, "label"] if code in cats.index else code, r, p_pair[s, i, j], p_fw[s, i, j], q[s, i, j]))
        print(kind, "done", flush=True)
    df = pd.DataFrame(rows, columns=["analysis", "slice_id", "slice", "event", "question", "question_label", "r", "p_pair", "p_familywise", "q_bh"])
    df["level"] = np.where(df["p_familywise"] < 0.05, "significant", np.where(df["q_bh"] < 0.10, "suggestive", ""))
    df.to_csv(OUT / "all_pairs.csv", index=False)
    flagged = df[df["level"] != ""].copy()
    flagged.sort_values(["level", "p_familywise", "q_bh"], ascending=[False, True, True]).to_csv(OUT / "significant.csv", index=False)
    print(df.groupby(["analysis", "slice_id"]).agg(pairs=("r", "size"), significant=("level", lambda x: (x == "significant").sum()), suggestive=("level", lambda x: (x == "suggestive").sum())).to_string())

    findings = []
    for r in flagged.itertuples():
        i, j = ROOTS.index(r.event), codes.index(r.question)
        l = loo(r.analysis, r.slice_id, i, j, X3, Y3, yrs)
        worst = int(np.nanargmin(np.abs(l)))
        hi, lo = high_side(r.question, parsed)
        findings.append(dict(analysis=r.analysis, slice_id=int(r.slice_id), slice=r.slice, event=r.event, question=r.question, question_label=r.question_label,
                             question_text=parsed[r.question]["parsed_question"].strip().replace("\n", " ") if r.question in parsed else "", high=hi, low=lo,
                             r=round(float(r.r), 3), n=int(len(points(r.analysis, r.slice_id, i, j, X3, Y3, DX, DY))), p_pair=round(float(r.p_pair), 4),
                             p_fw=round(float(r.p_familywise), 4), q=round(float(r.q_bh), 4), level=r.level,
                             loo_min=round(float(np.nanmin(l)), 3), loo_max=round(float(np.nanmax(l)), 3), loo_country=COUNTRIES[worst], r_without=round(float(l[worst]), 3),
                             points=points(r.analysis, r.slice_id, i, j, X3, Y3, DX, DY)))
    meta = dict(n_perm=P, tests={f"{k}": int(((df["analysis"] == k)).sum()) for k in ("levels", "changes")},
                n_significant=int((df["level"] == "significant").sum()), n_suggestive=int((df["level"] == "suggestive").sum()))
    (OUT / "corr.json").write_text(json.dumps({"meta": meta, "findings": findings}, separators=(",", ":")))
    print(meta)
    print(flagged.sort_values("p_familywise").head(30)[["analysis", "slice", "event", "question", "question_label", "r", "p_familywise", "q_bh", "level"]].to_string(index=False, max_colwidth=36))


if __name__ == "__main__":
    main()
