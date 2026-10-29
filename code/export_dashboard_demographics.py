"""Data file for the dashboard's "Demographics" tab: for each country, wave, and demographic split (sex, age,
settlement size), which questions show a significant difference in how the two groups answered -- same protocol
as change_detection.py (permutation test + effect-size threshold), just comparing two demographic groups within
one wave instead of two waves.

    python code/export_dashboard_demographics.py

Reads the raw WVS CSV directly (demographic columns aren't in data/question_categories.csv's usual load path).
Writes results/direction_baseline/five_country_set/demographics_data.js (window.DEMOGRAPHICS).

Each row has TWO different numbers, which look similar but are not the same thing:
  effect  the test statistic the significance decision is actually based on (Wasserstein distance for scales,
          total variation distance for categorical/ranked, share-change for binary/mention/knowledge) -- compared
          against `thr`, the minimum for that format.
  diff    group_b's average answer minus group_a's (mean, for scale/count questions; share giving the tracked
          answer, in points, for binary/mention/knowledge/categorical/forced_choice/ranked_choice).
For binary-style (2-category) questions these two coincide by construction (total variation distance on 2
categories collapses to |diff|); for scales they usually do not (Wasserstein distance captures distribution shape,
not just the mean shift), so look at `diff`, not `effect`, to know which group's answer was actually higher.
mean_a/mean_b (scale/count) or share_a/share_b (everything else) are the two groups' own raw average answers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_detection import KNOWLEDGE_CORRECT, RANKED_PAIRS, ChangeConfig, _clean, detect_country_changes, load_question_info  # noqa: E402
from demographic_splice import COUNTRIES, SUBGROUPS, load_raw  # noqa: E402
from direction_baseline import COUNTRY_NAMES, RESULTS_DIR, parse_choices  # noqa: E402
from export_dashboard_baseline import thresholds  # noqa: E402

OUT = RESULTS_DIR / "five_country_set" / "demographics_data.js"
WAVES = (5, 6, 7)
GROUP_LABEL = {"sex": {"male": "Male", "female": "Female"},
               "age": {"under45": "Under 45", "45plus": "45 or older"},
               "settlement": {"smaller": "Smaller settlement", "larger": "Larger settlement"}}
RANKED_SECOND = {a: b for a, b in RANKED_PAIRS}


def ordered_stats(gA, gB, code):
    x1, x2 = _clean(gA[code].to_numpy(), code), _clean(gB[code].to_numpy(), code)
    return float(x1.mean()), float(x2.mean())


def share_stats(gA, gB, code, fmt):
    x1, x2 = _clean(gA[code].to_numpy(), code), _clean(gB[code].to_numpy(), code)
    if fmt == "knowledge":
        ok = KNOWLEDGE_CORRECT[code]
        x1, x2 = (x1 == ok).astype(float), (x2 == ok).astype(float)
        tracked = 1.0
    else:
        tracked = max(x1.max(initial=-1), x2.max(initial=-1))  # the highest code, same as _test_share's vals[-1]
    return float((x1 == tracked).mean()) * 100, float((x2 == tracked).mean()) * 100


def option_stats(gA, gB, code, top_gain_option):
    x1, x2 = _clean(gA[code].to_numpy(), code), _clean(gB[code].to_numpy(), code)
    return float((x1 == top_gain_option).mean()) * 100, float((x2 == top_gain_option).mean()) * 100


def ranked_stats(gA, gB, first, second, top_gain_option):
    def share(g):
        f, s = _clean(g[first].to_numpy(), first), _clean(g[second].to_numpy(), second)
        n = min(len(f), len(s))  # mentions() in change_detection.py keeps only respondents valid on both
        f, s = f[:n], s[:n]
        return float(((f == top_gain_option).sum() + (s == top_gain_option).sum()) / (2 * n)) * 100
    return share(gA), share(gB)


def build():
    df = load_raw()
    cats, parsed = load_question_info()
    cats_idx = cats.set_index("code")
    cfg = ChangeConfig()
    thr = thresholds(cfg)

    rows = []
    trend = {}  # "country|wave|demo|code" -> {v_a, v_b, stat_kind, significant} for EVERY tested question
    # (not just significant ones), so a wave-to-wave trend line doesn't have holes where a wave just wasn't
    # significant. Only for ordered (mean) and binary/mention_flag/knowledge (share of a fixed tracked value)
    # formats: categorical/forced_choice/ranked_choice's "tracked option" is chosen per-pair as whichever option
    # moved most, so it isn't a fixed, comparable quantity across waves -- skipped for the trend line.
    for country in COUNTRIES:
        for wave in WAVES:
            base = df[(df["COUNTRY_ALPHA"] == country) & (df["S002VS"] == wave)]
            for demo, groups in SUBGROUPS.items():
                names = list(groups.keys())
                gA = base[groups[names[0]](base)].copy()
                gB = base[groups[names[1]](base)].copy()
                if len(gA) < cfg.min_n or len(gB) < cfg.min_n:
                    continue
                gA2, gB2 = gA.copy(), gB.copy()
                gA2["S002VS"], gB2["S002VS"] = 1, 2
                tmp = pd.concat([gA2, gB2], ignore_index=True)
                t_all = detect_country_changes(tmp, country, 1, 2, cfg, cats, parsed,
                                               country_col="COUNTRY_ALPHA", wave_col="S002VS")
                t_all = t_all[t_all["level"].isin(["question", "item"])]

                for _, r in t_all.iterrows():
                    fmt = r["format"]
                    first = r["code"].split("+")[0]
                    if fmt in ("ordinal_scale", "numeric_scale", "count"):
                        v_a, v_b = ordered_stats(gA, gB, first)
                        kind = "mean"
                    elif fmt in ("binary", "mention_flag", "knowledge"):
                        v_a, v_b = share_stats(gA, gB, first, fmt)
                        kind = "share"
                    else:
                        continue
                    trend[f"{country}|{wave}|{demo}|{r['code']}"] = dict(
                        v_a=round(v_a, 3), v_b=round(v_b, 3), stat_kind=kind, significant=bool(r["significant"]))

                t = t_all[t_all["significant"].astype(bool)]
                for _, r in t.iterrows():
                    code = r["code"]
                    first = code.split("+")[0]
                    if first not in cats_idx.index or first not in parsed:
                        continue
                    ch = parse_choices(first, parsed)
                    fmt = r["format"]
                    extra = {}
                    if fmt in ("ordinal_scale", "numeric_scale", "count"):
                        mean_a, mean_b = ordered_stats(gA, gB, first)
                        diff = mean_b - mean_a
                        extra = dict(stat_kind="mean", mean_a=round(mean_a, 3), mean_b=round(mean_b, 3),
                                    lo_label=ch[0][1], hi_label=ch[-1][1])
                    elif fmt in ("binary", "mention_flag", "knowledge"):
                        share_a, share_b = share_stats(gA, gB, first, fmt)
                        diff = share_b - share_a
                        tracked_label = "the correct answer" if fmt == "knowledge" else ch[-1][1]
                        extra = dict(stat_kind="share", share_a=round(share_a, 1), share_b=round(share_b, 1), tracked=tracked_label)
                    elif fmt == "ranked_choice":
                        second = RANKED_SECOND.get(first)
                        gained = int(r["top_gain_option"]) if pd.notna(r["top_gain_option"]) else None
                        label_map = dict(ch)
                        share_a, share_b = ranked_stats(gA, gB, first, second, gained) if (gained is not None and second) else (np.nan, np.nan)
                        diff = share_b - share_a
                        extra = dict(stat_kind="share", share_a=round(share_a, 1), share_b=round(share_b, 1),
                                    tracked=label_map.get(gained, gained))
                    else:  # categorical, forced_choice
                        gained = int(r["top_gain_option"]) if pd.notna(r["top_gain_option"]) else None
                        label_map = dict(ch)
                        share_a, share_b = option_stats(gA, gB, first, gained) if gained is not None else (np.nan, np.nan)
                        diff = share_b - share_a
                        extra = dict(stat_kind="share", share_a=round(share_a, 1), share_b=round(share_b, 1),
                                    tracked=label_map.get(gained, gained))
                    dirn = "up" if diff > 0 else "down"
                    delta_txt = (f'higher for {GROUP_LABEL[demo][names[1]]}' if dirn == "up" else f'higher for {GROUP_LABEL[demo][names[0]]}')
                    rows.append(dict(
                        country=country, name=COUNTRY_NAMES.get(country, country), wave=int(wave), demo=demo,
                        group_a=GROUP_LABEL[demo][names[0]], group_b=GROUP_LABEL[demo][names[1]],
                        code=code, label=cats_idx.loc[first, "label"], fmt=fmt, metric=r["effect_name"],
                        n_a=int(r["n1"]), n_b=int(r["n2"]), effect=round(float(r["effect"]), 4),
                        thr=float(thr.get(fmt, thr["ordinal_scale"])), diff=round(float(diff), 3),
                        p=float(r["p_value"]), dir=dirn, delta_text=delta_txt, **extra))
            print(f"  {country} wave {wave} done")
    return rows, trend


def main():
    rows, trend = build()
    meta = dict(n=len(rows), countries=COUNTRIES, waves=list(WAVES), demos=list(SUBGROUPS),
                config=dict(alpha=ChangeConfig().alpha, min_n=ChangeConfig().min_n))
    OUT.write_text("window.DEMOGRAPHICS=" + json.dumps({"meta": meta, "rows": rows, "trend": trend}, separators=(",", ":"), ensure_ascii=True) + ";\n",
                   encoding="utf-8")
    print(f"wrote {len(rows)} rows and {len(trend)} trend points to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
