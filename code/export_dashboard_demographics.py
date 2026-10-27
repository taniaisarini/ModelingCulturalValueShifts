"""Data file for the dashboard's "Demographics" tab: for each country, wave, and demographic split (sex, age,
settlement size), which questions show a significant difference in how the two groups answered -- same protocol
as change_detection.py (permutation test + effect-size threshold), just comparing two demographic groups within
one wave instead of two waves.

    python code/export_dashboard_demographics.py

Reads the raw WVS CSV directly (demographic columns aren't in data/question_categories.csv's usual load path).
Writes results/direction_baseline/five_country_set/demographics_data.js (window.DEMOGRAPHICS).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_detection import ChangeConfig, detect_country_changes, load_question_info  # noqa: E402
from demographic_splice import COUNTRIES, SUBGROUPS, load_raw  # noqa: E402
from direction_baseline import COUNTRY_NAMES, RESULTS_DIR, parse_choices  # noqa: E402
from export_dashboard_baseline import thresholds  # noqa: E402

OUT = RESULTS_DIR / "five_country_set" / "demographics_data.js"
WAVES = (5, 6, 7)
GROUP_LABEL = {"sex": {"male": "Male", "female": "Female"},
               "age": {"under45": "Under 45", "45plus": "45 or older"},
               "settlement": {"smaller": "Smaller settlement", "larger": "Larger settlement"}}


def build():
    df = load_raw()
    cats, parsed = load_question_info()
    cats_idx = cats.set_index("code")
    cfg = ChangeConfig()
    thr = thresholds(cfg)

    rows = []
    for country in COUNTRIES:
        for wave in WAVES:
            base = df[(df["COUNTRY_ALPHA"] == country) & (df["S002VS"] == wave)]
            for demo, groups in SUBGROUPS.items():
                names = list(groups.keys())
                gA = base[groups[names[0]](base)].copy()
                gB = base[groups[names[1]](base)].copy()
                if len(gA) < cfg.min_n or len(gB) < cfg.min_n:
                    continue
                gA["S002VS"], gB["S002VS"] = 1, 2
                tmp = pd.concat([gA, gB], ignore_index=True)
                t = detect_country_changes(tmp, country, 1, 2, cfg, cats, parsed,
                                           country_col="COUNTRY_ALPHA", wave_col="S002VS")
                t = t[t["level"].isin(["question", "item"]) & t["significant"].astype(bool)]
                for _, r in t.iterrows():
                    code = r["code"]
                    first = code.split("+")[0]
                    if first not in cats_idx.index or first not in parsed:
                        continue
                    ch = parse_choices(first, parsed)
                    fmt = r["format"]
                    if fmt in ("ordinal_scale", "numeric_scale", "count"):
                        dirn, delta_txt = ("up" if r["signed_change"] > 0 else "down"), None
                        endpoint = ch[-1][1] if dirn == "up" else ch[0][1]
                        delta_txt = f'higher for {GROUP_LABEL[demo][names[1]]}' if dirn == "up" else f'higher for {GROUP_LABEL[demo][names[0]]}'
                    elif fmt in ("binary", "mention_flag", "knowledge"):
                        dirn = "up" if r["signed_change"] > 0 else "down"
                        delta_txt = f'more common for {GROUP_LABEL[demo][names[1]]}' if dirn == "up" else f'more common for {GROUP_LABEL[demo][names[0]]}'
                    else:
                        dirn = "shift"
                        gained = int(r["top_gain_option"]) if pd.notna(r["top_gain_option"]) else None
                        label_map = dict(ch)
                        delta_txt = f'biggest gap: "{label_map.get(gained, gained)}"' if gained is not None else "distribution differs"
                    rows.append(dict(
                        country=country, name=COUNTRY_NAMES.get(country, country), wave=int(wave), demo=demo,
                        group_a=GROUP_LABEL[demo][names[0]], group_b=GROUP_LABEL[demo][names[1]],
                        code=code, label=cats_idx.loc[first, "label"], fmt=fmt, metric=r["effect_name"],
                        n_a=int(r["n1"]), n_b=int(r["n2"]), effect=float(r["effect"]), thr=float(thr.get(fmt, thr["ordinal_scale"])),
                        p=float(r["p_value"]), dir=dirn, delta_text=delta_txt))
            print(f"  {country} wave {wave} done")
    return rows


def main():
    rows = build()
    meta = dict(n=len(rows), countries=COUNTRIES, waves=list(WAVES), demos=list(SUBGROUPS),
                config=dict(alpha=ChangeConfig().alpha, min_n=ChangeConfig().min_n))
    OUT.write_text("window.DEMOGRAPHICS=" + json.dumps({"meta": meta, "rows": rows}, separators=(",", ":"), ensure_ascii=True) + ";\n",
                   encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
