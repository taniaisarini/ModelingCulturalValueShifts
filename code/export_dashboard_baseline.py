"""Turn direction-baseline results into the data file used by the dashboard's "LLM baseline" view.

    python code/export_dashboard_baseline.py [results_csv] [out_js]

Defaults: results/direction_baseline/five_country_set/all_gpt-4.1-mini.csv ->
          results/direction_baseline/five_country_set/baseline_results.js
Adds human-readable truth / prediction text, a question group, and how large the true change was relative to the
minimum size that counted as a change (from ChangeConfig).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_detection import ChangeConfig, load_question_info  # noqa: E402
from direction_baseline import COUNTRY_NAMES, LLM_PARAMS, RESULTS_DIR, parse_choices  # noqa: E402

DEFAULT_IN = RESULTS_DIR / "five_country_set" / "all_gpt-4.1-mini.csv"
DEFAULT_OUT = RESULTS_DIR / "five_country_set" / "baseline_results.js"


def thresholds(cfg):
    return {"ordinal_scale": cfg.ordinal_w1, "count": cfg.count_w1, "binary": cfg.binary_pp,
            "mention_flag": cfg.binary_pp, "knowledge": cfg.binary_pp, "categorical": cfg.categorical_tvd,
            "forced_choice": cfg.categorical_tvd, "ranked_choice": cfg.ranked_tvd}


def build_rows(df):
    cats, parsed = load_question_info()
    cats = cats.set_index("code")
    thr = thresholds(ChangeConfig())

    rows = []
    for _, r in df.iterrows():
        first = r["code"].split("+")[0]
        ch = parse_choices(first, parsed)
        label = dict(ch)
        fam, truth, pred = r["family"], r["truth"], r["pred"] if pd.notna(r["pred"]) else None

        def text(v):
            if v is None:
                return "(no answer)"
            if fam == "ordered":
                end = ch[-1][1] if v == "toward_high" else ch[0][1]
                return f'moved toward "{end}"'
            if fam == "share":
                tracked = ch[-1][1] if r["format"] != "knowledge" else None
                if r["format"] == "knowledge":
                    tracked = "the correct answer"
                return f'share answering "{tracked}" {"increased" if v == "increase" else "decreased"}'
            return f'gained the most: "{label.get(int(v), v)}"'

        norm = {"toward_high": "up", "increase": "up", "toward_low": "down", "decrease": "down"}
        title = parsed[first]["parsed_question"].strip().replace("\n", " ")
        short = cats.loc[first, "label"] if first in cats.index else first
        group = cats.loc[first, "family"] if first in cats.index and cats.loc[first, "family"] else "(standalone question)"
        rows.append(dict(
            country=r["country"], name=COUNTRY_NAMES.get(r["country"], r["country"]),
            wa=int(r["wave_a"]), wb=int(r["wave_b"]), ya=str(r["years_a"]), yb=str(r["years_b"]),
            code=r["code"], label=short, question=title, group=group, fmt=r["format"], fam=fam,
            truth=truth, pred=pred, tdir=norm.get(truth), pdir=norm.get(pred),
            truth_text=text(truth), pred_text=text(pred), correct=bool(r["correct"]),
            ratio=round(float(r["effect"]) / thr[r["format"]], 3), nopt=len(ch) if fam == "option" else 2,
            response=r["response"], prompt=r["prompt"], key=r["cache_key"][:12]))

    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else DEFAULT_IN
    out = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    rows = build_rows(pd.read_csv(src))
    meta = dict(model="gpt-4.1-mini", params=LLM_PARAMS, n=len(rows), source=str(src.name))
    out.write_text("window.BASELINE=" + json.dumps({"meta": meta, "rows": rows}, separators=(",", ":"),
                                                   ensure_ascii=True) + ";\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out} ({out.stat().st_size / 1e6:.2f} MB)")


MODELS_OUT = RESULTS_DIR / "five_country_set" / "baseline_models.js"
SLIM = ["pred", "pdir", "pred_text", "correct", "response", "key"]


def export_models():
    """Other models' results for the dashboard's model selector: only what differs from the gpt-4.1-mini rows
    (prompts and question text are shared), matched to them by country|waves|question code."""
    models = {}
    for d in sorted((RESULTS_DIR / "five_country_set" / "by_model").iterdir()):
        name, _, reasoning = d.name.partition("__reasoning-")
        rows = build_rows(pd.read_csv(d / f"all_{d.name}.csv"))
        params = "default temperature, reasoning effort " + ("minimal" if reasoning == "off" else "medium")
        models[d.name] = dict(
            meta=dict(model=name, reasoning=reasoning or None, params=dict(temperature=params), n=len(rows),
                      label=f"{name} (reasoning {reasoning})" if reasoning else name),
            rows={f"{r['country']}|{r['wa']}|{r['wb']}|{r['code']}": {k: r[k] for k in SLIM} for r in rows})
    MODELS_OUT.write_text("window.BASELINE_MODELS=" + json.dumps(models, separators=(",", ":"), ensure_ascii=True) + ";\n",
                          encoding="utf-8")
    print(f"wrote {len(models)} models to {MODELS_OUT} ({MODELS_OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    export_models() if "--models" in sys.argv else main()
