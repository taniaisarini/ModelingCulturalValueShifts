"""Does the LLM's predicted direction (already collected in results/direction_baseline/five_country_set/
all_gpt-4.1-mini.csv) align better with any ONE demographic subgroup's actual answer shift than with the whole
population's? No new LLM calls -- reuses change_detection.py's existing significance-test machinery (the same
effect-size + permutation protocol used for the overall-population "truth"), just run on data filtered to each
subgroup.

    python code/demographic_splice.py

Demographics, coarsened to 2 groups each for statistical power (country-wave samples here are ~1-2.5k respondents;
change_detection.ChangeConfig.min_n = 200 minimum per wave per test):
  sex        X001: 1=male, 2=female (native 2-group, no coarsening needed)
  age        X003R (WVS's own 6 age bins) coarsened to <45 (bins 1-3) / >=45 (bins 4-6)
  settlement X049 (WVS's own 8 settlement-size bins) coarsened to smaller (bins 1-4) / larger (bins 5-8); skipped
             for a country-wave with < 50% valid X049 (some don't collect it at all)

Output: gdelt_plots/demographic_splice/{subgroup_truth_long.csv, llm_alignment_summary.csv, llm_vs_subgroup_flips.csv}
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_detection import ChangeConfig, detect_country_changes, load_question_info  # noqa: E402
from direction_baseline import DATA_PATH, truth_label  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "gdelt_plots" / "demographic_splice"
RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "direction_baseline" / "five_country_set" / "all_gpt-4.1-mini.csv"
COUNTRIES = ["EGY", "HKG", "THA", "UKR", "URY"]
WAVE_PAIRS = [(5, 6), (6, 7)]

SUBGROUPS = {
    "sex": {"male": lambda df: df["X001"] == 1, "female": lambda df: df["X001"] == 2},
    "age": {"under45": lambda df: df["X003R"].between(1, 3), "45plus": lambda df: df["X003R"].between(4, 6)},
    "settlement": {"smaller": lambda df: df["X049"].between(1, 4), "larger": lambda df: df["X049"].between(5, 8)},
}
MIN_VALID_FRAC = 0.5  # skip a demographic variable for a country-wave if less than half the rows have it


def load_raw():
    cats, _ = load_question_info()
    need = {"S002VS", "COUNTRY_ALPHA", "S020", "X001", "X003R", "X049"} | set(cats["code"])
    header = pd.read_csv(DATA_PATH, nrows=0).columns
    cols = [c for c in header if c in need]
    parts = []
    for chunk in pd.read_csv(DATA_PATH, usecols=cols, chunksize=200_000, low_memory=False):
        parts.append(chunk[chunk["COUNTRY_ALPHA"].isin(COUNTRIES) & chunk["S002VS"].isin([5, 6, 7])])
    df = pd.concat(parts, ignore_index=True)
    for c in ("X001", "X003R", "X049"):
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda v: v >= 0)
    return df


def subgroup_truth(df, cfg, cats, parsed, country, wa, wb, demo, mask):
    sub = df[mask]
    if (sub[(sub["COUNTRY_ALPHA"] == country) & (sub["S002VS"] == wa)].shape[0] < cfg.min_n or
            sub[(sub["COUNTRY_ALPHA"] == country) & (sub["S002VS"] == wb)].shape[0] < cfg.min_n):
        return pd.DataFrame()
    t = detect_country_changes(sub, country, wa, wb, cfg, cats, parsed)
    t = t[t["level"].isin(["question", "item"])]
    t["truth_label"] = t.apply(truth_label, axis=1)
    t["demo"] = demo
    return t[["code", "n1", "n2", "significant", "truth_label", "demo"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_CSV)
    cats, parsed = load_question_info()
    cfg = ChangeConfig()
    print("loading raw WVS data for the 5 countries...")
    df = load_raw()

    rows = []
    for country in COUNTRIES:
        for wa, wb in WAVE_PAIRS:
            for demo, groups in SUBGROUPS.items():
                if demo == "settlement":
                    cw = df[(df["COUNTRY_ALPHA"] == country) & (df["S002VS"].isin([wa, wb]))]
                    if cw["X049"].notna().mean() < MIN_VALID_FRAC:
                        continue
                for sub_name, pred in groups.items():
                    mask = pred(df)
                    t = subgroup_truth(df, cfg, cats, parsed, country, wa, wb, f"{demo}:{sub_name}", mask)
                    if not t.empty:
                        t["country"], t["wave_a"], t["wave_b"] = country, wa, wb
                        rows.append(t)
            print(f"  {country} {wa}->{wb} done")
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(OUT / "subgroup_truth_long.csv", index=False)
    print(f"subgroup truth rows: {len(long)}")

    # merge LLM predictions against every subgroup's truth for the same (country, wave_a, wave_b, code)
    m = results.merge(long, on=["country", "wave_a", "wave_b", "code"], how="inner", suffixes=("", "_sub"))
    m["matches_subgroup"] = m["significant"] & (m["pred"] == m["truth_label"])
    m["matches_population"] = m["correct"]  # already computed against the population truth in the results csv

    summary = (m[m["significant"]].groupby("demo")
               .agg(n=("matches_subgroup", "size"), subgroup_accuracy=("matches_subgroup", "mean"))
               .reset_index())
    pop_acc = results["correct"].mean()
    summary["population_accuracy_same_rows_n"] = len(results)
    summary["population_accuracy"] = pop_acc
    summary = summary.sort_values("subgroup_accuracy", ascending=False)
    summary.to_csv(OUT / "llm_alignment_summary.csv", index=False)
    print(summary.to_string(index=False))

    # the interesting case: LLM wrong about the population, but right about one particular subgroup
    flips = m[(~m["matches_population"]) & m["matches_subgroup"]]
    flips[["country", "wave_a", "wave_b", "code", "pred", "truth", "demo", "truth_label"]].to_csv(
        OUT / "llm_vs_subgroup_flips.csv", index=False)
    print(f"\nrows wrong vs. population but right vs. a subgroup: {len(flips)} / {len(results)} total LLM rows")
    print(flips["demo"].value_counts().to_string())


if __name__ == "__main__":
    main()
