"""Direction baseline: can an LLM predict which way a country's answers moved between two WVS waves?

Input : a country, a survey question, and the survey years of wave A and wave B (looked up from the data).
Task  : predict the direction of change. The prompt depends on the question's format:
  ordered  (ordinal scales, numeric scales, counts): did the typical answer move toward one end of the
           scale or the other, or stay about the same?
  share    (yes/no, select-many items, knowledge)  : did the share giving one answer go up, down, or stay?
  option   (categorical, forced choice, ranked)    : which answer option gained the most share, or none?
Ground truth comes from change_detection.py: a change only counts if it is significant under that protocol,
otherwise the truth is "none".  All LLM calls go through fast_llm.

    python code/direction_baseline.py --country EGY --wave-a 5 --wave-b 6 --limit 100
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_detection import (CHILD_QUALITY_CODE, KNOWLEDGE_CORRECT, OFF_SCALE, ChangeConfig,  # noqa: E402
                              detect_country_changes, load_question_info)
from fast_llm import run_llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WVS_Time_Series_1981-2022_csv_v5_0.csv"
PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = {  # prompt name -> file under code/prompts/
    "system": "system/direction_baseline/all_question_types.txt",
    "ordered": "user/direction_baseline/scale_and_count.txt",
    "share": "user/direction_baseline/yes_no_and_select_many.txt",
    "option": "user/direction_baseline/categorical_and_ranked_choice.txt",
}
RESULTS_DIR = ROOT / "results" / "direction_baseline"
LLM_PARAMS = dict(temperature=0.0, max_tokens=250)   # part of the cache key: change these and calls are re-run

FAMILY = {"ordinal_scale": "ordered", "count": "ordered",
          "binary": "share", "mention_flag": "share", "knowledge": "share",
          "categorical": "option", "forced_choice": "option", "ranked_choice": "option"}

COUNTRY_NAMES = {
    "ALB": "Albania", "AND": "Andorra", "ARG": "Argentina", "ARM": "Armenia", "AUS": "Australia",
    "AZE": "Azerbaijan", "BFA": "Burkina Faso", "BGD": "Bangladesh", "BGR": "Bulgaria",
    "BIH": "Bosnia and Herzegovina", "BLR": "Belarus", "BOL": "Bolivia", "BRA": "Brazil", "CAN": "Canada",
    "CHE": "Switzerland", "CHL": "Chile", "CHN": "China", "COL": "Colombia", "CYP": "Cyprus",
    "CZE": "Czech Republic", "DEU": "Germany", "DOM": "Dominican Republic", "DZA": "Algeria",
    "ECU": "Ecuador", "EGY": "Egypt", "ESP": "Spain", "EST": "Estonia", "ETH": "Ethiopia", "FIN": "Finland",
    "FRA": "France", "GBR": "Great Britain", "GEO": "Georgia", "GHA": "Ghana", "GRC": "Greece",
    "GTM": "Guatemala", "HKG": "Hong Kong", "HRV": "Croatia", "HTI": "Haiti", "HUN": "Hungary",
    "IDN": "Indonesia", "IND": "India", "IRN": "Iran", "IRQ": "Iraq", "ISR": "Israel", "ITA": "Italy",
    "JOR": "Jordan", "JPN": "Japan", "KAZ": "Kazakhstan", "KEN": "Kenya", "KGZ": "Kyrgyzstan",
    "KOR": "South Korea", "KWT": "Kuwait", "LBN": "Lebanon", "LBY": "Libya", "LTU": "Lithuania",
    "LVA": "Latvia", "MAC": "Macao", "MAR": "Morocco", "MDA": "Moldova", "MDV": "Maldives",
    "MEX": "Mexico", "MKD": "North Macedonia", "MLI": "Mali", "MMR": "Myanmar", "MNE": "Montenegro",
    "MNG": "Mongolia", "MYS": "Malaysia", "NGA": "Nigeria", "NIC": "Nicaragua", "NIR": "Northern Ireland",
    "NLD": "Netherlands", "NOR": "Norway", "NZL": "New Zealand", "PAK": "Pakistan", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRI": "Puerto Rico", "PSE": "Palestine", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "RWA": "Rwanda", "SAU": "Saudi Arabia", "SGP": "Singapore",
    "SLV": "El Salvador", "SRB": "Serbia", "SVK": "Slovakia", "SVN": "Slovenia", "SWE": "Sweden",
    "THA": "Thailand", "TJK": "Tajikistan", "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TUR": "Turkey",
    "TWN": "Taiwan", "TZA": "Tanzania", "UGA": "Uganda", "UKR": "Ukraine", "URY": "Uruguay",
    "USA": "United States", "UZB": "Uzbekistan", "VEN": "Venezuela", "VNM": "Vietnam", "YEM": "Yemen",
    "ZAF": "South Africa", "ZMB": "Zambia", "ZWE": "Zimbabwe",
}


def resolve_country(text):
    """Accept an alpha code ('EGY') or a name ('Egypt'); returns the alpha code."""
    t = text.strip()
    if t.upper() in COUNTRY_NAMES:
        return t.upper()
    for code, name in COUNTRY_NAMES.items():
        if name.lower() == t.lower():
            return code
    raise ValueError(f"unknown country {text!r}")


# ------------------------------------------------------------------ data

def load_countries_waves(countries, waves, path=DATA_PATH):
    """Reads only the rows for these countries and waves (the full file is large), in a single pass."""
    cats, _ = load_question_info()
    need = {"S002VS", "COUNTRY_ALPHA", "S020"} | set(cats["code"]) | {
        "A027", "A029", "A030", "A032", "A034", "A035", "A038", "A039", "A040", "A041", "A042", "A043B"}
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in header if c in need]
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=100_000, low_memory=False):
        parts.append(chunk[chunk["COUNTRY_ALPHA"].isin(countries) & chunk["S002VS"].isin(waves)])
    return pd.concat(parts, ignore_index=True)


def load_country_waves(country, waves, path=DATA_PATH):
    return load_countries_waves([country], waves, path)


def survey_years(df, country, wave):
    """Fieldwork year(s) of a wave in a country, e.g. '2008' or '2008-2009'."""
    y = df[(df["COUNTRY_ALPHA"] == country) & (df["S002VS"] == wave)]["S020"].dropna().astype(int)
    if y.empty:
        raise ValueError(f"no data for {country} wave {wave}")
    lo, hi = int(y.min()), int(y.max())
    return str(lo) if lo == hi else f"{lo}-{hi}"


# ------------------------------------------------------------------ prompt construction

def parse_choices(code, parsed):
    """[(numeric code, label)] for the on-scale answer options of a question."""
    out = []
    for c in parsed.get(code, {}).get("choices", []):
        m = re.match(r"\s*(-?\d+)\s*(.*)$", str(c))
        if not m:
            continue
        n = int(m.group(1))
        if n < 0 or n in OFF_SCALE.get(code, []):
            continue
        label = re.sub(r"^\((.*)\)$", r"\1", m.group(2).strip()).strip()
        out.append((n, label if label and label != str(n) else str(n)))
    return out


def _describe_scale(ch):
    labeled = [x for x in ch if x[1] != str(x[0])]
    if len(ch) > 6 and len(labeled) <= 3:
        lo, hi = ch[0], ch[-1]
        return f"a {len(ch)}-point scale from {lo[0]} ({lo[1]}) to {hi[0]} ({hi[1]})"
    return "; ".join(f"{n} = {label}" for n, label in ch)


def _seed(code):
    return int(hashlib.md5(code.encode()).hexdigest(), 16)


def _template(name):
    return (PROMPT_ROOT / PROMPT_FILES[name]).read_text().strip()


def build_prompt(row, parsed, country_name, years_a, years_b, allow_none=False, flip=False):
    """Returns (family, user_prompt, meta) where meta lets us decode the model's letter answer.

    allow_none: also offer a "stayed about the same" answer (only sensible when unchanged questions are included).
    flip: reverse the order of the answer options relative to the default (A/B swapped for scale and yes/no questions;
          the numbered option list reversed for pick-one questions). Wording is otherwise identical.
    """
    code, fmt = row["code"], row["format"]
    family = FAMILY[fmt]
    first = code.split("+")[0]
    question = parsed[first]["parsed_question"].strip()
    ch = parse_choices(first, parsed)
    seed = _seed(code)
    extra = "\n(Each person picked a first and a second choice; count both.)" if fmt == "ranked_choice" else ""
    common = dict(country=country_name, question=question, years_a=years_a, years_b=years_b, extra=extra)

    if family == "ordered":
        low, high = ch[0][1], ch[-1][1]
        pair = [("toward_low", low), ("toward_high", high)]
        if (seed % 2 == 1) != flip:
            pair.reverse()
        prompt = _template("ordered").format(scale=_describe_scale(ch), opt_a=pair[0][1], opt_b=pair[1][1],
                                             none_line="\nC) It stayed about the same" if allow_none else "",
                                             letters="A, B or C" if allow_none else "A or B", **common)
        return family, prompt, _ab(pair, allow_none)

    if family == "share":
        if fmt == "knowledge":
            tracked = dict(ch)[KNOWLEDGE_CORRECT[code]]
        else:
            tracked = ch[-1][1]  # the higher code; change_detection measures the share giving it
        pair = [("increase", "increased"), ("decrease", "decreased")]
        if (seed % 2 == 1) != flip:
            pair.reverse()
        prompt = _template("share").format(choices="; ".join(label for _, label in ch), tracked=tracked,
                                           opt_a=pair[0][1], opt_b=pair[1][1],
                                           none_line="\nC) The share stayed about the same" if allow_none else "",
                                           letters="A, B or C" if allow_none else "A or B", **common)
        return family, prompt, _ab(pair, allow_none)

    # option family: shuffle the options (deterministically) and add a final "no meaningful change"
    order = list(range(len(ch)))
    np.random.default_rng(seed % (2**32)).shuffle(order)
    if flip:
        order.reverse()
    labels = [str(i + 1) for i in range(len(ch) + 1)]   # numbers, so they cannot clash with A/B/C inside the question
    lines = [f"{labels[i]}: {ch[j][1]}" for i, j in enumerate(order)]
    if allow_none:
        lines.append(f"{labels[len(ch)]}: No meaningful change")
    hint = " If nothing changed meaningfully, choose the last option." if allow_none else ""
    prompt = _template("option").format(option_lines="\n".join(lines), none_hint=hint, **common)
    meta = {labels[i]: str(ch[j][0]) for i, j in enumerate(order)}
    if allow_none:
        meta[labels[len(ch)]] = "none"
    return family, prompt, meta


def _ab(pair, allow_none):
    meta = {"A": pair[0][0], "B": pair[1][0]}
    if allow_none:
        meta["C"] = "none"
    return meta


def parse_answer(text, meta):
    """The model's final 'ANSWER: X' letter, decoded via `meta`; None if missing or invalid."""
    if not text:
        return None
    found = re.findall(r"ANSWER:\s*\**\s*([A-Za-z]|\d+)\b", text)
    return meta.get(found[-1].upper()) if found else None


# ------------------------------------------------------------------ ground truth

def truth_label(row):
    """Decode the change_detection row into the same label space the model answers in."""
    if not bool(row["significant"]):
        return "none"
    family = FAMILY[row["format"]]
    if family == "ordered":
        return "toward_high" if row["signed_change"] > 0 else "toward_low"
    if family == "share":
        return "increase" if row["signed_change"] > 0 else "decrease"
    return str(int(row["top_gain_option"]))


# ------------------------------------------------------------------ run + score

def prepare_direction_baseline(country, wave_a, wave_b, df, limit=None, seed=0, questions=None,
                               only_changed=True, cfg=None, flip=False):
    """Find the questions to run and build one prompt per question. No LLM calls.

    Returns a dict with the ground-truth rows (`truth`), the requests for fast_llm (`reqs`) and what is needed
    to decode and score the answers later (`metas`).
    questions: optional list of question codes to run (default: every question testable in both waves).
    limit    : optional random subsample of questions (for quick tests).
    flip     : reverse the order of the answer options in every prompt (see build_prompt).
    only_changed: keep only questions whose answers changed significantly (the default); the prompts then do not
               offer a "stayed about the same" answer. Set False to include unchanged questions and offer it.
    """
    country = resolve_country(country)
    cfg = cfg or ChangeConfig()
    cats, parsed = load_question_info()
    years_a, years_b = survey_years(df, country, wave_a), survey_years(df, country, wave_b)

    truth = detect_country_changes(df, country, wave_a, wave_b, cfg, cats, parsed)
    truth = truth[truth["level"].isin(["question", "item"]) & truth["format"].isin(FAMILY)]
    truth = truth[truth["code"] != CHILD_QUALITY_CODE]
    # skip questions with no parsed text (e.g. the child-qualities sub-items, merged into one combined question)
    truth = truth[truth["code"].map(lambda c: c.split("+")[0] in parsed)].reset_index(drop=True)
    if only_changed:
        truth = truth[truth["significant"].astype(bool)]
    if questions is not None:
        truth = truth[truth["code"].isin(questions)]
    if limit and len(truth) > limit:
        truth = truth.sample(limit, random_state=seed)
    truth = truth.reset_index(drop=True)

    name = COUNTRY_NAMES.get(country, country)
    reqs, metas = [], {}
    for _, r in truth.iterrows():
        family, prompt, meta = build_prompt(r, parsed, name, years_a, years_b, allow_none=not only_changed, flip=flip)
        # the prompt hash keeps ids unique per prompt
        rid = f"{country}_{wave_a}_{wave_b}_{r['code']}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
        reqs.append({"id": rid, "system": _template("system"), "user": prompt})
        metas[rid] = (family, prompt, meta)
    return dict(country=country, wave_a=wave_a, wave_b=wave_b, years_a=years_a, years_b=years_b,
                truth=truth, reqs=reqs, metas=metas)


def assemble_direction_results(prep, res):
    """Decode the model's answers (`res` from fast_llm.run_llm) and attach ground truth. Returns a DataFrame."""
    rows = []
    for (_, r), q in zip(prep["truth"].iterrows(), prep["reqs"]):
        family, prompt, meta = prep["metas"][q["id"]]
        out = res[q["id"]]
        pred = parse_answer(out["text"], meta)
        t = truth_label(r)
        rows.append(dict(country=prep["country"], wave_a=prep["wave_a"], wave_b=prep["wave_b"],
                         years_a=prep["years_a"], years_b=prep["years_b"],
                         code=r["code"], format=r["format"], family=family, significant=bool(r["significant"]),
                         effect=r["effect"], truth=t, pred=pred, correct=(pred == t) if pred else False,
                         parsed=pred is not None, error=out["error"], response=out["text"],
                         finish_reason=out.get("finish_reason"), cached=out.get("cached"),
                         cache_key=out.get("cache_key"), prompt=prompt))
    return pd.DataFrame(rows)


def run_direction_baseline(country, wave_a, wave_b, model="gpt-4.1-mini", df=None, **prep_kwargs):
    """Prepare, call the LLM, and assemble results for one country and wave pair (see prepare_...)."""
    country = resolve_country(country)
    if df is None:
        df = load_country_waves(country, [wave_a, wave_b])
    prep = prepare_direction_baseline(country, wave_a, wave_b, df, **prep_kwargs)
    res = run_llm(prep["reqs"], model=model, **LLM_PARAMS)   # cached calls are not re-sent
    return assemble_direction_results(prep, res)


def score(res):
    """Accuracy by family, against simple reference strategies (always 'none', and most common truth)."""
    def block(g):
        n = len(g)
        real = g[g["truth"] != "none"]
        return pd.Series({
            "n": n,
            "parsed": g["parsed"].mean(),
            "accuracy": g["correct"].mean(),
            "always_none": (g["truth"] == "none").mean(),
            "majority": g["truth"].value_counts(normalize=True).iloc[0],
            "acc_when_real_change": real["correct"].mean() if len(real) else np.nan,
            "n_real_change": len(real),
        })
    out = pd.DataFrame({fam: block(g) for fam, g in res.groupby("family")}).T
    out.loc["ALL"] = block(res)
    out["n"] = out["n"].astype(int)
    out["n_real_change"] = out["n_real_change"].astype(int)
    return out.round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, help="alpha code (EGY) or name (Egypt)")
    ap.add_argument("--wave-a", type=int, required=True)
    ap.add_argument("--wave-b", type=int, required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=None, help="random subsample of questions")
    ap.add_argument("--questions", nargs="*", default=None, help="only these question codes")
    ap.add_argument("--out", default=None, help="CSV path for the full results")
    a = ap.parse_args()
    res = run_direction_baseline(a.country, a.wave_a, a.wave_b, model=a.model, limit=a.limit,
                                 questions=a.questions)
    out = a.out or str(RESULTS_DIR / f"{resolve_country(a.country)}_{a.wave_a}_{a.wave_b}_{a.model}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out}\n")
    print(score(res).to_string())


if __name__ == "__main__":
    main()
