"""Rule-based first pass at categorizing WVS questions by answer format.

Output: data/question_categories.csv with columns
  code, format, family, group, n_choices, label

`format` says how the answers should be handled statistically:
  identifier      - no answer choices (country codes, ids, open codes); drop
  mention_flag    - 0/1 "mentioned / not mentioned" item, part of a select-many group
  binary          - yes/no, or two-option items
  ordinal_scale   - ordered labeled scale (agree, trust, importance, frequency, satisfaction...)
  numeric_scale   - 1-10 style scale with only endpoints labeled
  categorical     - unordered options (party, religion, occupation...); means are meaningless
  forced_choice   - pick one of two opposed positions (+ both/neither/other); compare shares
  ranked_choice   - "first choice"/"second choice" pairs over the same option list; analyze together
  knowledge       - factual quiz item with a correct answer
  count           - a number (e.g. ideal number of children)
  unclassified    - needs a manual look
`group` names the select-many battery a mention_flag belongs to.
`family` groups items that share a label prefix ("Neighbours: ...", "Trust: ...").
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "parsed_survey_data_cleaned.jsonl"
OUT = ROOT / "data" / "question_categories.csv"

ORDINAL_WORDS = re.compile(
    r"agree|trust|confiden|important|satisf|often|never|frequen|daily|weekly|"
    r"justif|much|great deal|not at all|very|somewhat|fairly|better|worse|"
    r"strongly|approve|likely|proud|happy|interested|good|bad",
    re.I,
)
MENTION = re.compile(r"mentioned|not used|used last week|not relevant|^0 \(?not|^1 \(?(mentioned|used)", re.I)
YESNO = re.compile(r"\b(yes|no|true|false)\b", re.I)


def choice_texts(choices):
    return [re.sub(r"^\s*-?\d+\s*", "", str(c)).strip(" ()") for c in choices]


def classify(code, v):
    choices = [c for c in v["choices"] if not str(c).strip().startswith("-")]  # drop missing-value codes
    texts = choice_texts(choices)
    joined = " | ".join(texts)
    n = len(choices)

    if n == 0:
        return "identifier"
    if n == 2 and MENTION.search(" ".join(map(str, choices))):
        return "mention_flag"
    if n == 2:
        return "binary"
    if n == 3 and MENTION.search(" ".join(map(str, choices))):
        return "mention_flag"
    # 1-10 scales with only endpoints labeled: many numbered choices, few with words
    if n >= 6 and sum(bool(re.search(r"[A-Za-z]{3,}", t)) for t in texts) <= 3:
        return "numeric_scale"
    if 3 <= n <= 11 and ORDINAL_WORDS.search(joined):
        # long lists of distinct nouns are categorical even if a word like "good" appears
        avg_len = sum(len(t) for t in texts) / n
        if not (n >= 6 and avg_len > 25):
            return "ordinal_scale"
    if n >= 3:
        return "categorical"
    return "unclassified"


# Manual corrections from hand-checking the rule-based output (categorical bucket).
OVERRIDES = {}
for _c in ["E269", "E270", "E271", "E272", "E273", "E196", "E242", "E243", "E244", "E245", "E246",
           "H009", "H010", "H011", "E129A", "E130", "E188", "G018", "E143"]:
    OVERRIDES[_c] = "ordinal_scale"
for _c in ["A026", "A169", "B008", "B009", "B016", "B017", "C061", "E022", "E032", "E034", "E131",
           "F034", "F062", "F201", "D025", "E211", "F189"]:
    OVERRIDES[_c] = "forced_choice"
for _c in ["A044", "A045", "E001", "E002", "E005", "E006", "E190", "E191", "E238", "E239", "E240",
           "E241", "G001", "G002", "G024", "G025"]:
    OVERRIDES[_c] = "ranked_choice"
for _c in ["E279", "E280", "E281"]:
    OVERRIDES[_c] = "knowledge"
OVERRIDES["D017"] = "count"
OVERRIDES["C008"] = "numeric_scale"
# Second pass: full sweep over ordinal / binary / numeric / identifier / mention_flag buckets.
for _c in ["C009", "C010", "E003", "E004"]:
    OVERRIDES[_c] = "ranked_choice"
for _c in ["E062", "E132", "F022", "F200", "E118", "E119", "E128", "F188", "H008_07", "H008_08", "A025"]:
    OVERRIDES[_c] = "forced_choice"
# Agree/Disagree/Neither coded 1/2/3 (or with a non-ordered extra option): "Neither" is not the extreme
# end, so a mean over the codes is wrong. Compare answer shares instead.
for _c in ["C001", "C002", "C004", "C005", "F004", "F005", "F006", "F007", "F008", "F009", "F010",
           "D023", "D024", "E204", "F176", "E263", "E264"]:
    OVERRIDES[_c] = "categorical"
# Nominal variables with no listed choices in the codebook (country-specific codes).
for _c in ["E179WVS", "E180WVS", "E182", "F025_WVS", "G015", "G015B", "G016", "G026_01", "G027_01", "G003CS"]:
    OVERRIDES[_c] = "categorical"
# Select-many batteries stored as one 0/1 column per option.
for _c in ["D003", "D004", "D005", "D006", "D007", "D008", "D009",
           "D010", "D011", "D012", "D013", "D014", "D015", "D016",
           "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"]:
    OVERRIDES[_c] = "mention_flag"
# Membership: 0 not / 1 inactive / 2 active is ordinal; immigrant-effect items are ordinal too.
for _c in ["A098", "A099", "A100", "A101", "A102", "A103", "A104", "A105", "A106", "A106B", "A106C",
           "A106D", "A174"]:
    OVERRIDES[_c] = "ordinal_scale"
# Middle code is not in scale order ("Hard to say" between Not relevant / Relevant; "only in times of
# crisis" between hardly ever / never), so means over the codes would mislead -> categorical.
for _c in ["G053", "G054", "G055", "G056", "G057", "G058", "G059", "G060", "F067"]:
    OVERRIDES[_c] = "categorical"
for _c in ["C059", "D019", "G027A", "G027B"]:
    OVERRIDES[_c] = "binary"
# Final sweep: three or more distinct positions, not two opposed ones -> categorical.
for _c in ["F189", "F034", "D025", "E034", "E211", "F062"]:
    OVERRIDES[_c] = "categorical"
OVERRIDES["G001CS"] = OVERRIDES["G002CS"] = "identifier"  # country-specific codes, not comparable


def mention_group(code, label):
    """Group id for select-many (mention_flag) items so a battery is analysed as one unit."""
    if code.startswith("A124_"):
        return "neighbours"
    if re.fullmatch(r"A0(8\d|9[0-7])[A-C]?", code):
        return "voluntary_work"
    if re.fullmatch(r"A0(6[4-9]|7\d|80)[A-C]?", code) or code.startswith("A080_"):
        return "org_membership"
    if re.fullmatch(r"C0(1[1-9]|2[0-5])", code):
        return "job_qualities"
    if code.startswith("C042B"):
        return "why_people_work"
    if re.fullmatch(r"E0(4[89]|5[0-6])", code):
        return "personal_characteristics"
    if re.fullmatch(r"E2(4[89]|5[0-4]|5[89]|6[0-2])", code):
        return "info_source_last_week"
    if re.fullmatch(r"D0(0[3-9]|1[0-6])", code):
        return "sharing_attitudes"
    if code.startswith("A027_"):
        return "child_qualities"
    return ""


def family(label):
    return label.split(":")[0].strip() if ":" in label else ""


def main():
    d = {}
    for line in open(SRC):
        d.update(json.loads(line))
    rows = []
    for code, v in d.items():
        rows.append(
            dict(
                code=code,
                format=OVERRIDES.get(code) or classify(code, v),
                family=family(v["label"]),
                group=mention_group(code, v["label"]) if (OVERRIDES.get(code) or classify(code, v)) == "mention_flag" else "",
                n_choices=len(v["choices"]),
                label=v["label"],
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df["format"].value_counts().to_string())
    return df


if __name__ == "__main__":
    main()
