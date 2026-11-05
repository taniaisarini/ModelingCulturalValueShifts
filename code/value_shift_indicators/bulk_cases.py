"""Generic (not hand-authored) case builder for scaling the language-hypothesis experiment across many
countries/questions. Unlike cases.py (hand-written attitude descriptions for ~30 USA questions), this
builds the attitude_description/direction_a/direction_b/direction_labels fields mechanically from the
codebook's own `parsed_question` + `choices` text (data/parsed_survey_data_cleaned.jsonl), using
data/question_categories.csv to restrict to ordinal_scale/numeric_scale/binary/mention_flag questions
(the formats with a meaningful two-ended direction; categorical/ranked_choice/forced_choice questions
don't have one). No hand-written per-question text, so phrasing is mechanical, not tailored.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import build_case  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
USABLE_FORMATS = {"ordinal_scale", "numeric_scale", "binary", "mention_flag"}

COUNTRY_NAMES = {"USA": "United States", "EGY": "Egypt", "THA": "Thailand", "UKR": "Ukraine",
                "URY": "Uruguay", "HKG": "Hong Kong"}

# Real wave 5 / wave 6 fieldwork years per country (from the raw WVS time series file).
FIELDWORK_YEARS = {"USA": (2006, 2011), "EGY": (2008, 2013), "THA": (2007, 2013),
                   "UKR": (2006, 2011), "URY": (2006, 2011), "HKG": (2005, 2014)}


def _load_codebook():
    labels, fmt = {}, {}
    with open(ROOT / "data" / "parsed_survey_data_cleaned.jsonl") as f:
        for line in f:
            d = json.loads(line)
            for code, v in d.items():
                labels[code] = v
    with open(ROOT / "data" / "question_categories.csv") as f:
        for r in csv.DictReader(f):
            fmt[r["code"]] = r["format"]
    return labels, fmt


CODEBOOK, FORMATS = _load_codebook()


def _clean_choice(choice):
    m = re.match(r"^-?\d+\s*\(?(.*?)\)?\s*$", choice.strip())
    return (m.group(1).strip() if m and m.group(1).strip() else choice).rstrip(".")


def usable_question(question):
    """True if this question has a codebook entry, a usable (two-ended) format, and >=2 distinct choices."""
    entry = CODEBOOK.get(question)
    if not entry or not entry.get("choices") or not entry.get("parsed_question"):
        return False
    if FORMATS.get(question) not in USABLE_FORMATS:
        return False
    choices = entry["choices"]
    return len(choices) >= 2 and _clean_choice(choices[0]) != _clean_choice(choices[-1])


def build_generic_case(question, country_alpha):
    """Returns a pipeline.build_case()-shaped dict, built mechanically from the codebook. Raises
    KeyError/ValueError if `question` isn't usable_question() or `country_alpha` isn't in FIELDWORK_YEARS."""
    entry = CODEBOOK[question]
    choices = entry["choices"]
    low, high = _clean_choice(choices[0]), _clean_choice(choices[-1])
    year_a, year_b = FIELDWORK_YEARS[country_alpha]
    question_text = entry["parsed_question"].strip()
    if question_text and question_text[-1] not in ".?!":
        question_text += "."
    attitude_description = (
        f"{question_text} A nationally representative survey asked this question at both "
        f'time points, on a scale running from "{low}" at one end to "{high}" at the other.')
    direction_a = f'average answer moved toward "{high}".'
    direction_b = f'average answer moved toward "{low}".'
    direction_labels = {"A": f'answers moved toward "{high}"', "B": f'answers moved toward "{low}"'}
    return build_case(COUNTRY_NAMES[country_alpha], year_a, year_b, attitude_description,
                      direction_a, direction_b, direction_labels)
