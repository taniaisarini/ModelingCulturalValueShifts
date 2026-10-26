"""Group the baseline's questions into broad themes (religion, family and gender, confidence in institutions, ...).

    from question_themes import theme_of
    theme_of("F028", "How often do you attend religious services")   # -> "Religion and faith"

The groups are hand-made from question codes and titles, for reading related questions side by side; they are not an
official WVS taxonomy. Questions made of two codes (E001+E002) use the first code. Rules are tried in order.
"""
import re

# (theme, regex on the first code, regex on the label); a question matches when either regex matches
RULES = [
    ("Religion and faith", r"^(A006|A098|F025|F028B?|F034|F050|F063|F200|F202|F203|E220)$", r"^Important in life: Religion"),
    ("Science and technology", r"^(E217|E218|E234|I001|I002)$", r"technology"),
    ("Media and information", r"^(E248B|E255)", r"^Information source"),
    ("Gender and family", r"^(D054|D057|D059|D060|D061|D066_B|D078|D001_B|C001)", r"^Important in life: Family"),
    ("Confidence in institutions", r"^E069_", r"^Confidence:"),
    ("Moral attitudes", r"^F11[4-9]|^F12|^F13", r"^Justifiable"),
    ("Civic membership", r"^A(099|10[0-6])", r"^Active/Inactive membership"),
    ("Security and hardship", r"^H0", r"^(Secure in|Frequency|Worries|Things done for reasons of security|Respondent)"),
    ("National identity", r"^(G006|G019|G02[0-7]|E012)", r"^I see myself|^How proud"),
    ("Trust and social ties", r"^(A165|A168A|G007|A124)", r"^Important in life: Friends|^Neighbours|^Trust"),
    ("Economy and work", r"^(B008|C002|C006|E035|E036|E037|E039|E040|E041)$", r"^Important in life: Work|Future changes: Less importance placed on work"),
    ("Politics and democracy", r"^(E00[1-6]|E02[3-8]|E033|E11[4-7]|E124|E22[2-9]|E233|E23[56]|E26[3-5])", r"^Important in life: Politics|Future changes: Greater respect for authority|^Democracy:|^Political (system|action)|^Vote in"),
    ("Wellbeing and values", r"^(A008|A009|A170|A173|A189|F001)", r"^Important in life: Leisure|^Schwartz"),
]
THEMES = [t for t, _, _ in RULES]


def theme_of(code, label=""):
    first = code.split("+")[0]
    for theme, code_re, label_re in RULES:
        if re.search(code_re, first) or (label and re.search(label_re, label)):
            return theme
    return "Other"
