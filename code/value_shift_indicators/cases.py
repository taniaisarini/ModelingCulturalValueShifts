"""The 10 USA wave5->6 (2006->2011) test cases (see STATUS.md/run_pipeline.py for how they were picked:
large, clean, significant change_detection.py results on simple ordinal/binary scales).

Each case's `question` + `section` point at the ground-truth row in
data/significant_changes/ks/<section>_wave5_to_wave6.csv (read only for the final report, never shown
to the model). `true_letter_if_mean_increases` records, for bookkeeping, which of our hand-written
A/B directions corresponds to the mean going up vs down -- set per question because some scales are
coded with 1 = the more "active"/accepting end and others the reverse.
"""
from pipeline import build_case

CASES = [
    dict(
        question="F118", section="F",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how justifiable people consider homosexuality, on a scale from 1 '
                                   '(never justifiable) to 10 (always justifiable). A nationally '
                                   'representative survey asked this question at both time points.'),
            direction_a='average answer moved toward "always justifiable" (attitudes became more accepting).',
            direction_b='average answer moved toward "never justifiable" (attitudes became less accepting).',
            direction_labels={"A": "attitudes became more accepting of homosexuality",
                               "B": "attitudes became less accepting of homosexuality"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="A106", section="A",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people are a member of, and how active they are in, a '
                                   'voluntary organization not covered by other specific categories '
                                   '(such as clubs or civic groups), on a scale from 0 (not a member) to '
                                   '2 (active member). A nationally representative survey asked this '
                                   'question at both time points.'),
            direction_a='average answer moved toward "active member" (civic participation increased).',
            direction_b='average answer moved toward "not a member" (civic participation decreased).',
            direction_labels={"A": "civic participation in this kind of voluntary organization increased",
                               "B": "civic participation in this kind of voluntary organization decreased"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E222", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people have personally taken a form of political action not '
                                   'covered by other specific categories, on a scale from 1 (have done '
                                   'this) to 3 (would never do this). A nationally representative survey '
                                   'asked this question at both time points.'),
            direction_a='average answer moved toward "have done this" (people became more willing to take this kind of political action).',
            direction_b='average answer moved toward "would never do this" (people became less willing to take this kind of political action).',
            direction_labels={"A": "people became more willing to take this kind of political action",
                               "B": "people became less willing to take this kind of political action"},
        ),
        direction_if_mean_increases="B",
    ),
    dict(
        question="E035", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people think incomes should be made more equal or that larger '
                                   'income differences are needed as incentives, on a scale from 1 '
                                   '(incomes should be made more equal) to 10 (we need larger income '
                                   'differences as incentives). A nationally representative survey asked '
                                   'this question at both time points.'),
            direction_a='average answer moved toward "we need larger income differences as incentives" (people became more accepting of income inequality).',
            direction_b='average answer moved toward "incomes should be made more equal" (people became more in favor of income equality).',
            direction_labels={"A": "people became more accepting of income inequality",
                               "B": "people became more in favor of income equality"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E229", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people see civil rights that protect individual liberty '
                                   'against oppression as an essential characteristic of democracy, on a '
                                   'scale from 0 (against democracy) to 10 (an essential characteristic of '
                                   'democracy). A nationally representative survey asked this question at '
                                   'both time points.'),
            direction_a='average answer moved toward "an essential characteristic of democracy" (people placed more importance on civil-rights protections as part of democracy).',
            direction_b='average answer moved toward lower importance (people placed less importance on civil-rights protections as part of democracy).',
            direction_labels={"A": "people placed more importance on civil-rights protections as part of democracy",
                               "B": "people placed less importance on civil-rights protections as part of democracy"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="F063", section="F",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how important God is in people\'s lives, on a scale from 1 (not at all '
                                   'important) to 10 (very important). A nationally representative survey '
                                   'asked this question at both time points.'),
            direction_a="average answer moved toward \"very important\" (religion became more central to people's lives).",
            direction_b="average answer moved toward \"not at all important\" (people became more secular, religion less central).",
            direction_labels={"A": "religion became more central to people's lives",
                               "B": "people became more secular (religion less central)"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E221B", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people have personally attended a peaceful, lawful '
                                   'demonstration recently, coded 1 (have done) or 2 (not done). A '
                                   'nationally representative survey asked this question at both time '
                                   'points.'),
            direction_a='average answer moved toward "have done" (more people recently attended a peaceful demonstration).',
            direction_b='average answer moved toward "not done" (fewer people recently attended a peaceful demonstration).',
            direction_labels={"A": "more people recently attended a peaceful demonstration",
                               "B": "fewer people recently attended a peaceful demonstration"},
        ),
        direction_if_mean_increases="B",
    ),
    dict(
        question="F121", section="F",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how justifiable people consider divorce, on a scale from 1 (never '
                                   'justifiable) to 10 (always justifiable). A nationally representative '
                                   'survey asked this question at both time points.'),
            direction_a='average answer moved toward "always justifiable" (attitudes became more accepting of divorce).',
            direction_b='average answer moved toward "never justifiable" (attitudes became less accepting of divorce).',
            direction_labels={"A": "attitudes became more accepting of divorce",
                               "B": "attitudes became less accepting of divorce"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E026B", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people have personally joined in a boycott recently, coded 1 '
                                   '(have done) or 2 (not done). A nationally representative survey asked '
                                   'this question at both time points.'),
            direction_a='average answer moved toward "have done" (more people recently joined in a boycott).',
            direction_b='average answer moved toward "not done" (fewer people recently joined in a boycott).',
            direction_labels={"A": "more people recently joined in a boycott",
                               "B": "fewer people recently joined in a boycott"},
        ),
        direction_if_mean_increases="B",
    ),
    dict(
        question="E217", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people agree that science and technology are making our lives '
                                   'healthier, easier and more comfortable, on a scale from 1 (completely '
                                   'disagree) to 10 (completely agree). A nationally representative survey '
                                   'asked this question at both time points.'),
            direction_a='average answer moved toward "completely agree" (people became more positive about the effect of science and technology on daily life).',
            direction_b='average answer moved toward "completely disagree" (people became more negative about the effect of science and technology on daily life).',
            direction_labels={"A": "people became more positive about the effect of science and technology on daily life",
                               "B": "people became more negative about the effect of science and technology on daily life"},
        ),
        direction_if_mean_increases="A",
    ),
]
