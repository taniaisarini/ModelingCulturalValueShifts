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
    dict(
        question="E037", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people think individuals should take more responsibility for '
                                   'providing for themselves, or the government should take more '
                                   'responsibility for ensuring everyone is provided for, on a scale from '
                                   '1 (people should take more responsibility) to 10 (the government should '
                                   'take more responsibility). A nationally representative survey asked '
                                   'this question at both time points.'),
            direction_a='average answer moved toward "the government should take more responsibility" (people became more supportive of government responsibility for provision).',
            direction_b='average answer moved toward "people should take more responsibility" (people became more supportive of individual responsibility for provision).',
            direction_labels={"A": "people became more supportive of government responsibility for provision",
                               "B": "people became more supportive of individual responsibility for provision"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="F120", section="F",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how justifiable people consider abortion, on a scale from 1 (never '
                                   'justifiable) to 10 (always justifiable). A nationally representative '
                                   'survey asked this question at both time points.'),
            direction_a='average answer moved toward "always justifiable" (attitudes became more accepting of abortion).',
            direction_b='average answer moved toward "never justifiable" (attitudes became less accepting of abortion).',
            direction_labels={"A": "attitudes became more accepting of abortion",
                               "B": "attitudes became less accepting of abortion"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="F123", section="F",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how justifiable people consider suicide, on a scale from 1 (never '
                                   'justifiable) to 10 (always justifiable). A nationally representative '
                                   'survey asked this question at both time points.'),
            direction_a='average answer moved toward "always justifiable" (attitudes became more accepting of suicide).',
            direction_b='average answer moved toward "never justifiable" (attitudes became less accepting of suicide).',
            direction_labels={"A": "attitudes became more accepting of suicide",
                               "B": "attitudes became less accepting of suicide"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="C006", section="C",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=("people's satisfaction with their household's financial situation, on a "
                                   'scale from 1 (dissatisfied) to 10 (satisfied). A nationally '
                                   'representative survey asked this question at both time points.'),
            direction_a='average answer moved toward "satisfied" (people became more satisfied with their household finances).',
            direction_b='average answer moved toward "dissatisfied" (people became less satisfied with their household finances).',
            direction_labels={"A": "people became more satisfied with their household finances",
                               "B": "people became less satisfied with their household finances"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E234", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people think the world is better off or worse off because of '
                                   'science and technology, on a scale from 1 (a lot worse off) to 10 (a '
                                   'lot better off). A nationally representative survey asked this '
                                   'question at both time points.'),
            direction_a='average answer moved toward "a lot better off" (people became more positive about the overall effect of science and technology on the world).',
            direction_b='average answer moved toward "a lot worse off" (people became more negative about the overall effect of science and technology on the world).',
            direction_labels={"A": "people became more positive about the overall effect of science and technology on the world",
                               "B": "people became more negative about the overall effect of science and technology on the world"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E041", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people believe wealth can only be accumulated at others\' '
                                   'expense, or that wealth can grow so there is enough for everyone, on a '
                                   'scale from 1 (people can only get rich at the expense of others) to 10 '
                                   '(wealth can grow so there\'s enough for everyone). A nationally '
                                   'representative survey asked this question at both time points.'),
            direction_a="average answer moved toward \"wealth can grow so there's enough for everyone\" (people became more optimistic that wealth creation benefits everyone).",
            direction_b='average answer moved toward "people can only get rich at the expense of others" (people became more convinced that wealth comes at others\' expense).',
            direction_labels={"A": "people became more optimistic that wealth creation benefits everyone",
                               "B": "people became more convinced that wealth comes at others' expense"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E233", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people see women having the same rights as men as an essential '
                                   'characteristic of democracy, on a scale from 0 (against democracy) to '
                                   '10 (an essential characteristic of democracy). A nationally '
                                   'representative survey asked this question at both time points.'),
            direction_a='average answer moved toward "an essential characteristic of democracy" (people placed more importance on equal rights for women as part of democracy).',
            direction_b='average answer moved toward lower importance (people placed less importance on equal rights for women as part of democracy).',
            direction_labels={"A": "people placed more importance on equal rights for women as part of democracy",
                               "B": "people placed less importance on equal rights for women as part of democracy"},
        ),
        direction_if_mean_increases="B",
    ),
    dict(
        question="D059", section="D",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people agree that men make better political leaders than women '
                                   'do, coded 1 (agree strongly) to 4 (strongly disagree). A nationally '
                                   'representative survey asked this question at both time points.'),
            direction_a='average answer moved toward "strongly disagree" (people became less likely to think men make better political leaders than women).',
            direction_b='average answer moved toward "agree strongly" (people became more likely to think men make better political leaders than women).',
            direction_labels={"A": "people became less likely to think men make better political leaders than women",
                               "B": "people became more likely to think men make better political leaders than women"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="G021", section="G",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('whether people see themselves as a citizen of their nation, coded 1 '
                                   '(strongly agree) to 4 (strongly disagree). A nationally representative '
                                   'survey asked this question at both time points.'),
            direction_a='average answer moved toward "strongly disagree" (people became less likely to identify as a citizen of their nation).',
            direction_b='average answer moved toward "strongly agree" (people became more likely to identify as a citizen of their nation).',
            direction_labels={"A": "people became less likely to identify as a citizen of their nation",
                               "B": "people became more likely to identify as a citizen of their nation"},
        ),
        direction_if_mean_increases="A",
    ),
    dict(
        question="E069_01", section="E",
        case=build_case(
            country="United States", year_a=2006, year_b=2011,
            attitude_description=('how much confidence people have in churches, coded 1 (a great deal) to '
                                   '4 (none at all). A nationally representative survey asked this '
                                   'question at both time points.'),
            direction_a='average answer moved toward "none at all" (people had less confidence in churches).',
            direction_b='average answer moved toward "a great deal" (people had more confidence in churches).',
            direction_labels={"A": "people had less confidence in churches",
                               "B": "people had more confidence in churches"},
        ),
        direction_if_mean_increases="A",
    ),
]
