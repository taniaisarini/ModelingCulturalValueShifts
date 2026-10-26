"""Data file for the dashboard's "Reasoning" tab: topics / clusters of the models' reasoning and every reply.

    python code/export_dashboard_reasoning.py     # run reasoning_tfidf.py and reasoning_embed.py first

Reads reasoning_analysis/{tfidf,embed}_assignments.csv and *_summary.csv, writes
results/direction_baseline/five_country_set/reasoning_data.js (window.REASONING). The topic names below were written by
hand from each topic's top words; the check on `marker` stops a stale name being attached if a re-run reorders topics.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dashboard_baseline import build_rows  # noqa: E402
from question_themes import THEMES, theme_of  # noqa: E402
from reasoning_common import OUT_DIR, SET_DIR, load_replies  # noqa: E402

# id -> (name, marker word that must appear in the topic's top words)
TFIDF_NAMES = {6: ("Economic growth, technology, crises", "economic"), 4: ("Military coups and democracy", "military"),
               1: ("Gender attitudes and acceptance", "gender"), 2: ("Generic: “significant political changes”", "experienced significant"),
               7: ("Umbrella Movement and Beijing", "umbrella"), 8: ("Neighbours and shares saying so", "neighbors"),
               11: ("Crime, security, alcohol", "crime"), 9: ("Religion and secularization", "religious"),
               10: ("Revolution and political upheaval", "upheaval"), 0: ("Trust in institutions", "confidence"),
               5: ("Civic membership and participation", "membership"), 3: ("Internet and media", "internet")}
EMBED_NAMES = {1: ("Egypt: upheaval, revolution, state", "upheaval"), 4: ("Latin America: stable, progressive", "latin"),
               3: ("Hong Kong: Umbrella Movement, Beijing", "umbrella"), 5: ("Military coup and turmoil", "military"),
               0: ("Ukraine: Euromaidan, Russia, war", "euromaidan"), 2: ("Thailand: Buddhism, urbanization, youth", "buddhist")}


def build(prefix, col, words_col, names):
    summ = pd.read_csv(OUT_DIR / f"{prefix}_summary.csv").set_index(col)
    out = []
    for i, r in summ.iterrows():
        name, marker = names[int(i)]
        assert marker in r[words_col], f"{prefix} {i}: expected '{marker}' in {r[words_col]}"
        out.append(dict(id=int(i), name=name, words=r[words_col]))
    return out


def country_topic_name(words):
    top = words.split(", ")
    return "Generic: \u201csignificant political changes\u201d" if top[0].startswith("experienced significant") else " \u00b7 ".join(top[:3])


def main():
    tf = pd.read_csv(OUT_DIR / "tfidf_assignments.csv")
    em = pd.read_csv(OUT_DIR / "embed_assignments.csv")
    ct = pd.read_csv(OUT_DIR / "tfidf_country_assignments.csv")
    assert len(tf) == len(em) == len(ct) and (tf["reasoning"] == em["reasoning"]).all()
    models = list(dict.fromkeys(tf["model"]))

    # the model's predicted answer (as text) and the question's theme, rebuilt from the same result files
    full = load_replies()
    assert len(full) == len(tf) and (full["code"] == tf["code"]).all() and (full["model"] == tf["model"]).all()
    extra = []
    for name, grp in full.groupby("model", sort=False):
        res = build_rows(grp.drop(columns=["reasoning", "waves", "words"]).reset_index(drop=True))
        extra += [(r["pred_text"], theme_of(r["code"], r["label"])) for r in res]
    assert len(extra) == len(tf)

    replies = [[models.index(m), c, w, code, int(ok), int(t), int(e), txt, int(ctp), pt, THEMES.index(th)]
               for m, c, w, code, ok, t, e, txt, ctp, (pt, th) in
               zip(tf["model"], tf["country"], tf["waves"], tf["code"], tf["correct"], tf["topic"], em["cluster"],
                   tf["reasoning"], ct["ctopic"], extra)]
    csum = pd.read_csv(OUT_DIR / "tfidf_country_topic_summary.csv")
    ctopics = {c: [dict(id=int(r.topic), name=country_topic_name(r.top_words), words=r.top_words) for r in g.itertuples()]
               for c, g in csum.groupby("country")}
    data = dict(models=models, themes=THEMES, countries=sorted(ctopics),
                methods=dict(tfidf=dict(label="Topics (TF-IDF)", groups=build("tfidf_topic", "topic", "top_words", TFIDF_NAMES)),
                             embed=dict(label="Clusters (embeddings)", groups=build("embed_cluster", "cluster", "distinctive_words", EMBED_NAMES))),
                country_topics=ctopics, replies=replies)
    out = SET_DIR / "reasoning_data.js"
    out.write_text("window.REASONING=" + json.dumps(data, separators=(",", ":"), ensure_ascii=True) + ";\n", encoding="utf-8")
    print(f"wrote {len(replies)} replies to {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
