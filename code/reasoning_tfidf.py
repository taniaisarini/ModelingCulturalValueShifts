"""Topic model (TF-IDF + NMF) of the reasoning the models gave in the direction baseline.

    python code/reasoning_tfidf.py [--topics 12]

Fits one topic model on all models' replies together (country names and years removed), then reports for each topic
its top words, example replies, how common it is in each model's replies, and the accuracy of replies on it.
Writes to results/direction_baseline/five_country_set/reasoning_analysis/tfidf_*.csv and tfidf_topics.md.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from reasoning_common import NAME_WORDS, OUT_DIR, load_replies


def by_country(df, stop, k):
    """A separate topic model for each country's replies (all models pooled), so topics can be specific to that country."""
    out, rows = df[["model", "country", "waves", "code"]].copy(), []
    out["ctopic"] = -1
    for country, sub in df.groupby("country"):
        vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=4, max_df=0.5,
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True)
        X = vec.fit_transform(sub["reasoning"])
        nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=500)
        W = nmf.fit_transform(X)
        terms = np.array(vec.get_feature_names_out())
        out.loc[sub.index, "ctopic"] = W.argmax(axis=1)
        for t in range(k):
            rows.append(dict(country=country, topic=t, n=int((W.argmax(axis=1) == t).sum()),
                             top_words=", ".join(terms[np.argsort(nmf.components_[t])[::-1][:10]])))
    return out["ctopic"], pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=12)
    ap.add_argument("--country-topics", type=int, default=8, help="topics per country in the per-country models")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_replies()
    stop = list(ENGLISH_STOP_WORDS | set(NAME_WORDS) | {"people", "answer", "likely", "moved", "toward", "typical"})
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True)
    X = vec.fit_transform(df["reasoning"])
    nmf = NMF(n_components=a.topics, init="nndsvda", random_state=0, max_iter=500)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    df["topic"] = W.argmax(axis=1)
    df["topic_weight"] = W.max(axis=1)
    print(f"{len(df)} replies, {X.shape[1]} terms, {a.topics} topics")

    top = {t: ", ".join(terms[np.argsort(nmf.components_[t])[::-1][:10]]) for t in range(a.topics)}
    share = (pd.crosstab(df["topic"], df["model"], normalize="columns") * 100).round(1)
    acc = (df.groupby("topic")["correct"].mean() * 100).round(1)
    summary = pd.DataFrame({"top_words": pd.Series(top), "n": df["topic"].value_counts().sort_index(),
                            "accuracy_%": acc}).join(share.add_prefix("share_%_"))
    summary.to_csv(OUT_DIR / "tfidf_topic_summary.csv", index_label="topic")
    df[["model", "country", "waves", "code", "correct", "topic", "topic_weight", "reasoning"]].to_csv(
        OUT_DIR / "tfidf_assignments.csv", index=False)

    lines = ["# TF-IDF / NMF topics in model reasoning\n",
             f"{len(df)} replies from {df['model'].nunique()} model runs; each reply is assigned its strongest topic. "
             "Shares are the percent of each model's replies in the topic.\n"]
    for t in summary.sort_values("n", ascending=False).index:
        r = summary.loc[t]
        lines.append(f"## Topic {t}: {r['top_words']}\n\n{int(r['n'])} replies, accuracy {r['accuracy_%']}%\n")
        lines.append("| model | share % |\n|---|---|\n" + "\n".join(f"| {m} | {share.loc[t, m] if t in share.index else 0} |"
                                                                    for m in share.columns) + "\n")
        for _, e in df[df["topic"] == t].nlargest(3, "topic_weight").iterrows():
            lines.append(f"- *{e['model']}, {e['country']} {e['waves']} {e['code']}:* {e['reasoning']}")
        lines.append("")
    (OUT_DIR / "tfidf_topics.md").write_text("\n".join(lines), encoding="utf-8")
    ctopic, csum = by_country(df, stop, a.country_topics)
    pd.concat([df[["model", "country", "waves", "code"]], ctopic], axis=1).to_csv(OUT_DIR / "tfidf_country_assignments.csv", index=False)
    csum.to_csv(OUT_DIR / "tfidf_country_topic_summary.csv", index=False)
    print(f"per-country topics ({a.country_topics} each):\n" + csum.sort_values(["country", "n"], ascending=[True, False]).to_string(index=False))
    print(summary[["top_words", "n", "accuracy_%"]].sort_values("n", ascending=False).to_string())
    print(f"\nshare of each model's replies by topic (%):\n{share.to_string()}")


if __name__ == "__main__":
    main()
