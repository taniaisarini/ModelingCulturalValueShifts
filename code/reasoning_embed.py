"""Embed the models' reasoning (OpenAI embeddings API) and cluster it.

    python code/reasoning_embed.py                 # embeds anything not yet cached, then clusters
    python code/reasoning_embed.py --clusters 10   # fixed number of clusters (default: best silhouette in 6..14)
    python code/reasoning_embed.py --dry-run       # counts texts that would be sent, sends nothing

Embeddings are cached on disk (embedding_cache/<model>.jsonl, keyed by a hash of the text) so a text is embedded once.
This is the only code that calls the API outside fast_llm.py, because fast_llm only does chat completions.
Output: results/direction_baseline/five_country_set/reasoning_analysis/embed_*.csv and embed_clusters.md.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

from reasoning_common import NAME_WORDS, OUT_DIR, ROOT, load_replies

EMBED_MODEL = "text-embedding-3-small"
CACHE = ROOT / "embedding_cache" / f"{EMBED_MODEL}.jsonl"
BATCH = 256


def key(text):
    return hashlib.sha256(f"{EMBED_MODEL}\n{text}".encode("utf-8")).hexdigest()


def load_cache():
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out.setdefault(r["key"], r["embedding"])
            except (json.JSONDecodeError, KeyError):
                continue          # skip a line truncated by a crash
    return out


def embed_missing(texts, dry_run):
    cache = load_cache()
    todo = sorted({t for t in texts if key(t) not in cache})
    print(f"{len(set(texts))} distinct texts, {len(set(texts)) - len(todo)} cached, {len(todo)} to embed")
    if dry_run or not todo:
        return cache
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            for t, d in zip(chunk, resp.data):
                cache[key(t)] = d.embedding
                f.write(json.dumps({"key": key(t), "model": EMBED_MODEL, "text": t, "embedding": d.embedding}) + "\n")
            f.flush()
            print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}")
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    df = load_replies()
    cache = embed_missing(df["reasoning"].tolist(), a.dry_run)
    if a.dry_run:
        return
    E = np.array([cache[key(t)] for t in df["reasoning"]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.clusters:
        k = a.clusters
    else:
        scores = {}
        for kk in range(6, 15):
            scores[kk] = silhouette_score(E, KMeans(kk, n_init=5, random_state=0).fit_predict(E), sample_size=2000, random_state=0)
        k = max(scores, key=scores.get)
        print("silhouette by k:", {kk: round(v, 3) for kk, v in scores.items()}, "-> using", k)
    km = KMeans(k, n_init=10, random_state=0).fit(E)
    df["cluster"] = km.labels_
    dist = np.linalg.norm(E - km.cluster_centers_[km.labels_], axis=1)
    df["dist"] = dist

    # words that set a cluster apart from the rest: mean TF-IDF in cluster minus mean elsewhere
    stop = list(ENGLISH_STOP_WORDS | set(NAME_WORDS) | {"people", "answer", "likely", "moved", "toward", "typical"})
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True)
    X = vec.fit_transform(df["reasoning"])
    terms = np.array(vec.get_feature_names_out())
    top = {}
    for c in range(k):
        m = (df["cluster"] == c).to_numpy()
        diff = np.asarray(X[m].mean(axis=0)).ravel() - np.asarray(X[~m].mean(axis=0)).ravel()
        top[c] = ", ".join(terms[np.argsort(diff)[::-1][:10]])

    share = (pd.crosstab(df["cluster"], df["model"], normalize="columns") * 100).round(1)
    summary = pd.DataFrame({"distinctive_words": pd.Series(top), "n": df["cluster"].value_counts().sort_index(),
                            "accuracy_%": (df.groupby("cluster")["correct"].mean() * 100).round(1)}).join(share.add_prefix("share_%_"))
    summary.to_csv(OUT_DIR / "embed_cluster_summary.csv", index_label="cluster")
    df[["model", "country", "waves", "code", "correct", "cluster", "dist", "reasoning"]].to_csv(OUT_DIR / "embed_assignments.csv", index=False)

    lines = [f"# Embedding clusters of model reasoning ({EMBED_MODEL}, k-means, k={k})\n",
             "Shares are the percent of each model's replies in the cluster. Examples are the replies closest to the cluster centre.\n"]
    for c in summary.sort_values("n", ascending=False).index:
        r = summary.loc[c]
        lines.append(f"## Cluster {c}: {r['distinctive_words']}\n\n{int(r['n'])} replies, accuracy {r['accuracy_%']}%\n")
        lines.append("| model | share % |\n|---|---|\n" + "\n".join(f"| {m} | {share.loc[c, m]} |" for m in share.columns) + "\n")
        for _, e in df[df["cluster"] == c].nsmallest(3, "dist").iterrows():
            lines.append(f"- *{e['model']}, {e['country']} {e['waves']} {e['code']}:* {e['reasoning']}")
        lines.append("")
    (OUT_DIR / "embed_clusters.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary[["distinctive_words", "n", "accuracy_%"]].sort_values("n", ascending=False).to_string())
    print(f"\nshare of each model's replies by cluster (%):\n{share.to_string()}")


if __name__ == "__main__":
    main()
