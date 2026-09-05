"""Compute similarity between vocab entries based on accepted_answers.

Splits each entry's comma-separated accepted_answers into normalized phrase
tokens, builds a TF-IDF representation over those phrases, then for every
word finds its top-k nearest neighbors by cosine similarity.

Outputs:
  data/accepted_answers_similarity.csv - long-format top-k neighbor table
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

VOCAB_CSV = "data/vocab.csv"
OUTPUT_FILE = "data/accepted_answers_similarity.csv"

TOP_K = 5

_WHITESPACE_RE = re.compile(r"\s+")


def split_answers(text: str) -> list[str]:
    """Split a comma-separated accepted_answers string into normalized phrases.

    Each phrase (e.g. "to move", "Chinese Ox constellation") is treated as a
    single indivisible token rather than word-tokenized, so partial word
    overlap (e.g. "to" shared by "to move" and "to eat") doesn't create
    spurious similarity between unrelated words. Trade-off: "cow" and "cow's
    milk" are then different tokens, so near-identical-but-not-exact phrases
    won't match - accepted, since word-level tokenization was worse (over-matched
    on generic connector words like "to"/"well"/"then").
    """
    return [
        _WHITESPACE_RE.sub(" ", phrase.strip().lower())
        for phrase in text.split(",")
        if phrase.strip()
    ]


def main() -> None:
    df = pd.read_csv(VOCAB_CSV)
    answers = df["accepted_answers"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        tokenizer=split_answers,
        token_pattern=None,   # required by sklearn when a custom tokenizer is supplied
        lowercase=False,      # split_answers already lowercases each phrase
        min_df=1,             # curated synonym lists have no typos to prune out
        sublinear_tf=True,    # dampen rows with a repeated phrase or very long lists (up to 100 items)
    )
    tfidf = vectorizer.fit_transform(answers)  # rows are L2-normalized -> dot product = cosine sim
    similarity = (tfidf @ tfidf.T).toarray()

    ids = df["id"].to_numpy()
    titles = df["title"].to_numpy()

    rows = []
    for i in tqdm(range(len(df)), desc="Computing top-k neighbors"):
        sims = similarity[i].copy()
        sims[i] = -np.inf  # exclude self
        top_indices = np.argpartition(-sims, TOP_K)[:TOP_K]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        for rank, j in enumerate(top_indices, start=1):
            rows.append(
                {
                    "id": ids[i],
                    "title": titles[i],
                    "rank": rank,
                    "similar_id": ids[j],
                    "similar_title": titles[j],
                    "similarity": round(float(sims[j]), 4),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(out)} rows ({len(df)} words x top {TOP_K}) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
