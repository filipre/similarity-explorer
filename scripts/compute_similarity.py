"""Compute semantic similarity between vocab entries based on nuance_translation.

Encodes each nuance_translation with a sentence-transformer model, then for
every word finds its top-k nearest neighbors by cosine similarity.

Outputs:
  data/nuance_embeddings.npz   - embeddings + ids (cached, reused on rerun)
  data/nuance_similarity.csv   - long-format top-k neighbor table
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

VOCAB_CSV = "data/vocab.csv"
EMBEDDINGS_FILE = "data/nuance_embeddings.npz"
OUTPUT_FILE = "data/nuance_similarity.csv"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 5


def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    try:
        cached = np.load(EMBEDDINGS_FILE)
        if np.array_equal(cached["ids"], df["id"].to_numpy()) and cached["model"] == MODEL_NAME:
            print(f"Using cached embeddings from {EMBEDDINGS_FILE}")
            return cached["embeddings"]
    except FileNotFoundError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(df)} entries with {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        df["nuance_translation"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.savez(EMBEDDINGS_FILE, ids=df["id"].to_numpy(), embeddings=embeddings, model=MODEL_NAME)
    return embeddings


def main() -> None:
    df = pd.read_csv(VOCAB_CSV)
    embeddings = load_embeddings(df)

    similarity = embeddings @ embeddings.T

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
