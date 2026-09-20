"""Find fill-in-the-blank study questions whose translation bolds an English
gloss that isn't semantically covered by the tested vocab word's own
`accepted_answers`.

Each study question's `translation` bolds the English gloss the blank is
testing (e.g. "<strong>mocking</strong>"). That gloss is meant to describe
the meaning of the vocab word being tested, so it should be one of that
word's own `accepted_answers` - otherwise a reverse-direction quiz (given
the English meaning, produce the Japanese word) would mark a translation
some readers would naturally use as unrecognized.

A raw string comparison over-flags: translations are written freely and
often use a different inflection or derived form (recommended/recommends
vs. accepted_answers' "recommendation") of a word that IS already accepted.
Rather than hand-rolling suffix-stripping rules for this, phrase similarity
is measured with the same embedding model this repo already uses for vocab
similarity (BAAI/bge-large-en-v1.5, see compute_similarity.py) - run
locally, no network LLM calls: the bolded gloss (rejoined across spans, for
an idiom split across a sentence like "<strong>make a</strong> strange
<strong>face</strong>") and each accepted_answers gloss are both encoded,
and the question is flagged only if none of them are close enough by cosine
similarity.

Even at a well-tuned threshold, most remaining flags are genuine synonym
gaps - translations reach for a natural English word (e.g. "mock") that the
curated `accepted_answers` list simply never included (e.g. only "sneer
at, ridicule"). This is a heuristic audit of `accepted_answers` coverage,
not a list of translation errors, so results should be spot-checked (and
often mean "add this gloss to accepted_answers") rather than treated as
ground truth.

Output:
  data/gloss_phrase_embeddings.npz - cached phrase embeddings (reused on rerun)
  data/missing_accepted_answers.csv - one row per flagged study question,
    sorted by vocab_title then study_question_id.
"""

import csv
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

VOCAB_CSV = "data/vocab.csv"
STUDY_QUESTIONS_CSV = "data/vocab_study_questions.csv"
OUTPUT_FILE = "data/missing_accepted_answers.csv"
EMBEDDINGS_CACHE = "data/gloss_phrase_embeddings.npz"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
# Cosine similarity below which a bolded gloss counts as "not covered" by the
# vocab's accepted_answers. Calibrated against known pairs: same-word
# inflection/derivation pairs (recommended/recommendation 0.91,
# humid/humidity 0.88, to look/look 0.88, make a face/making a face 0.92)
# score well above this; distinct-but-related synonym gaps worth flagging
# (mock/sneer at 0.64, wow/oh 0.70, before i knew it/in the blink of an eye
# 0.64) score below it.
SIMILARITY_THRESHOLD = 0.75

WS_RE = re.compile(r"\s+")
STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.DOTALL)


def normalize_phrase(text: str) -> str:
    return WS_RE.sub(" ", text.strip().lower())


def bold_phrases(translation: str) -> list[str]:
    return [normalize_phrase(m) for m in STRONG_RE.findall(translation) if normalize_phrase(m)]


def load_vocab() -> dict[str, dict]:
    vocab = {}
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            glosses = [normalize_phrase(p) for p in row["accepted_answers"].split(",") if p.strip()]
            vocab[row["id"]] = {"jlpt_level": row["jlpt_level"], "glosses": glosses}
    return vocab


def load_questions(vocab: dict[str, dict]) -> list[dict]:
    """Fill-in-the-blank rows for a known vocab entry, with a bolded-gloss
    candidate phrase (multiple bolded spans from one split idiom are
    rejoined into a single phrase)."""
    rows = []
    with open(STUDY_QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["answer"].strip():
                continue  # not a fill-in-the-blank question
            vid = row["vocab_id"]
            if vid not in vocab:
                continue
            bolds = bold_phrases(row["translation"])
            if not bolds:
                continue
            candidate = normalize_phrase(" ".join(bolds)) if len(bolds) > 1 else bolds[0]
            rows.append({**row, "vocab_id": vid, "bolds": bolds, "candidate": candidate})
    return rows


def get_embeddings(phrases: set[str], model: SentenceTransformer) -> dict[str, np.ndarray]:
    """phrase -> unit-normalized embedding, cached to disk across reruns."""
    cache: dict[str, np.ndarray] = {}
    try:
        data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
        if str(data["model"]) == MODEL_NAME:
            cache = dict(zip(data["phrases"].tolist(), data["embeddings"]))
    except FileNotFoundError:
        pass

    missing = sorted(phrases - cache.keys())
    if missing:
        print(f"Encoding {len(missing)} new phrases ({len(phrases) - len(missing)} already cached)...")
        embeddings = model.encode(missing, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        cache.update(zip(missing, embeddings))
        np.savez(
            EMBEDDINGS_CACHE,
            phrases=np.array(list(cache.keys())),
            embeddings=np.array(list(cache.values()), dtype=np.float32),
            model=MODEL_NAME,
        )

    return {p: cache[p] for p in phrases}


def main() -> None:
    vocab = load_vocab()
    rows = load_questions(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    all_glosses = {g for v in vocab.values() for g in v["glosses"]}
    all_candidates = {r["candidate"] for r in rows} | {b for r in rows for b in r["bolds"]}
    embeddings = get_embeddings(all_glosses | all_candidates, model)

    for v in vocab.values():
        v["gloss_embs"] = np.stack([embeddings[g] for g in v["glosses"]]) if v["glosses"] else None

    fieldnames = [
        "vocab_id",
        "vocab_title",
        "vocab_jlpt_level",
        "accepted_answers",
        "study_question_id",
        "content",
        "answer",
        "alternate_grammar",
        "alternate_answers",
        "translation",
        "missing_glosses",
        "closest_accepted_gloss",
        "similarity",
    ]
    flagged = []

    for row in rows:
        v = vocab[row["vocab_id"]]
        if v["gloss_embs"] is None:
            continue

        best_sim, best_gloss = -1.0, ""
        for candidate in {row["candidate"], *row["bolds"]}:
            sims = v["gloss_embs"] @ embeddings[candidate]
            i = int(np.argmax(sims))
            if float(sims[i]) > best_sim:
                best_sim, best_gloss = float(sims[i]), v["glosses"][i]

        if best_sim >= SIMILARITY_THRESHOLD:
            continue

        flagged.append(
            {
                "vocab_id": row["vocab_id"],
                "vocab_title": row["vocab_title"],
                "vocab_jlpt_level": v["jlpt_level"],
                "accepted_answers": ", ".join(v["glosses"]),
                "study_question_id": row["study_question_id"],
                "content": row["content"],
                "answer": row["answer"],
                "alternate_grammar": row["alternate_grammar"],
                "alternate_answers": row["alternate_answers"],
                "translation": row["translation"],
                "missing_glosses": "; ".join(row["bolds"]),
                "closest_accepted_gloss": best_gloss,
                "similarity": round(best_sim, 4),
            }
        )

    flagged.sort(key=lambda r: (r["vocab_title"], int(r["study_question_id"])))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged)

    n_vocab = len({r["vocab_id"] for r in flagged})
    print(f"Wrote {len(flagged)} flagged study questions across {n_vocab} vocab entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
