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

Embeddings alone don't reliably score inflected forms against the base form
("said" vs. "to say", "land" vs. "landing"), so a lemma check runs first: each
phrase is lemmatized token by token with lemminflect (leading "to" dropped),
and a bolded gloss that shares a lemma sequence with any gloss is treated as
covered without consulting the embeddings.

Suru-nouns (JMdict `vs`, e.g. 記憶 "memory") are also used as verbs, so for
them a single-word gloss also covers its derivational relatives from WordNet
(nltk) that share a word stem: "memorize" for "memory". The wordnet corpus is
downloaded on first run.

A word's gloss pool is its `accepted_answers` plus the English glosses from
the JMdict senses (`sense_glosses` in vocab.csv), which are far more numerous
(e.g. 'mock' is a JMdict gloss of あざ笑う but not one of its accepted_answers).
A gloss covered only by a sense gloss is not flagged: the flagged rows are the
ones neither list covers. The output's `sense_glosses` column lists the sense
glosses that are not already in accepted_answers, and `closest_gloss_source`
says which list the closest match came from.

Even at a well-tuned threshold, most remaining flags are genuine synonym
gaps - translations reach for a natural English word that neither the curated
`accepted_answers` nor the JMdict senses ever included. This is a heuristic
audit of gloss coverage,
not a list of translation errors, so results should be spot-checked (and
often mean "add this gloss to accepted_answers") rather than treated as
ground truth.

Output:
  data/gloss_phrase_embeddings.npz - cached phrase embeddings (reused on rerun)
  data/missing_accepted_answers.csv - one row per flagged study question,
    sorted by vocab_title then study_question_id.
"""

import csv
import itertools
import os
import re
from functools import lru_cache

import nltk
import numpy as np
import torch
from lemminflect import getAllLemmas
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

nltk.download("wordnet", quiet=True)  # no-op once cached in ~/nltk_data

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

# Shortest shared prefix for two words to count as sharing a stem when
# following WordNet derivational links (memory/memorize share "memor").
MIN_SHARED_STEM = 4

# Separator used inside vocab.csv's sense_glosses column (see vocab_json_to_csv.py).
SENSE_GLOSS_SEP = " | "

WS_RE = re.compile(r"\s+")
STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.DOTALL)


def normalize_phrase(text: str) -> str:
    return WS_RE.sub(" ", text.strip().lower())


def bold_phrases(translation: str) -> list[str]:
    return [normalize_phrase(m) for m in STRONG_RE.findall(translation) if normalize_phrase(m)]


def strip_infinitive(phrase: str) -> str:
    return phrase[3:] if phrase.startswith("to ") and len(phrase) > 3 else phrase


@lru_cache(maxsize=None)
def token_lemmas(token: str) -> tuple[str, ...]:
    """The token plus every lemma lemminflect knows for it, across parts of
    speech ("landing" -> landing, land; "said" -> said, say)."""
    lemmas = {token}
    for spellings in getAllLemmas(token).values():
        lemmas.update(spellings)
    return tuple(sorted(lemmas))


def wordnet_relatives(word: str) -> set[str]:
    """Single verbs WordNet lists as derivationally related to `word` that
    also share a word stem with it (memory -> memorize, storage -> store).
    The stem check drops WordNet's looser sense-level links (memory ->
    remember/retain, training -> educate); the verb check drops agent nouns
    and adjectives (guarantee -> guarantor, patience -> patient)."""
    relatives = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                name = related.name().lower()
                if (
                    related.synset().pos() == "v"
                    and "_" not in name
                    and len(os.path.commonprefix([word, name])) >= MIN_SHARED_STEM
                ):
                    relatives.add(name)
    return relatives


def derived_relatives(glosses: list[str]) -> set[str]:
    """Lemmas derivationally related to any single-word gloss."""
    relatives = set()
    for gloss in glosses:
        tokens = strip_infinitive(gloss).split()
        if len(tokens) == 1:
            for lemma in token_lemmas(tokens[0]):
                relatives |= wordnet_relatives(lemma)
    return relatives


def lemma_keys(phrase: str) -> set[tuple[str, ...]]:
    """Every way to lemmatize the phrase token by token, so two phrases that
    differ only in inflection ("gave up" / "to give up", "landing" / "land")
    share at least one key."""
    tokens = strip_infinitive(phrase).split()
    return set(itertools.product(*(token_lemmas(t) for t in tokens)))


def load_vocab() -> dict[str, dict]:
    vocab = {}
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            accepted = [normalize_phrase(p) for p in row["accepted_answers"].split(",") if p.strip()]
            senses = [normalize_phrase(p) for p in row["sense_glosses"].split(SENSE_GLOSS_SEP) if p.strip()]
            # accepted_answers first so a tie on similarity reports the curated gloss.
            glosses = list(dict.fromkeys(accepted + senses))
            # A suru-noun (記憶 "memory") is also used as a verb ("memorize"),
            # so a verb derived from one of its noun glosses counts as covered.
            is_suru_noun = "vs" in row["jmdict_pos"].split(",")
            vocab[row["id"]] = {
                "jlpt_level": row["jlpt_level"],
                "accepted": accepted,
                "senses": senses,
                "glosses": glosses,
                "keys": set().union(*(lemma_keys(g) for g in glosses)),
                "relatives": derived_relatives(glosses) if is_suru_noun else set(),
            }
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
        "vocab_slug",
        "vocab_jlpt_level",
        "accepted_answers",
        "sense_glosses",
        "study_question_id",
        "content",
        "answer",
        "alternate_grammar",
        "alternate_answers",
        "translation",
        "missing_glosses",
        "closest_accepted_gloss",
        "closest_gloss_source",
        "similarity",
    ]
    flagged = []

    for row in rows:
        v = vocab[row["vocab_id"]]
        if v["gloss_embs"] is None:
            continue

        # Same words as a known gloss up to inflection ("said" for "to say",
        # "land" for "landing") is covered.
        if any(lemma_keys(p) & v["keys"] for p in (row["candidate"], *row["bolds"])):
            continue

        # A suru-noun's verb form ("memorize" for 記憶 "memory") is covered.
        if v["relatives"] and any(
            len(tokens := strip_infinitive(p).split()) == 1 and v["relatives"].intersection(token_lemmas(tokens[0]))
            for p in (row["candidate"], *row["bolds"])
        ):
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
                "vocab_slug": row["vocab_slug"],
                "vocab_jlpt_level": v["jlpt_level"],
                "accepted_answers": ", ".join(v["accepted"]),
                "sense_glosses": SENSE_GLOSS_SEP.join(g for g in v["senses"] if g not in v["accepted"]),
                "study_question_id": row["study_question_id"],
                "content": row["content"],
                "answer": row["answer"],
                "alternate_grammar": row["alternate_grammar"],
                "alternate_answers": row["alternate_answers"],
                "translation": row["translation"],
                "missing_glosses": "; ".join(row["bolds"]),
                "closest_accepted_gloss": best_gloss,
                "closest_gloss_source": "accepted_answers" if best_gloss in v["accepted"] else "sense",
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
