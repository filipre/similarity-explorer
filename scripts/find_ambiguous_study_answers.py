"""Find fill-in-the-blank study questions where a synonym vocab word could
also correctly answer the blank, but isn't accepted as an answer.

Each fill-in-the-blank question's `translation` bolds the English gloss the
blank is testing (e.g. "<strong>death</strong>"). If another vocab entry's
*primary* (first-listed) accepted_answers gloss is the exact same phrase, and
it belongs to the same broad part of speech, it's a genuine near-synonym: a
student who knows that word could reasonably type it. It only counts as
correct if it's present in the question's `answer`/`alternate_answers` - when
it isn't, the question is flagged as a likely source of confusion.

Example: study_question_id 119223 tests 死 ("death") with only "し" accepted,
but 3146 死亡 ("death, mortality, to die, to pass away") also glosses
primarily to "death" and isn't accepted - a student answering 死亡 would be
marked wrong despite it being a correct translation of the English cue.

Matching only on each word's *primary* gloss (rather than any shared item
buried in a longer accepted_answers list) and requiring overlapping
part-of-speech tags keeps results to genuine word-for-word mixups (死/死亡,
あなた/君, いつも/常に) rather than incidental gloss overlaps between
unrelated words (e.g. a demonstrative adjective like あの "that" vs a
pronoun like それ "that", which aren't interchangeable despite the shared
English word). This is still a heuristic over English glosses, not a
grammar or sentence-semantics check, so results should be spot-checked
rather than treated as ground truth.

Outputs:
  data/ambiguous_study_answers.csv, sorted with the highest-confidence
  (smallest synonym cluster) flags first.
"""

import csv
import re
from collections import defaultdict

VOCAB_CSV = "data/vocab.csv"
VOCAB_POS_CSV = "data/vocab_pos.csv"
STUDY_QUESTIONS_CSV = "data/vocab_study_questions.csv"
OUTPUT_FILE = "data/ambiguous_study_answers.csv"

# A cluster of vocab entries that share the same primary English gloss.
# Above this size the shared gloss is usually too generic (function words,
# broad adverbs) to represent a genuine synonym pair.
MAX_CLUSTER_SIZE = 6

WS_RE = re.compile(r"\s+")
STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.DOTALL)

# Comparing kana script-insensitively (e.g. katakana ワイシャツ vs a hiragana
# alternate_answers spelling) avoids treating a same-word script difference
# as a distinct, unaccepted candidate.
KATA_TO_HIRA = str.maketrans({chr(c): chr(c - 0x60) for c in range(0x30A1, 0x30F7)})


def normalize_phrase(text: str) -> str:
    return WS_RE.sub(" ", text.strip().lower())


def to_hiragana(text: str) -> str:
    return text.translate(KATA_TO_HIRA)


def primary_gloss(accepted_answers: str) -> str | None:
    parts = [p for p in accepted_answers.split(",") if p.strip()]
    return normalize_phrase(parts[0]) if parts else None


def load_pos() -> dict[str, set[str]]:
    pos = {}
    with open(VOCAB_POS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pos[row["id"]] = {t.strip() for t in row["jmdict_pos"].split(",") if t.strip()}
    return pos


def load_vocab() -> tuple[dict, dict[str, set[str]]]:
    pos_by_id = load_pos()
    vocab = {}
    primary_to_ids = defaultdict(set)
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["id"]
            primary = primary_gloss(row["accepted_answers"] or "")
            vocab[vid] = {
                "title": row["title"],
                "kana": row["kana"],
                "accepted_answers": row["accepted_answers"],
                "primary": primary,
                "pos": pos_by_id.get(vid, set()),
            }
            if primary:
                primary_to_ids[primary].add(vid)
    clusters = {p: ids for p, ids in primary_to_ids.items() if 2 <= len(ids) <= MAX_CLUSTER_SIZE}
    return vocab, clusters


def accepted_forms_for_question(row: dict) -> set[str]:
    forms = set()
    for raw in [row["answer"], *row["alternate_answers"].split(",")]:
        raw = raw.strip()
        if raw:
            forms.add(to_hiragana(raw.lower()))
    return forms


def main() -> None:
    vocab, clusters = load_vocab()

    fieldnames = [
        "cluster_size",
        "study_question_id",
        "vocab_id",
        "vocab_title",
        "content",
        "answer",
        "alternate_answers",
        "translation",
        "shared_meaning",
        "candidate_vocab_id",
        "candidate_title",
        "candidate_kana",
        "candidate_accepted_answers",
    ]
    flagged = []

    with open(STUDY_QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["answer"].strip():
                continue  # not a fill-in-the-blank question

            vid = row["vocab_id"]
            v = vocab.get(vid)
            if not v or not v["primary"]:
                continue

            cluster = clusters.get(v["primary"])
            if not cluster:
                continue

            bold_phrases = {normalize_phrase(m) for m in STRONG_RE.findall(row["translation"])}
            if v["primary"] not in bold_phrases:
                continue  # this question isn't testing the word's primary sense

            accepted_forms = accepted_forms_for_question(row)

            for cid in cluster:
                if cid == vid:
                    continue
                cand = vocab[cid]
                if v["pos"] and cand["pos"] and not (v["pos"] & cand["pos"]):
                    continue  # different part of speech - not really interchangeable

                cand_forms = {to_hiragana(cand["kana"].lower()), to_hiragana(cand["title"].lower())}
                if cand_forms & accepted_forms:
                    continue  # synonym is already accepted - not ambiguous

                flagged.append(
                    {
                        "cluster_size": len(cluster),
                        "study_question_id": row["study_question_id"],
                        "vocab_id": vid,
                        "vocab_title": row["vocab_title"],
                        "content": row["content"],
                        "answer": row["answer"],
                        "alternate_answers": row["alternate_answers"],
                        "translation": row["translation"],
                        "shared_meaning": v["primary"],
                        "candidate_vocab_id": cid,
                        "candidate_title": cand["title"],
                        "candidate_kana": cand["kana"],
                        "candidate_accepted_answers": cand["accepted_answers"],
                    }
                )

    flagged.sort(key=lambda r: (r["cluster_size"], r["study_question_id"]))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged)

    n_questions = len({r["study_question_id"] for r in flagged})
    print(f"Wrote {len(flagged)} ambiguous answer flags across {n_questions} study questions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
