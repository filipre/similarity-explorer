"""Find fill-in-the-blank study questions where a synonym vocab word could
also correctly answer the blank, but isn't accepted as an answer.

Each fill-in-the-blank question's `translation` bolds the English gloss the
blank is testing (e.g. "<strong>death</strong>"). A candidate vocab word is
flagged as a likely source of confusion if it shares that gloss but its kana
and kanji forms are absent from the question's `answer`/`alternate_answers`.
Candidates come from two independent sources:

1. cluster - vocab entries whose *primary* (first-listed) accepted_answers
   gloss exactly matches the tested word's primary gloss, and which share a
   part-of-speech tag with it. Matching on primary gloss only (rather than
   any item buried in a longer accepted_answers list) and requiring POS
   overlap keeps this to genuine word-for-word mixups (死/死亡, あなた/君,
   いつも/常に) rather than incidental overlaps between unrelated words
   (e.g. the adnominal あの "that" vs the pronoun それ "that" - not
   interchangeable despite the shared English word).

2. wrong_answers - the dataset itself lists, for some questions, Japanese
   words test-takers commonly answer with instead of the accepted answer.
   Whenever a listed wrong answer resolves to a vocab entry whose
   accepted_answers overlaps the question's bolded gloss, the dataset has
   independently confirmed the same kind of mixup this script is looking
   for (e.g. お祖父さん "grandfather" vs the wrong answer おじさん "uncle"
   isn't a gloss match and is correctly rejected as wrong - but a wrong
   answer that *does* share the gloss is exactly the 死/死亡 pattern, just
   already caught by hand).

A row's `source` column says which of the two found it; `cluster+wrong_answers`
rows are the highest-confidence flags, since the heuristic and the dataset's
own curation agree. This is still a heuristic over English glosses, not a
grammar or sentence-semantics check, so results should be spot-checked rather
than treated as ground truth.

Outputs:
  data/ambiguous_study_answers.csv, sorted with the highest-confidence flags
  first (dataset-confirmed, then smallest synonym cluster).
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


def split_semicolon_list(text: str) -> list[str]:
    """Split a `;`-separated field into word-like tokens.

    alternate_answers/wrong_answers use `;` as the item separator (not `,` -
    a comma can appear inside a single item, e.g. an explanatory note like
    "だいたい; Although this could work here in this context, ..."). Tokens
    containing whitespace are such notes rather than a Japanese answer form,
    so they're dropped.
    """
    return [t.strip() for t in text.split(";") if t.strip() and " " not in t.strip()]


def primary_gloss(accepted_answers: str) -> str | None:
    parts = [p for p in accepted_answers.split(",") if p.strip()]
    return normalize_phrase(parts[0]) if parts else None


def all_glosses(accepted_answers: str) -> set[str]:
    return {normalize_phrase(p) for p in accepted_answers.split(",") if p.strip()}


def load_pos() -> dict[str, set[str]]:
    pos = {}
    with open(VOCAB_POS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pos[row["id"]] = {t.strip() for t in row["jmdict_pos"].split(",") if t.strip()}
    return pos


def load_vocab():
    pos_by_id = load_pos()
    vocab = {}
    primary_to_ids = defaultdict(set)
    title_to_ids = defaultdict(set)
    kana_to_ids = defaultdict(set)
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["id"]
            primary = primary_gloss(row["accepted_answers"] or "")
            vocab[vid] = {
                "title": row["title"],
                "kana": row["kana"],
                "accepted_answers": row["accepted_answers"],
                "glosses": all_glosses(row["accepted_answers"] or ""),
                "primary": primary,
                "pos": pos_by_id.get(vid, set()),
            }
            if primary:
                primary_to_ids[primary].add(vid)
            title_to_ids[row["title"]].add(vid)
            kana_to_ids[to_hiragana(row["kana"].lower())].add(vid)
    clusters = {p: ids for p, ids in primary_to_ids.items() if 2 <= len(ids) <= MAX_CLUSTER_SIZE}
    return vocab, clusters, title_to_ids, kana_to_ids


def accepted_forms_for_question(row: dict) -> set[str]:
    forms = set()
    for raw in [row["answer"], *split_semicolon_list(row["alternate_answers"])]:
        raw = raw.strip()
        if raw:
            forms.add(to_hiragana(raw.lower()))
    return forms


def cluster_candidates(row, vid, v, vocab, clusters, accepted_forms) -> dict[str, str]:
    """vocab_id -> matched gloss, for same-primary-gloss/POS candidates not already accepted."""
    if not v["primary"]:
        return {}
    cluster = clusters.get(v["primary"])
    if not cluster:
        return {}

    bold_phrases = {normalize_phrase(m) for m in STRONG_RE.findall(row["translation"])}
    if v["primary"] not in bold_phrases:
        return {}  # this question isn't testing the word's primary sense

    matches = {}
    for cid in cluster:
        if cid == vid:
            continue
        cand = vocab[cid]
        if v["pos"] and cand["pos"] and not (v["pos"] & cand["pos"]):
            continue  # different part of speech - not really interchangeable
        cand_forms = {to_hiragana(cand["kana"].lower()), to_hiragana(cand["title"].lower())}
        if cand_forms & accepted_forms:
            continue  # already accepted - not ambiguous
        matches[cid] = v["primary"]
    return matches


def wrong_answer_candidates(row, vid, vocab, title_to_ids, kana_to_ids, accepted_forms) -> dict[str, str]:
    """vocab_id -> matched gloss, for dataset-listed wrong answers that share a gloss with the question."""
    tokens = split_semicolon_list(row["wrong_answers"])
    if not tokens:
        return {}

    bold_phrases = {normalize_phrase(m) for m in STRONG_RE.findall(row["translation"])}
    if not bold_phrases:
        return {}

    matches = {}
    for token in tokens:
        candidate_ids = title_to_ids.get(token, set()) | kana_to_ids.get(to_hiragana(token.lower()), set())
        for cid in candidate_ids:
            if cid == vid or cid in matches:
                continue
            cand = vocab[cid]
            cand_forms = {to_hiragana(cand["kana"].lower()), to_hiragana(cand["title"].lower())}
            if cand_forms & accepted_forms:
                continue  # dataset lists it as wrong, but it's actually an accepted form here
            shared = cand["glosses"] & bold_phrases
            if shared:
                matches[cid] = sorted(shared)[0]
    return matches


def main() -> None:
    vocab, clusters, title_to_ids, kana_to_ids = load_vocab()

    fieldnames = [
        "source",
        "cluster_size",
        "study_question_id",
        "vocab_id",
        "vocab_title",
        "content",
        "answer",
        "alternate_answers",
        "wrong_answers",
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
            if not v:
                continue

            accepted_forms = accepted_forms_for_question(row)
            from_cluster = cluster_candidates(row, vid, v, vocab, clusters, accepted_forms)
            from_wrong = wrong_answer_candidates(row, vid, vocab, title_to_ids, kana_to_ids, accepted_forms)

            for cid in from_cluster.keys() | from_wrong.keys():
                in_cluster = cid in from_cluster
                in_wrong = cid in from_wrong
                source = "cluster+wrong_answers" if in_cluster and in_wrong else ("cluster" if in_cluster else "wrong_answers")
                cand = vocab[cid]
                flagged.append(
                    {
                        "source": source,
                        "cluster_size": len(clusters[v["primary"]]) if in_cluster else "",
                        "study_question_id": row["study_question_id"],
                        "vocab_id": vid,
                        "vocab_title": row["vocab_title"],
                        "content": row["content"],
                        "answer": row["answer"],
                        "alternate_answers": row["alternate_answers"],
                        "wrong_answers": row["wrong_answers"],
                        "translation": row["translation"],
                        "shared_meaning": from_cluster.get(cid) or from_wrong.get(cid),
                        "candidate_vocab_id": cid,
                        "candidate_title": cand["title"],
                        "candidate_kana": cand["kana"],
                        "candidate_accepted_answers": cand["accepted_answers"],
                    }
                )

    source_rank = {"cluster+wrong_answers": 0, "wrong_answers": 1, "cluster": 2}
    flagged.sort(key=lambda r: (source_rank[r["source"]], r["cluster_size"] or 999, r["study_question_id"]))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged)

    n_questions = len({r["study_question_id"] for r in flagged})
    by_source = defaultdict(int)
    for r in flagged:
        by_source[r["source"]] += 1
    print(f"Wrote {len(flagged)} ambiguous answer flags across {n_questions} study questions to {OUTPUT_FILE}")
    print(f"By source: {dict(by_source)}")


if __name__ == "__main__":
    main()
