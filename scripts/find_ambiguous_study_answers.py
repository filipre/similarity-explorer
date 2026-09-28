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

Every flag is then checked against the sentence's grammatical slot using the
JMdict part-of-speech in vocab.csv (see grammar_conflict): a suru-verb slot
(`____する`), a na-adjective slot (`____な人`), or a transitive slot (`を____`)
can't be filled by a candidate that's only a noun / i-adjective / intransitive
verb. Such rows aren't dropped - the `grammar_conflict` column says why, so the
report can hide them by default while keeping the rule auditable.

Outputs:
  data/ambiguous_study_answers.csv, sorted with the highest-confidence flags
  first (dataset-confirmed, then smallest synonym cluster).
"""

import csv
import re
from collections import defaultdict

VOCAB_CSV = "data/vocab.csv"
STUDY_QUESTIONS_CSV = "data/vocab_study_questions.csv"
OUTPUT_FILE = "data/ambiguous_study_answers.csv"

# A cluster of vocab entries that share the same primary English gloss.
# Above this size the shared gloss is usually too generic (function words,
# broad adverbs) to represent a genuine synonym pair.
MAX_CLUSTER_SIZE = 6

# Separator used inside vocab.csv's sense_glosses column (see vocab_json_to_csv.py).
SENSE_GLOSS_SEP = " | "

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


def load_vocab():
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
                "slug": row["slug"],
                "kana": row["kana"],
                "jlpt_level": row["jlpt_level"],
                "accepted_answers": row["accepted_answers"],
                # Sense glosses widen only the wrong_answers overlap check; the
                # cluster path deliberately keys on the primary gloss alone.
                "glosses": all_glosses(row["accepted_answers"] or "")
                | {normalize_phrase(g) for g in (row["sense_glosses"] or "").split(SENSE_GLOSS_SEP) if g.strip()},
                "primary": primary,
                "pos": {t.strip() for t in row["jmdict_pos"].split(",") if t.strip()},
            }
            if primary:
                primary_to_ids[primary].add(vid)
            title_to_ids[row["title"]].add(vid)
            kana_to_ids[to_hiragana(row["kana"].lower())].add(vid)
    clusters = {p: ids for p, ids in primary_to_ids.items() if 2 <= len(ids) <= MAX_CLUSTER_SIZE}
    return vocab, clusters, title_to_ids, kana_to_ids


def accepted_forms_for_question(row: dict) -> set[str]:
    """Every Japanese answer form this question's grader accepts.

    alternate_grammar lists other accepted spellings/conjugations of the
    *same* answer (e.g. plain vs polite form, or dropping an honorific
    prefix like おふろ -> ふろ) - it always includes `answer` itself among
    its semicolon-separated items. alternate_answers lists accepted
    synonyms/variants beyond that one word. Both count as "already
    accepted" - missing either makes a genuinely-accepted answer look
    unaccepted.
    """
    forms = set()
    for raw in [
        row["answer"],
        *split_semicolon_list(row["alternate_grammar"]),
        *split_semicolon_list(row["alternate_answers"]),
    ]:
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


# ---- grammar check -------------------------------------------------------
# Sharing an English gloss doesn't make a candidate a valid fill for the blank:
# the sentence around it fixes the grammatical slot (`____する` needs a
# suru-verb, `____な人` a na-adjective, `を____` a transitive verb, ...), and
# a candidate whose JMdict POS can't fill that slot is a false positive. Rules
# fire only on positive evidence from the words next to the blank and are
# skipped when tested word and candidate share a grammatical class.

FURIGANA_RE = re.compile(r"（[^）]*）")
TAG_RE = re.compile(r"<[^>]+>")
KANJI_KATAKANA_RE = re.compile(r"[一-鿿々ァ-ヺ]")

SURU_POST_RE = re.compile(r"^(する|して|した|しま|しな|しよう|しろ|さ[れせ]|でき|いたし|致し|せず)")
# Continuations that only follow a na-adjective/noun, never an i-adjective:
# な+noun, に, だ (but not だろう), だっ(た), で (but not です), では, じゃ.
NA_ONLY_POST_RE = re.compile(r"^(な(?![。！？!?」、,…])|に|だ(?!ろ)|だっ|で(?!す)|では|じゃ)")
ATTRIBUTIVE_NA_RE = re.compile(r"^な[一-鿿々ァ-ヺ]")

# Prenominal/expression entries take their own attributive forms, so a plain
# adjective-type comparison doesn't apply to them.
FREE_FORM_POS = {"adj-pn", "adj-f", "exp"}

REASON_SURU = "sentence needs a suru-verb (blank followed by する) but candidate isn't one"
REASON_NA_ATTRIBUTIVE = "sentence needs a na-adjective (blank followed by な + noun) but candidate isn't one"
REASON_NA_VS_I = "sentence needs a na-adjective/noun form but candidate is an i-adjective"
REASON_I_ADJ = "sentence needs an i-adjective (blank directly before a noun) but candidate needs な"
REASON_VERB = "sentence needs a verb but candidate isn't one"
REASON_TRANSITIVE = "blank follows を (transitive) but candidate is intransitive"
REASON_INTRANSITIVE = "blank follows が (intransitive) but candidate is transitive-only"


def slot_context(content: str) -> tuple[str, str]:
    """(character before, text after) the `____` blank, with markup and furigana removed."""
    text = FURIGANA_RE.sub("", TAG_RE.sub("", content))
    i = text.find("____")
    if i == -1:
        return "", ""
    return text[i - 1 : i], text[i + 4 : i + 10]


def is_verb(pos: set[str]) -> bool:
    return any(p.startswith(("v1", "v2", "v4", "v5", "vk", "vz", "vr", "vn")) for p in pos)


def is_suru(pos: set[str]) -> bool:
    return any(p.startswith("vs") for p in pos)


def grammar_classes(pos: set[str]) -> set[str]:
    """Coarse grammatical classes; a shared class means the two words can fill the same kind of slot."""
    classes = set()
    for p in pos:
        if p.startswith(("v1", "v2", "v4", "v5")) or p in {"vk", "vz", "vr", "vn", "aux-v"}:
            classes.add("verb")
        elif p.startswith("vs"):
            classes.add("suru")
        elif p in {"adj-i", "adj-ix", "aux-adj"}:
            classes.add("i-adj")
        elif p not in {"vt", "vi"}:
            classes.add(p)
    return classes


def grammar_conflict(content: str, tested: set[str], cand: set[str]) -> str:
    """Why `cand` can't grammatically fill the blank meant for `tested`, or "" if it can."""
    before, after = slot_context(content)
    shared = grammar_classes(tested) & grammar_classes(cand)

    if is_suru(tested) and SURU_POST_RE.match(after) and not is_suru(cand):
        return REASON_SURU

    if "adj-na" in tested and "adj-na" not in cand and not cand & FREE_FORM_POS:
        if ATTRIBUTIVE_NA_RE.match(after):
            return REASON_NA_ATTRIBUTIVE
        if "adj-i" in cand and "n" not in cand and NA_ONLY_POST_RE.match(after):
            return REASON_NA_VS_I

    pure_i_adj = "adj-i" in tested and not tested & {"adj-na", "n"}
    if pure_i_adj and not shared and not cand & FREE_FORM_POS and KANJI_KATAKANA_RE.match(after):
        return REASON_I_ADJ

    if is_verb(tested) and not shared and not is_verb(cand) and not is_suru(cand):
        return REASON_VERB

    if is_verb(tested) and is_verb(cand):
        # JMdict leaves vt/vi off verbs whose transitivity it doesn't record, so
        # only a candidate positively tagged as the opposite counts as a conflict.
        if before == "を" and "vt" in tested and "vi" in cand and "vt" not in cand:
            return REASON_TRANSITIVE
        if before == "が" and "vi" in tested and "vt" not in tested and "vt" in cand and "vi" not in cand:
            return REASON_INTRANSITIVE

    return ""


def main() -> None:
    vocab, clusters, title_to_ids, kana_to_ids = load_vocab()

    fieldnames = [
        "source",
        "cluster_size",
        "study_question_id",
        "vocab_id",
        "vocab_title",
        "vocab_slug",
        "vocab_jlpt_level",
        "content",
        "answer",
        "alternate_grammar",
        "alternate_answers",
        "wrong_answers",
        "translation",
        "shared_meaning",
        "candidate_vocab_id",
        "candidate_title",
        "candidate_slug",
        "candidate_kana",
        "candidate_jlpt_level",
        "candidate_accepted_answers",
        "grammar_conflict",
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
                        "vocab_slug": v["slug"],
                        "vocab_jlpt_level": v["jlpt_level"],
                        "content": row["content"],
                        "answer": row["answer"],
                        "alternate_grammar": row["alternate_grammar"],
                        "alternate_answers": row["alternate_answers"],
                        "wrong_answers": row["wrong_answers"],
                        "translation": row["translation"],
                        "shared_meaning": from_cluster.get(cid) or from_wrong.get(cid),
                        "candidate_vocab_id": cid,
                        "candidate_title": cand["title"],
                        "candidate_slug": cand["slug"],
                        "candidate_kana": cand["kana"],
                        "candidate_jlpt_level": cand["jlpt_level"],
                        "candidate_accepted_answers": cand["accepted_answers"],
                        "grammar_conflict": grammar_conflict(row["content"], v["pos"], cand["pos"]),
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
    by_conflict = defaultdict(int)
    for r in flagged:
        if r["grammar_conflict"]:
            by_conflict[r["grammar_conflict"]] += 1
    print(f"Grammar conflicts: {sum(by_conflict.values())} of {len(flagged)} flags")
    for reason, n in sorted(by_conflict.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {reason}")


if __name__ == "__main__":
    main()
