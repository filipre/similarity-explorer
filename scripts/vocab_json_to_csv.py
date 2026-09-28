"""Build a CSV of vocab fields from the JSON files in data/vocab_json."""

import csv
import json
import re
from pathlib import Path

from tqdm import tqdm

INPUT_DIR = Path("data/vocab_json")
OUTPUT_FILE = Path("data/vocab.csv")
LEVEL_KANJI_FILE = Path("data/level_kanji.csv")

KANJI_RE = re.compile(r"[一-鿿々]")  # CJK ideographs + 々

FIXED_FIELDS = [
    "id",
    "title",
    "jlpt_level",
    "wanikani_level",
    "wanikani_unknown_kanji",
    "furigana",
    "kana",
    "slug",
    "pitch_accent_stress",
    "female_audio_url",
    "male_audio_url",
    "meaning",
    "accepted_answers",
    "sense_glosses",
    "jmdict_pos",
    "word_types",
    "nuance_translation",
]

# Separator for sense_glosses; JMdict glosses routinely contain commas (and
# occasionally semicolons), so neither works as a delimiter, and none contain "|".
SENSE_GLOSS_SEP = " | "


def load_kanji_levels() -> dict[str, int]:
    with LEVEL_KANJI_FILE.open(encoding="utf-8") as f:
        return {row["kanji"]: int(row["level"]) for row in csv.DictReader(f)}


def wanikani_info(title: str, kanji_levels: dict[str, int]) -> tuple[int, str]:
    """Return (highest WaniKani level needed to read title, unknown kanji found in title)."""
    kanji_in_title = KANJI_RE.findall(title)
    known_levels = []
    unknown_kanji = []
    for kanji in dict.fromkeys(kanji_in_title):  # de-dupe, keep order
        level = kanji_levels.get(kanji)
        if level is None:
            unknown_kanji.append(kanji)
        else:
            known_levels.append(level)

    required_level = max(known_levels, default=0)
    return required_level, "".join(unknown_kanji)


# JMdict part-of-speech code -> readable word types. Exact codes first; the
# verb families (v1, v5r, vk, v2h-k, v4r, ...) are matched by prefix below.
POS_TO_TYPES = {
    "n": ["noun"],
    "n-suf": ["noun", "suffix"],
    "n-pref": ["noun", "prefix"],
    "pn": ["pronoun"],
    "num": ["numeral"],
    "ctr": ["counter"],
    "vt": ["transitive"],
    "vi": ["intransitive"],
    "vs": ["verb", "suru verb"],
    "vs-s": ["verb", "suru verb"],
    "vs-i": ["verb", "suru verb"],
    "vs-c": ["verb", "suru verb"],
    "vk": ["verb"],
    "vz": ["verb"],
    "vr": ["verb"],
    "vn": ["verb"],
    "adj-i": ["i-adjective"],
    "adj-ix": ["i-adjective"],
    "adj-na": ["na-adjective"],
    "adj-no": ["no-adjective"],
    "adj-pn": ["prenominal adjective"],
    "adj-f": ["prenominal"],
    "adj-t": ["taru-adjective"],
    "adj-ku": ["ku-adjective"],
    "adv": ["adverb"],
    "adv-to": ["adverb"],
    "conj": ["conjunction"],
    "int": ["interjection"],
    "prt": ["particle"],
    "aux": ["auxiliary"],
    "aux-v": ["auxiliary", "verb"],
    "aux-adj": ["auxiliary", "adjective"],
    "pref": ["prefix"],
    "suf": ["suffix"],
    "exp": ["expression"],
    "unc": ["unclassified"],
}
VERB_PREFIXES = ("v1", "v2", "v4", "v5")


def pos_codes(reviewable: dict) -> list[str]:
    """JMdict POS codes for the word.

    Bunpro's own jmdict_pos is used as-is when present; a few entries (e.g.
    カフェ) have it empty even though their JMdict senses are tagged, so fall
    back to the codes on the senses that carry an English gloss.
    """
    codes = list(reviewable.get("jmdict_pos") or [])
    if not codes:
        for sense in (reviewable.get("jmdict_data") or {}).get("sense", []):
            if any(g.get("lang") == "eng" for g in sense.get("gloss", [])):
                codes += sense.get("partOfSpeech", [])
    return list(dict.fromkeys(codes))


def word_types(codes: list[str]) -> list[str]:
    """Readable, de-duplicated word types for JMdict POS codes, in code order."""
    types: dict[str, None] = {}
    for code in codes:
        if code in POS_TO_TYPES:
            found = POS_TO_TYPES[code]
        elif code.startswith(VERB_PREFIXES):
            found = ["verb"]
        else:
            found = [f"unclassified ({code})"]  # keep unknown codes visible
        types.update(dict.fromkeys(found))
    return list(types)


def english_sense_glosses(reviewable: dict) -> str:
    """Every English gloss across the JMdict senses, de-duped, in source order.

    jmdict_data.sense[].gloss[] mixes glosses in many languages (dut, ger,
    hun, fre, ...); only the English ones describe the word's meaning in the
    terms the study questions use. This is a much wider net than
    accepted_answers, which is a short hand-curated subset of these.
    """
    glosses = {}
    for sense in (reviewable.get("jmdict_data") or {}).get("sense", []):
        for gloss in sense.get("gloss", []):
            text = (gloss.get("text") or "").strip()
            if gloss.get("lang") == "eng" and text:
                glosses[text] = None
    return SENSE_GLOSS_SEP.join(glosses)


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.json"))
    kanji_levels = load_kanji_levels()

    rows = []
    frequency_fields: list[str] = []
    for file in tqdm(files, desc="Reading vocab JSON"):
        data = json.loads(file.read_text())
        reviewable = data["props"]["pageProps"]["reviewable"]

        for key in reviewable:
            if key.startswith("frequency_") and key not in frequency_fields:
                frequency_fields.append(key)

        reviewable["wanikani_level"], reviewable["wanikani_unknown_kanji"] = (
            wanikani_info(reviewable["title"], kanji_levels)
        )

        reviewable["sense_glosses"] = english_sense_glosses(reviewable)
        codes = pos_codes(reviewable)
        reviewable["jmdict_pos"] = ",".join(codes)
        reviewable["word_types"] = ",".join(word_types(codes))

        rows.append(reviewable)

    fieldnames = FIXED_FIELDS[:8] + sorted(frequency_fields) + FIXED_FIELDS[8:]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
