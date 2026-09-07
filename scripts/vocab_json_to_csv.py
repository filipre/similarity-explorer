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
    "nuance_translation",
]


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
