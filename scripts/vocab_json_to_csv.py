"""Build a CSV of vocab fields from the JSON files in data/vocab_json."""

import csv
import json
from pathlib import Path

from tqdm import tqdm

INPUT_DIR = Path("data/vocab_json")
OUTPUT_FILE = Path("data/vocab.csv")

FIXED_FIELDS = [
    "id",
    "title",
    "jlpt_level",
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


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.json"))

    rows = []
    frequency_fields: list[str] = []
    for file in tqdm(files, desc="Reading vocab JSON"):
        data = json.loads(file.read_text())
        reviewable = data["props"]["pageProps"]["reviewable"]

        for key in reviewable:
            if key.startswith("frequency_") and key not in frequency_fields:
                frequency_fields.append(key)

        rows.append(reviewable)

    fieldnames = FIXED_FIELDS[:6] + sorted(frequency_fields) + FIXED_FIELDS[6:]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
