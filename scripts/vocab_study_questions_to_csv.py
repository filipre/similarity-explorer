"""Build a CSV mapping vocab id/title/slug to their example sentences
(studyQuestions) from the JSON files in data/vocab_json.
"""

import csv
import json
from pathlib import Path

from tqdm import tqdm

INPUT_DIR = Path("data/vocab_json")
OUTPUT_FILE = Path("data/vocab_study_questions.csv")

FIELDNAMES = [
    "vocab_id",
    "vocab_title",
    "vocab_slug",
    "study_question_id",
    "content",
    "answer",
    "translation",
]


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.json"))

    rows = []
    for file in tqdm(files, desc="Reading vocab JSON"):
        data = json.loads(file.read_text())
        page_props = data["props"]["pageProps"]
        reviewable = page_props["reviewable"]
        study_questions = page_props.get("included", {}).get("studyQuestions", [])

        for question in study_questions:
            rows.append(
                {
                    "vocab_id": reviewable["id"],
                    "vocab_title": reviewable["title"],
                    "vocab_slug": reviewable["slug"],
                    "study_question_id": question["id"],
                    "content": question["content"],
                    "answer": question["answer"],
                    "translation": question["translation"],
                }
            )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
