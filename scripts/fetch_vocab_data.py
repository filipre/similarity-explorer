"""Fetch __NEXT_DATA__ JSON for each vocab URL listed in data/urls_n5.txt."""

import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

BASE_URL = "https://bunpro.jp"
URLS_FILE = Path("data/urls_n4.txt")
OUTPUT_DIR = Path("data/vocab_json")
SLEEP_SECONDS = 1
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S
)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def slug_for(path: str) -> str:
    vocab = unquote(urlparse(path).path.removeprefix("/vocabs/"))
    return re.sub(r"[^\w-]", "_", vocab)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [line.strip() for line in URLS_FILE.read_text().splitlines() if line.strip()]

    for i, path in enumerate(paths, start=1):
        url = BASE_URL + path
        out_file = OUTPUT_DIR / f"{slug_for(path)}.json"

        if out_file.exists():
            print(f"[{i}/{len(paths)}] {url} (skip, already fetched)")
            continue

        print(f"[{i}/{len(paths)}] {url}")

        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        match = NEXT_DATA_RE.search(response.text)
        if not match:
            print(f"  WARNING: __NEXT_DATA__ not found for {url}")
            time.sleep(SLEEP_SECONDS)
            continue

        data = json.loads(match.group(1))
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
