"""Regenerate data/missing_accepted_answers.csv and render it as an HTML report.

Run this after vocab.csv or vocab_study_questions.csv changes:
    uv run python3 scripts/generate_missing_accepted_answers_report.py

Outputs:
  data/missing_accepted_answers.csv          (via find_missing_accepted_answers)
  data/missing_accepted_answers_report.html  (self-contained - open directly in a browser)

The report's markup/styling/behavior lives entirely in
scripts/templates/missing_accepted_answers_report.html - edit that file to
change the report (columns shown, filters, charts, colors, etc.); this
script only reruns the analysis and splices the fresh CSV into the template in place of the
"__CSV_DATA__" marker. The embedded copy is trimmed to the columns the template
reads and obfuscated (embed_payload.pack) so it isn't plain text in View Source, and the
inline JS is minified with terser;
the template must keep each of "__CSV_DATA__" (inside the
`<script type="text/plain" id="csv-data">` tag) and "__UNPACK_JS__" exactly once.
"""

from pathlib import Path

import find_missing_accepted_answers as analysis
from embed_payload import UNPACK_JS, drop_csv_columns, minify_inline_scripts, pack, splice

SCRIPT_DIR = Path(__file__).parent
TEMPLATE_FILE = SCRIPT_DIR / "templates" / "missing_accepted_answers_report.html"
PLACEHOLDER = "__CSV_DATA__"
UNPACK_PLACEHOLDER = "__UNPACK_JS__"
# Columns the template never reads - left out of the embedded copy (the CSV on disk keeps them).
UNUSED_COLUMNS = {"alternate_grammar", "alternate_answers"}


def main() -> None:
    analysis.main()  # regenerates data/missing_accepted_answers.csv from the current vocab data

    csv_file = Path(analysis.OUTPUT_FILE)
    report_file = csv_file.with_name(csv_file.stem + "_report.html")

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    csv_text = drop_csv_columns(csv_file.read_text(encoding="utf-8"), UNUSED_COLUMNS)
    html = splice(template, {PLACEHOLDER: pack(csv_text), UNPACK_PLACEHOLDER: UNPACK_JS}, TEMPLATE_FILE.name)
    html = minify_inline_scripts(html)
    report_file.write_text(html, encoding="utf-8")

    print(f"Wrote {report_file} ({report_file.stat().st_size:,} bytes) - open it directly in a browser.")


if __name__ == "__main__":
    main()
