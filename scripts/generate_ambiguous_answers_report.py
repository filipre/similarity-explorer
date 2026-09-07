"""Regenerate data/ambiguous_study_answers.csv and render it as an HTML report.

Run this after vocab.csv or vocab_study_questions.csv changes:
    python3 scripts/generate_ambiguous_answers_report.py

Outputs:
  data/ambiguous_study_answers.csv          (via find_ambiguous_study_answers)
  data/ambiguous_study_answers_report.html  (self-contained - open directly in a browser)

The report's markup/styling/behavior lives entirely in
scripts/templates/ambiguous_answers_report.html - edit that file to change the
report (columns shown, filters, charts, colors, etc.); this script only reruns
the analysis and splices the fresh CSV into the template in place of the
"__CSV_DATA__" marker, so the template must keep that marker exactly once,
inside the `<script type="text/csv" id="csv-data">` tag.
"""

from pathlib import Path

import find_ambiguous_study_answers as analysis

SCRIPT_DIR = Path(__file__).parent
TEMPLATE_FILE = SCRIPT_DIR / "templates" / "ambiguous_answers_report.html"
PLACEHOLDER = "__CSV_DATA__"


def main() -> None:
    analysis.main()  # regenerates data/ambiguous_study_answers.csv from the current vocab data

    csv_file = Path(analysis.OUTPUT_FILE)
    report_file = csv_file.with_name(csv_file.stem + "_report.html")

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit(f"{TEMPLATE_FILE} must contain exactly one {PLACEHOLDER} marker")

    csv_text = csv_file.read_text(encoding="utf-8")
    report_file.write_text(template.replace(PLACEHOLDER, csv_text), encoding="utf-8")

    print(f"Wrote {report_file} ({report_file.stat().st_size:,} bytes) - open it directly in a browser.")


if __name__ == "__main__":
    main()
