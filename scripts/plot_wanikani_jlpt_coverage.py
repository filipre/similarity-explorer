"""Plot cumulative WaniKani-level coverage of vocab.csv, split by JLPT level.

For each JLPT level, a word counts as "covered" at WaniKani level L once every
kanji in its title is taught by WaniKani at or before L (see
`wanikani_level`/`wanikani_unknown_kanji` in vocab_json_to_csv.py). Words
containing kanji WaniKani never teaches can never be covered, so lines
plateau below 100%.

Output: data/wanikani_jlpt_coverage.png
"""

import matplotlib.pyplot as plt
import pandas as pd

VOCAB_CSV = "data/vocab.csv"
OUTPUT_FILE = "data/wanikani_jlpt_coverage.png"

JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1"]
MAX_WANIKANI_LEVEL = 60

# Same categorical slots/order used in the interactive artifact version.
SERIES_COLORS = {
    "N5": "#2a78d6",
    "N4": "#eb6834",
    "N3": "#1baf7a",
    "N2": "#eda100",
    "N1": "#e87ba4",
}


def coverage_curve(covered_levels: pd.Series) -> list[float]:
    """Cumulative % of covered_levels that are <= L, for L in 0..MAX_WANIKANI_LEVEL."""
    total = len(covered_levels)
    return [
        (covered_levels <= level).sum() / total * 100
        for level in range(MAX_WANIKANI_LEVEL + 1)
    ]


def main() -> None:
    df = pd.read_csv(VOCAB_CSV)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for jlpt_level in JLPT_LEVELS:
        subset = df[df["jlpt_level"] == jlpt_level]
        fully_readable = subset[subset["wanikani_unknown_kanji"].isna()]
        curve = coverage_curve(fully_readable["wanikani_level"])
        ax.plot(
            range(MAX_WANIKANI_LEVEL + 1),
            curve,
            label=f"{jlpt_level} ({len(subset):,} words)",
            color=SERIES_COLORS[jlpt_level],
            linewidth=2,
        )

    ax.set_xlim(0, MAX_WANIKANI_LEVEL)
    ax.set_ylim(0, 100)
    ax.set_xlabel("WaniKani level")
    ax.set_ylabel("% of vocabulary readable")
    ax.set_title("WaniKani level needed to read JLPT vocabulary")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
