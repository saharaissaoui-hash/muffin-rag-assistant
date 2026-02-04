"""
extract_muffins.py

ETL script that extracts only muffin-related recipes from a large
recipe dataset.

Steps:
1. Load the dataset from Hugging Face.
2. Stream through recipes to avoid loading everything in memory.
3. Keep only recipes whose title contains muffin-related keywords.
4. Save the filtered subset to data/processed/muffins_only.csv.

The output file is the first clean dataset used by the RAG pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "m3hrdadfi/recipe_nlg_lite"
SPLIT = "train"

TITLE_COL = "name"
INGREDIENTS_COL = "ingredients"

MUFFIN_REGEX = re.compile(r"\b(muffin|muffins|cupcake|cupcakes)\b", re.IGNORECASE)


def main() -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "muffins_only.csv"

    ds = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    seen = set()

    for i, ex in enumerate(ds):
        title = str(ex.get(TITLE_COL, "")).strip()

        if not title or not MUFFIN_REGEX.search(title):
            continue

        ingredients = str(ex.get(INGREDIENTS_COL, "")).strip()
        if not ingredients:
            continue

        key = (title.lower(), ingredients.lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "title": title,
            "ingredients": ingredients,
        })

        # safety limit to avoid huge files
        if len(rows) >= 500:
            break

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} muffin recipes to {out_path}")


if __name__ == "__main__":
    main()
