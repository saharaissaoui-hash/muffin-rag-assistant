"""
translate_muffins_to_french.py

Translate muffin recipes from English to French.

Steps:
1. Load the muffin-only dataset.
2. Translate title and ingredients.
3. Save the translated dataset to data/processed/muffins_only_fr.csv.

This file becomes the canonical dataset used by the RAG pipeline.
"""

from pathlib import Path
import pandas as pd
from transformers import pipeline


PROCESSED_DIR = Path("data/processed")
IN_PATH = PROCESSED_DIR / "muffins_only.csv"
OUT_PATH = PROCESSED_DIR / "muffins_only_fr.csv"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


def translate_texts(translator, texts, batch_size=8):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = translator(batch)
        outputs.extend([p["translation_text"] for p in preds])
    return outputs


def main() -> None:
    df = pd.read_csv(IN_PATH)

    translator = pipeline(
        "translation",
        model=MODEL_NAME,
    )

    titles = df["title"].fillna("").tolist()
    ingredients = df["ingredients"].fillna("").tolist()

    df["title_fr"] = translate_texts(translator, titles)
    df["ingredients_fr"] = translate_texts(translator, ingredients)

    out_df = df[["title_fr", "ingredients_fr"]].rename(
        columns={
            "title_fr": "title",
            "ingredients_fr": "ingredients",
        }
    )

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} translated recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
