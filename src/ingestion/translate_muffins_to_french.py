"""
translate_muffins_to_french.py

Translate the muffin-only dataset from English to French using a pretrained
seq2seq translation model.

Input:
  data/processed/muffins_only.csv  (columns: title, ingredients)

Output:
  data/processed/muffins_only_fr.csv (columns: title, ingredients)

This script avoids the Transformers pipeline registry and calls the model
directly (tokenizer -> generate -> decode), which is more robust across
environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROCESSED_DIR = Path("data/processed")
IN_PATH = PROCESSED_DIR / "muffins_only.csv"
OUT_PATH = PROCESSED_DIR / "muffins_only_fr.csv"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


def translate_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    texts: List[str],
    batch_size: int = 8,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
) -> List[str]:
    outputs: List[str] = []

    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t if isinstance(t, str) else "" for t in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["title"] = df["title"].fillna("").astype(str)
    df["ingredients"] = df["ingredients"].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    titles_fr = translate_batch(tokenizer, model, df["title"].tolist(), batch_size=8)
    ingredients_fr = translate_batch(tokenizer, model, df["ingredients"].tolist(), batch_size=8)

    out_df = pd.DataFrame(
        {
            "title": titles_fr,
            "ingredients": ingredients_fr,
        }
    )

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} translated recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
