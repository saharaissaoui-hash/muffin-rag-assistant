"""
inspect_hf_dataset.py

Utility script to inspect the schema of a Hugging Face dataset.

This script loads a dataset and prints:
- the available columns
- one example row

It is used during the ETL design phase to understand the structure
of the dataset before writing cleaning or filtering logic.
"""

import argparse
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    if args.streaming:
        first = next(iter(ds))
        print("Columns:", list(first.keys()))
        print("First example:")
        print(first)
    else:
        print("Columns:", ds.column_names)
        print("First example:")
        print(ds[0])


if __name__ == "__main__":
    main()


