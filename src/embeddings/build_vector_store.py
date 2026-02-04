"""
build_vector_store.py

Build a persistent vector store for the French muffin dataset.

Input:
  data/processed/muffins_only_fr.csv  (columns: title, ingredients)

Output:
  A persisted ChromaDB collection on disk (default: data/vector_store/chroma)

What it does:
  1) Loads the dataset
  2) Builds documents from "title + ingredients"
  3) Computes embeddings using a multilingual SentenceTransformer model
  4) Stores embeddings + metadata in ChromaDB (persisted locally)
  5) Runs a small retrieval test query in French
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "royaume_du_muffin"


def stable_id(title: str, ingredients: str) -> str:
    raw = (title.strip() + "||" + ingredients.strip()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_chroma_client(persist_dir: Path):
    """
    Create a persistent Chroma client compatible with multiple Chroma versions.
    """
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=str(persist_dir))

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/muffins_only_fr.csv")
    parser.add_argument("--persist_dir", default="data/vector_store/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate the collection.")
    parser.add_argument("--test_query", default="Je veux un muffin au chocolat")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"title", "ingredients"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: title, ingredients")

    df["title"] = df["title"].fillna("").astype(str)
    df["ingredients"] = df["ingredients"].fillna("").astype(str)

    documents: List[str] = (df["title"] + " : " + df["ingredients"]).tolist()
    metadatas = df[["title", "ingredients"]].to_dict(orient="records")
    ids = [stable_id(t, ing) for t, ing in zip(df["title"], df["ingredients"])]

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = get_chroma_client(persist_dir)

    if args.rebuild:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=args.collection, embedding_function=None)


    existing = set()
    try:
        existing_ids = collection.get(include=[])["ids"]
        existing = set(existing_ids)
    except Exception:
        existing = set()

    to_add = [(i, d, e, m) for i, d, e, m in zip(ids, documents, embeddings, metadatas) if i not in existing]

    if to_add:
        add_ids, add_docs, add_embs, add_metas = zip(*to_add)
        collection.add(
            ids=list(add_ids),
            documents=list(add_docs),
            embeddings=list(add_embs),
            metadatas=list(add_metas),
        )

    if hasattr(client, "persist"):
        client.persist()

    print(f"Collection: {args.collection}")
    print(f"Stored recipes: {collection.count()} (added {len(to_add)})")
    print(f"Persisted at: {persist_dir}")

    if args.test_query:
        q_emb = model.encode([args.test_query], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=args.top_k)

        print("\nTest query:", args.test_query)
        for rank, meta in enumerate(res["metadatas"][0], start=1):
            print(f"{rank}. {meta['title']}")


if __name__ == "__main__":
    main()
