"""
retrieve.py

Query the persisted ChromaDB vector store to retrieve the most relevant muffin
recipes given a French user query.

This module is used by the RAG pipeline to build the context that will be fed
to the generation model (Chef Muffin).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_COLLECTION = "royaume_du_muffin"
DEFAULT_PERSIST_DIR = "data/vector_store/chroma"


def get_chroma_client(persist_dir: Path):
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=str(persist_dir))

    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_dir),
        )
    )


def retrieve(
    query: str,
    top_k: int = 5,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, Any]]:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(f"Vector store directory not found: {persist_path}")

    model = SentenceTransformer(embed_model)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    client = get_chroma_client(persist_path)
    collection = client.get_or_create_collection(name=collection_name)

    res = collection.query(query_embeddings=q_emb, n_results=top_k)

    results = []
    for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        results.append(
            {
                "title": meta.get("title"),
                "ingredients": meta.get("ingredients"),
                "document": doc,
                "distance": dist,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    hits = retrieve(args.query, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for i, h in enumerate(hits, start=1):
        print(f"{i}. {h['title']}")
        print(f"   distance: {h['distance']}")
        print(f"   ingredients: {h['ingredients']}\n")


if __name__ == "__main__":
    main()
