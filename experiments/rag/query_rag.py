"""Query a RAG index: retrieve relevant documents for a query."""

import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Query a RAG index")
    parser.add_argument("--index", default="outputs/rag_index", help="Index directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("query", nargs="?", default=None, help="Search query")
    args = parser.parse_args()

    logger = setup_logging()
    index_dir = Path(args.index)
    index = faiss.read_index(str(index_dir / "index.faiss"))

    with open(index_dir / "documents.txt") as f:
        documents = [line.strip() for line in f if line.strip()]

    query = args.query or input("Query: ").strip()
    if not query:
        print("No query provided.")
        return

    encoder = SentenceTransformer(args.model)
    q_emb = encoder.encode([query], normalize_embeddings=True).astype(np.float32)

    scores, indices = index.search(q_emb, args.top_k)

    logger.info(f"Top {args.top_k} results for: {query}")
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            print(f"  [{score:.4f}] {documents[idx][:120]}")


if __name__ == "__main__":
    main()
