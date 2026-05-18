"""Build a FAISS index from a document corpus for RAG retrieval."""

import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index for RAG")
    parser.add_argument("--docs", required=True, help="Path to documents file (one per line)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--output", default="outputs/rag_index", help="Output directory")
    args = parser.parse_args()

    logger = setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    with open(args.docs) as f:
        documents = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(documents)} documents")

    # Encode
    logger.info(f"Encoding with {args.model} ...")
    encoder = SentenceTransformer(args.model)
    embeddings = encoder.encode(documents, show_progress_bar=True, normalize_embeddings=True)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "documents.txt", "w") as f:
        for doc in documents:
            f.write(doc + "\n")

    logger.info(f"Index with {index.ntotal} vectors saved to {output_dir}")


if __name__ == "__main__":
    main()
