"""Full RAG pipeline: retrieval + generation with a local LLM.

RAG (Retrieval-Augmented Generation):
  1. Retrieve: search a knowledge base for relevant documents
  2. Generate: feed docs + question to an LLM

Usage:
  python experiments/rag/build_index.py --docs /tmp/chn_docs.txt --model BAAI/bge-small-zh-v1.5 --output outputs/my_rag
  python experiments/rag/rag_pipeline.py --index outputs/my_rag --query "酒店环境怎么样"

Reference: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
"""

import argparse
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="RAG: retrieve + generate")
    parser.add_argument("--index", default="outputs/my_rag")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    # ---- Load retrieval ----
    idx_dir = Path(args.index)
    print(f"Loading index from {idx_dir} ...")
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    with open(idx_dir / "documents.txt") as f:
        docs = [l.strip() for l in f if l.strip()]

    print(f"Loading embedding model ...")
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # ---- Retrieve ----
    print(f"Searching: {args.query}")
    q = embedder.encode([args.query])
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    scores, indices = index.search(q.astype(np.float32), args.top_k)
    retrieved = [docs[i] for i in indices[0]]

    print(f"\n{'='*60}")
    print(f"RETRIEVED (Top-{args.top_k}):")
    print(f"{'='*60}")
    for i, (doc, s) in enumerate(zip(retrieved, scores[0])):
        print(f"  [{i+1}] {s:.3f}  {doc[:100]}")

    # Free embedder before loading LLM — 两个模型不能同时占内存
    del embedder
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # ---- Generate ----
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading LLM ({MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, trust_remote_code=True
    )

    context = "\n".join(f"- {d}" for d in retrieved)
    prompt = f"根据以下用户评论回答。\n\n评论:\n{context}\n\n问题: {args.query}\n\n用一句话回答："

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\n{'='*60}")
    print(f"LLM ANSWER:")
    print(f"{'='*60}")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, temperature=0.3, do_sample=True)
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  {answer}\n")


if __name__ == "__main__":
    main()
