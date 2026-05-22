"""Step 2 of RAG: feed retrieved documents to a local LLM and generate an answer.

Prerequisite: run query_rag.py first to get the retrieved documents.

Usage:
  python experiments/rag/query_rag.py --index outputs/my_rag --query "酒店环境怎么样" --top_k 3

  Then copy the output into:

  python experiments/rag/generate_answer.py --docs "评论1 ||| 评论2 ||| 评论3" --query "酒店环境怎么样"
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="RAG Step 2: generate answer from retrieved docs")
    parser.add_argument("--docs", required=True, help="Retrieved documents, separated by |||")
    parser.add_argument("--query", required=True, help="Original question")
    args = parser.parse_args()

    docs = [d.strip() for d in args.docs.split("|||")]
    print(f"Retrieved {len(docs)} documents")
    for i, d in enumerate(docs):
        print(f"  [{i+1}] {d[:80]}")

    print(f"\nLoading {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, trust_remote_code=True
    )

    prompt = f"根据以下用户评论回答。\n\n评论:\n" + "\n".join(f"- {d}" for d in docs)
    prompt += f"\n\n问题: {args.query}\n\n用一句话回答："

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
