"""Minimal LLM chat agent with tool-calling loop (HuggingFace transformers)."""

import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_model_engineering_lab.utils.device import get_device
from ai_model_engineering_lab.utils.logging import setup_logging

TOOLS = {
    "calculator": {
        "description": "Evaluate a math expression. Input: a string like '2 + 3 * 4'.",
        "fn": lambda expr: str(eval(expr)),
    },
    "reverse_string": {
        "description": "Reverse a string. Input: the string to reverse.",
        "fn": lambda s: s[::-1],
    },
}

SYSTEM_PROMPT = """You are a helpful assistant with access to tools. Use tools when needed.

Available tools:
- calculator: Evaluate a math expression. Input: a string like "2 + 3 * 4".
- reverse_string: Reverse a string. Input: the string to reverse.

To use a tool, respond with EXACTLY:
<tool>tool_name</tool>
<input>your input</input>

Otherwise respond normally. After a tool result is provided, continue with your final answer.
"""


def run_tool(name: str, input_str: str) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    try:
        return tool["fn"](input_str)
    except Exception as e:
        return f"Tool error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Chat with a tool-using agent")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM model name")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max tool-calling rounds")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Chat Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        for _ in range(args.max_rounds):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            if "<tool>" in response and "</tool>" in response:
                t_start = response.index("<tool>") + len("<tool>")
                t_end = response.index("</tool>")
                i_start = response.index("<input>") + len("<input>")
                i_end = response.index("</input>")
                tool_name = response[t_start:t_end].strip()
                tool_input = response[i_start:i_end].strip()

                result = run_tool(tool_name, tool_input)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {result}"})
                continue

            messages.append({"role": "assistant", "content": response})
            print(f"\nAssistant: {response}\n")
            break


if __name__ == "__main__":
    main()
