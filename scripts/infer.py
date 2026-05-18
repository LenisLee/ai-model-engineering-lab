"""Run inference on input texts using a trained model."""

import argparse
import sys

from ai_model_engineering_lab.pipelines.inference import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", default="outputs/checkpoints", help="Path to model directory")
    parser.add_argument("texts", nargs="*", help="Texts to classify")
    args = parser.parse_args()

    texts = args.texts if args.texts else [input("Text: ").strip()]
    if not texts[0]:
        print("No input text provided.")
        sys.exit(1)

    pipeline = InferencePipeline(args.model)
    results = pipeline.run(texts)

    for r in results:
        print(f"{r['text'][:60]:<60} -> label={r['label']}, probs={r['probabilities']}")


if __name__ == "__main__":
    main()
