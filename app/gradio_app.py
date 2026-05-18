"""Gradio interactive demo for text classification."""

import argparse

import gradio as gr

from ai_model_engineering_lab.inference.engine import InferenceEngine
from ai_model_engineering_lab.utils.logging import setup_logging


def create_app(model_path: str):
    logger = setup_logging()
    logger.info(f"Loading model from {model_path} ...")
    engine = InferenceEngine.from_pretrained(model_path)

    def classify(text: str) -> dict:
        if not text.strip():
            return {"error": "Please enter some text."}
        result = engine.predict_single(text)
        return {
            label: prob
            for label, prob in zip(
                engine.model.config.id2label.values(), result["probabilities"]
            )
        }

    with gr.Blocks(title="Text Classification Demo") as demo:
        gr.Markdown("# Text Classification Demo")
        gr.Markdown("Enter text to classify using the fine-tuned model.")

        with gr.Row():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type your text here...",
                lines=4,
            )

        with gr.Row():
            submit_btn = gr.Button("Classify", variant="primary")
            clear_btn = gr.Button("Clear")

        label_output = gr.Label(label="Prediction", num_top_classes=5)

        submit_btn.click(fn=classify, inputs=text_input, outputs=label_output)
        clear_btn.click(fn=lambda: ("", None), outputs=[text_input, label_output])

        gr.Examples(
            examples=[
                "This movie was fantastic! I loved every minute of it.",
                "Terrible waste of time. The acting was wooden and the plot made no sense.",
                "It was okay, nothing special but not bad either.",
            ],
            inputs=text_input,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--model", default="outputs/checkpoints", help="Path to model directory")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    demo = create_app(args.model)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
