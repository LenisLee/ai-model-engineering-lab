import sys
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    with open("configs/base/config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    model = AutoModelForSequenceClassification.from_pretrained(train_cfg["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["output_dir"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = sys.argv[1:] if len(sys.argv) > 1 else [input("Text: ")]
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = logits.argmax(dim=-1)

    id2label = model.config.id2label
    for text, pred in zip(texts, preds):
        print(f"{text[:60]:<60} -> {id2label[pred.item()]}")


if __name__ == "__main__":
    main()
