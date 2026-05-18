from transformers import PreTrainedTokenizerBase


class TextPreprocessor:
    """Text preprocessing utilities for NLP datasets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = "label",
        padding: bool | str = "max_length",
        truncation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.padding = padding
        self.truncation = truncation

    def __call__(self, examples):
        tokenized = self.tokenizer(
            examples[self.text_column],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        if self.label_column in examples:
            tokenized["labels"] = examples[self.label_column]
        return tokenized
