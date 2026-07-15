import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_HIGHLIGHT = "\033[1;93m"
_RESET = "\033[0m"


def infer(model_dir: str, test_strings: list[str]) -> None:
    """
    Run inference with a text model on given input strings, printing the output.

    Parameters
    ----------
    model_dir : str
        Directory of saved model
    test_strings : list[str]
        Strings to classify (each run independently)
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    label_names = model.config.id2label

    for idx, test_string in enumerate(test_strings):
        if idx > 0:
            print()
        print(f"Classifying: {test_string}")

        encoded_input = tokenizer(
            test_string,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)

        for i, prob in enumerate(probs[0]):
            pct = prob.item()
            line = f"Probability of label `{label_names[i]}`: {pct:.2%}"
            if pct > 0.5:
                line = f"{_HIGHLIGHT}{line}{_RESET}"
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Androcles")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Model folder path",
    )

    parser.add_argument(
        "--test_string",
        type=str,
        nargs="+",
        required=True,
        help="One or more input strings to classify",
    )

    args = parser.parse_args()

    infer(model_dir=args.model_dir, test_strings=args.test_string)
