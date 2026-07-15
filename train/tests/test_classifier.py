import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_model_loading(model_path):
    """Test that the model loads correctly."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    assert model is not None
    assert model.config.problem_type == "multi_label_classification" or model.config.num_labels >= 1


def test_tokenizer_loading(model_path):
    """Test that the tokenizer loads correctly."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    assert tokenizer is not None


def test_batch_inference(model_path):
    """Test model inference with different batch sizes; probs in [0, 1] via sigmoid."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dummy_texts = [
        "Write a Python script",
        "Translate this paragraph from French to English",
        "Generate an image of two dogs",
        "What's the weather in London tomorrow?",
    ] * (32 // 4)
    tokenized_texts = tokenizer(
        dummy_texts, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        results = model(**tokenized_texts)

    assert results.logits.shape[0] == len(dummy_texts)
    assert results.logits.shape[1] == model.config.num_labels

    probs = torch.sigmoid(results.logits)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("Write Python code to parse a CSV", "Coding"),
        ("Generate an image of two dogs", "Image Generation"),
        ("Translate this paragraph from French to English", "Translation"),
        ("Rispondi in italiano per favore", "Multilingualism"),
    ],
)
def test_specific_cases(model_path, text, expected_label):
    """
    Smoke-test that known prompts fire the expected Androcles 2 label.

    Skips if the model was not trained with that label name (e.g. older checkpoints).
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    label2id = model.config.label2id or {}
    if expected_label not in label2id:
        pytest.skip(f"Label {expected_label!r} not in model.config.label2id")

    tokenized = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**tokenized).logits

    probs = torch.sigmoid(logits)[0]
    idx = label2id[expected_label]
    assert probs[idx].item() > 0.5, (
        f"Expected {expected_label} > 0.5 for {text!r}, got {probs[idx].item():.3f}"
    )
