import argparse
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EvalPrediction,
)


def multi_label_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, threshold=0.5
) -> Dict:
    """
    Calculate evaluation metrics for multi-label classification

    Parameters
    ----------
    predictions : torch.Tensor
        Raw model predictions of shape (batch_size, num_labels)
    labels : torch.Tensor
        Ground truth labels of shape (batch_size, num_labels)
    threshold : float
        Threshold for converting probabilities to binary predictions

    Returns
    -------
    dict: Dict
        Dictionary of {metric_name: metric_value}
    """
    logits = np.asarray(predictions, dtype=np.float64)
    y_true_raw = np.asarray(labels)

    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if y_true_raw.ndim == 1:
        y_true_raw = y_true_raw.reshape(-1, 1)

    if y_true_raw.shape != logits.shape:
        raise ValueError(
            f"Label/prediction shape mismatch: labels {y_true_raw.shape}, "
            f"predictions {logits.shape}"
        )

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(logits)).numpy()

    y_true = (y_true_raw >= 0.5).astype(np.int8)
    y_pred = (probs >= threshold).astype(np.int8)

    f1_micro_average = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probs,
        average="micro",
    )

    accuracy = accuracy_score(y_true, y_pred)

    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}

    return metrics


def compute_metrics(p: EvalPrediction) -> Dict:
    """
    Compute the metrics based on model outputs

    Parameters
    ----------
    p : EvalPrediction
        Prediction from the model

    Returns
    -------
    result : Dict
        A dictionary of metric names (str) to metric values (float)
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    result = multi_label_metrics(predictions=preds, labels=p.label_ids)

    return result


def train(
    train_file: str,
    base_model: str,
    max_context_length: int,
    input_col: str,
    batch_size: int,
    best_metric_name: str,
    checkpoint_dir: str,
    output_dir: str,
    test_size: float,
    split_seed: int,
    eval_output: Optional[str],
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
) -> None:
    """
    Train a multilabel text classifier

    Parameters
    ----------
    train_file : str
        Path to training file
    base_model : str
        Name of model to use as base for finetuning
    max_context_length : int
        Maximum number of tokens to use in the model
    input_col : str
        Name of column with input text
    batch_size : int
        Batch size to use while training
    best_metric_name : str
        Metric to use for selecting best model
    checkpoint_dir : str
        Directory to save checkpoints to
    output_dir : str
        Directory to save best model to after training
    test_size : float
        Train:Test split for the data
    split_seed : int
        RNG seed for the train/eval split (reproducible eval set)
    eval_output : str, optional
        If set, path to write the held-out eval split as CSV (for evaluate.py)
    learning_rate : float
        Learning rate for training
    num_epochs : int
        Number of training epochs
    weight_decay : float
        Rate of weight decay during training
    """
    dataset = Dataset.from_csv(train_file)

    dataset = dataset.train_test_split(test_size=test_size, seed=split_seed)

    if eval_output:
        dataset["test"].to_pandas().to_csv(eval_output, index=False)
        print(f"Saved eval split ({len(dataset['test'])} rows) to {eval_output}")

    labels = [
        label for label in dataset["train"].features.keys() if label not in [input_col]
    ]

    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess_data(examples):
        text = examples[input_col]

        encoding = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_context_length
        )

        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}

        labels_matrix = np.zeros((len(text), len(labels)))

        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    encoded_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset["train"].column_names
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric_name,
        bf16=use_bf16,
        dataloader_num_workers=8 if use_cuda else 0,
        dataloader_pin_memory=use_cuda,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    trainer.save_model("test_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Androcles")

    # Add arguments
    parser.add_argument(
        "--train_file", type=str, default="train.csv", help="Path to training file"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Name of model to use as base for finetuning",
    )

    parser.add_argument(
        "--max_context_length",
        type=int,
        default=512,
        help="Maximum number of tokens to use in the model",
    )

    parser.add_argument(
        "--input_col",
        type=str,
        default="message",
        help="Name of column with input text",
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size to use while training"
    )

    parser.add_argument(
        "--best_metric_name",
        type=str,
        default="f1",
        help="Metric to use for selecting best model",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints to",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_model",
        help="Directory to save best model to after training",
    )

    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Train:Test split for the data"
    )

    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Seed for train/eval split (same seed + file reproduces the same eval rows)",
    )

    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="If set, write the held-out eval split to this CSV (e.g. eval.csv for evaluate.py)",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Rate of weight decay during training",
    )

    args = parser.parse_args()

    train(**vars(args))
