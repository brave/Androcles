#!/usr/bin/env python3
"""
Model evaluation script for multi-label classification.

Computes per-label metrics (F1, precision, recall, support) and saves to JSON.
Can be run as a command-line script or imported as a module.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate(
    model_dir: str,
    test_file: str,
    input_col: str = "message",
    label_col: str = "malicious",
    max_context_length: int = 8192,
    batch_size: int = 8,
    output_file: Optional[str] = None,
    use_chunking: bool = False,
    chunk_overlap: int = 0,
    probabilities_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    prechunked_file: Optional[str] = None,
    save_every_n_batches: Optional[int] = None,
    output_metrics_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a multi-label classification model and compute per-label metrics.

    Parameters
    ----------
    model_dir : str
        Directory containing the trained model
    test_file : str
        Path to test CSV file
    input_col : str
        Name of column with input text (default: message)
    label_col : str
        Name of column with labels (default: malicious)
        Note: For multi-label, expects multiple binary columns
    max_context_length : int
        Maximum context length (default: 8192)
    batch_size : int
        Batch size for evaluation (default: 8)
    output_file : str, optional
        Path to save full JSON output
    use_chunking : bool
        Enable chunking for long inputs (default: False)
    chunk_overlap : int
        Token overlap between chunks (default: 0)
    probabilities_file : str, optional
        Path to save probabilities CSV
    max_samples : int, optional
        Maximum samples to evaluate
    prechunked_file : str, optional
        Path to pre-chunked CSV file
    save_every_n_batches : int, optional
        Save intermediate results every N batches
    output_metrics_json : str, optional
        Path to save per-label metrics JSON

    Returns
    -------
    dict
        Dictionary containing:
        - per_label_metrics: Dict mapping label_name -> {f1, precision, recall, support}
        - aggregate_metrics: Dict with overall metrics
        - f1_score: Overall micro-averaged F1
        - precision: Overall micro-averaged precision
        - recall: Overall micro-averaged recall
        - balanced_accuracy: Balanced accuracy score
        - total_samples: Number of samples evaluated
    """
    # Load model and tokenizer
    # Check if model directory has the model files, otherwise look in checkpoints
    model_path = Path(model_dir)
    if not (model_path / "config.json").exists():
        # Check if there's a checkpoints directory
        checkpoints_dir = model_path / "checkpoints"
        if checkpoints_dir.exists():
            # Find the best checkpoint (highest checkpoint number, or check trainer_state.json)
            checkpoint_dirs = sorted(
                [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]) if "-" in x.name else 0,
                reverse=True
            )
            if checkpoint_dirs:
                # Use the latest checkpoint (or we could check trainer_state.json for best)
                model_path = checkpoint_dirs[0]
                print(f"Model not found in {model_dir}, using checkpoint: {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model not found in {model_dir} and no checkpoints found in {checkpoints_dir}"
                )
        else:
            raise FileNotFoundError(
                f"Model directory {model_dir} does not contain config.json and no checkpoints directory found"
            )
    
    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.eval()

    # Get label names from model config
    id2label = model.config.id2label
    label_names = [id2label[i] for i in range(len(id2label))]

    # Load test dataset
    print(f"Loading test data from {test_file}")
    df = pd.read_csv(test_file)

    # Limit samples if requested
    if max_samples:
        df = df.head(max_samples)

    # Identify label columns (all columns except input_col)
    label_cols = [col for col in df.columns if col != input_col]
    
    # Ensure all model labels are present in the dataset
    missing_labels = [label for label in label_names if label not in label_cols]
    if missing_labels:
        print(f"Warning: Missing labels in dataset: {missing_labels}")
        print("Adding missing labels with zeros")
        for label in missing_labels:
            df[label] = 0

    # Ensure dataset columns match model labels (in correct order)
    label_cols = [label for label in label_names if label in df.columns]

    # Prepare data
    texts = df[input_col].tolist()
    labels = df[label_cols].values.astype(np.float32)

    print(f"Evaluating {len(texts)} samples with {len(label_cols)} labels")

    # Run inference in batches
    print("Running inference...")
    all_predictions = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # Tokenize batch
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_context_length,
                return_tensors="pt",
            )
            batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}

            # Get predictions
            outputs = model(**batch_encodings)
            logits = outputs.logits

            # Apply sigmoid and threshold
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits)
            preds = (probs >= 0.5).cpu().numpy().astype(int)

            all_predictions.append(preds)
            all_labels.append(batch_labels)

            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_texts)}/{len(texts)} samples")

    # Concatenate all predictions and labels
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    print("Computing metrics...")

    # Compute per-label metrics
    per_label_metrics = {}
    labels_without_positive_samples = []
    for idx, label_name in enumerate(label_cols):
        y_true = all_labels[:, idx]
        y_pred = all_predictions[:, idx]

        # Check if no positive samples in ground truth
        if y_true.sum() == 0:
            labels_without_positive_samples.append(label_name)
            continue

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        support = int(y_true.sum())

        per_label_metrics[label_name] = {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "support": support,
        }

    # Fail if any labels have no positive samples - this should never happen
    if labels_without_positive_samples:
        error_msg = (
            f"ERROR: Labels with no positive samples in test set: {labels_without_positive_samples}. "
            f"This indicates a problem with the dataset split or labeling. "
            f"All labels in the model must have at least one positive sample in the evaluation set."
        )
        print(error_msg)
        raise ValueError(error_msg)

    # Compute aggregate metrics
    f1_micro = f1_score(all_labels, all_predictions, average="micro", zero_division=0)
    precision_micro = precision_score(
        all_labels, all_predictions, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        all_labels, all_predictions, average="micro", zero_division=0
    )

    # Compute balanced accuracy (requires binary predictions per sample)
    # For multi-label, we'll compute it as the average per-label accuracy
    per_label_accuracies = []
    for idx in range(len(label_cols)):
        y_true = all_labels[:, idx]
        y_pred = all_predictions[:, idx]
        if y_true.sum() > 0:  # Only compute if there are positive samples
            acc = accuracy_score(y_true, y_pred)
            per_label_accuracies.append(acc)

    balanced_acc = (
        np.mean(per_label_accuracies) if per_label_accuracies else 0.0
    )

    results = {
        "per_label_metrics": per_label_metrics,
        "aggregate_metrics": {
            "f1_score": float(f1_micro),
            "precision": float(precision_micro),
            "recall": float(recall_micro),
            "balanced_accuracy": float(balanced_acc),
            "total_samples": len(texts),
        },
        "f1_score": float(f1_micro),
        "precision": float(precision_micro),
        "recall": float(recall_micro),
        "balanced_accuracy": float(balanced_acc),
        "total_samples": len(texts),
    }

    # Save per-label metrics JSON if requested
    if output_metrics_json:
        print(f"Saving per-label metrics to {output_metrics_json}")
        with open(output_metrics_json, "w") as f:
            json.dump(per_label_metrics, f, indent=2)

    # Save full output if requested
    if output_file:
        print(f"Saving full results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Total samples: {len(texts)}")
    print(f"Labels evaluated: {len(per_label_metrics)}")
    print("\nAggregate Metrics:")
    print(f"  F1 (micro): {f1_micro:.4f}")
    print(f"  Precision (micro): {precision_micro:.4f}")
    print(f"  Recall (micro): {recall_micro:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print("\nPer-Label Metrics:")
    for label_name, metrics in sorted(per_label_metrics.items()):
        print(
            f"  {label_name}: F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"Support={metrics['support']}"
        )
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate multi-label classification model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained model",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--input_col",
        type=str,
        default="message",
        help="Name of column with input text (default: message)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="malicious",
        help="Name of column with labels (default: malicious). "
        "Note: For multi-label, expects multiple binary columns",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=1024,
        help="Maximum context length (default: 1024)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save full JSON output",
    )
    parser.add_argument(
        "--output_metrics_json",
        type=str,
        default=None,
        help="Path to save per-label metrics JSON",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )

    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        test_file=args.test_file,
        input_col=args.input_col,
        label_col=args.label_col,
        max_context_length=args.max_context_length,
        batch_size=args.batch_size,
        output_file=args.output_file,
        output_metrics_json=args.output_metrics_json,
        max_samples=args.max_samples,
    )
