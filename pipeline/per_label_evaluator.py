#!/usr/bin/env python3
"""
Per-label evaluator wrapper that runs evaluate.py and returns structured per-label metrics.

This module provides a programmatic interface to the evaluation script,
extracting per-label F1, precision, recall, and support metrics.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def run_evaluation(
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
    metrics_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation and return per-label metrics.

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
    metrics_json_path : str, optional
        Path to save per-label metrics JSON (if None, uses temp file)

    Returns
    -------
    dict
        Dictionary with keys:
        - per_label_metrics: Dict mapping label_name -> {f1, precision, recall, support}
        - aggregate_metrics: Dict with overall f1_score, precision, recall, etc.
    """
    # Import evaluate function directly
    sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
    from evaluate import evaluate

    # Use temp file for metrics if not provided
    temp_metrics_file = None
    if metrics_json_path is None:
        import tempfile
        temp_metrics_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        metrics_json_path = temp_metrics_file.name
        temp_metrics_file.close()

    try:
        # Run evaluation
        results = evaluate(
            model_dir=model_dir,
            test_file=test_file,
            input_col=input_col,
            label_col=label_col,
            max_context_length=max_context_length,
            batch_size=batch_size,
            output_file=output_file,
            use_chunking=use_chunking,
            chunk_overlap=chunk_overlap,
            probabilities_file=probabilities_file,
            max_samples=max_samples,
            prechunked_file=prechunked_file,
            save_every_n_batches=save_every_n_batches,
            output_metrics_json=metrics_json_path,
        )

        # Load per-label metrics if available
        per_label_metrics = {}
        if Path(metrics_json_path).exists():
            with open(metrics_json_path, 'r') as f:
                per_label_metrics = json.load(f)
        elif "per_label_metrics" in results:
            per_label_metrics = results["per_label_metrics"]

        return {
            "per_label_metrics": per_label_metrics,
            "aggregate_metrics": {
                "f1_score": results.get("f1_score"),
                "precision": results.get("precision"),
                "recall": results.get("recall"),
                "balanced_accuracy": results.get("balanced_accuracy"),
                "total_samples": results.get("total_samples"),
            },
        }
    finally:
        # Clean up temp file if we created it
        if temp_metrics_file and Path(metrics_json_path).exists():
            Path(metrics_json_path).unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation and extract per-label metrics"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--metrics_json", type=str, required=True,
                        help="Path to save per-label metrics JSON")
    parser.add_argument("--input_col", type=str, default="message")
    parser.add_argument("--label_col", type=str, default="malicious")
    parser.add_argument("--max_context_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_chunking", action="store_true")
    parser.add_argument("--prechunked_file", type=str, default=None)

    args = parser.parse_args()

    results = run_evaluation(
        model_dir=args.model_dir,
        test_file=args.test_file,
        input_col=args.input_col,
        label_col=args.label_col,
        max_context_length=args.max_context_length,
        batch_size=args.batch_size,
        use_chunking=args.use_chunking,
        prechunked_file=args.prechunked_file,
        metrics_json_path=args.metrics_json,
    )

    print("\nPer-label metrics:")
    print(json.dumps(results["per_label_metrics"], indent=2))
