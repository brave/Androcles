#!/usr/bin/env python3
"""
Gap detector that identifies topics with low F1 scores or insufficient training data.

Reads per-label metrics JSON and ranks topics by performance gaps and data sparsity.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def detect_gaps(
    metrics_json_path: str,
    train_csv_path: Optional[str] = None,
    min_f1_threshold: float = 0.82,
    min_samples_threshold: int = 50,
    top_k: int = 10,
) -> List[Dict[str, any]]:
    """
    Detect gaps in model performance and data coverage.

    Parameters
    ----------
    metrics_json_path : str
        Path to per-label metrics JSON file
    train_csv_path : str, optional
        Path to training CSV to count samples per label
    min_f1_threshold : float
        F1 threshold below which a topic is considered underperforming
    min_samples_threshold : int
        Minimum samples per topic to consider it well-covered
    top_k : int
        Number of top gaps to return

    Returns
    -------
    list[dict]
        List of gap reports, each containing:
        - topic: label name
        - f1: current F1 score
        - precision: current precision
        - recall: current recall
        - support: number of samples in eval set
        - train_samples: number of samples in training set (if train_csv_path provided)
        - gap_type: "model_gap" (low F1) or "data_gap" (low samples) or "both"
        - priority_score: combined score for ranking (higher = more urgent)
    """
    # Load metrics
    with open(metrics_json_path, 'r') as f:
        per_label_metrics = json.load(f)

    # Load training data counts if provided
    train_counts = {}
    if train_csv_path and Path(train_csv_path).exists():
        import pandas as pd
        df = pd.read_csv(train_csv_path)
        # Count positive samples per label (assuming binary columns)
        for label in per_label_metrics.keys():
            if label in df.columns:
                train_counts[label] = int(df[label].sum())

    gaps = []
    for topic, metrics in per_label_metrics.items():
        f1 = metrics.get("f1", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        support = metrics.get("support", 0)
        train_samples = train_counts.get(topic, None)

        # Determine gap type
        has_model_gap = f1 < min_f1_threshold
        has_data_gap = train_samples is not None and train_samples < min_samples_threshold

        if not (has_model_gap or has_data_gap):
            continue  # Skip topics that meet both thresholds

        gap_type = "both" if (has_model_gap and has_data_gap) else (
            "model_gap" if has_model_gap else "data_gap"
        )

        # Calculate priority score
        # Higher priority for:
        # - Lower F1 scores (more urgent to fix)
        # - Lower training sample counts (if data gap)
        # - Higher support (more important in eval set)
        f1_gap = max(0, min_f1_threshold - f1)  # How far below threshold
        data_gap = 0
        if train_samples is not None:
            data_gap = max(0, min_samples_threshold - train_samples) / min_samples_threshold

        priority_score = (
            f1_gap * 10.0 +  # F1 gap weighted heavily
            data_gap * 5.0 +  # Data gap weighted moderately
            support / 100.0   # Support as tiebreaker (normalized)
        )

        gaps.append({
            "topic": topic,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
            "train_samples": train_samples,
            "gap_type": gap_type,
            "priority_score": priority_score,
        })

    # Sort by priority score (descending)
    gaps.sort(key=lambda x: x["priority_score"], reverse=True)

    return gaps[:top_k]


def generate_gap_report(
    gaps: List[Dict[str, any]],
    output_path: str,
) -> None:
    """
    Generate a JSON report of detected gaps.

    Parameters
    ----------
    gaps : list[dict]
        List of gap dictionaries from detect_gaps()
    output_path : str
        Path to save gap report JSON
    """
    report = {
        "total_gaps": len(gaps),
        "gaps": gaps,
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetected {len(gaps)} gaps:")
    print("=" * 80)
    for i, gap in enumerate(gaps, 1):
        print(f"{i}. {gap['topic']:30s} "
              f"F1: {gap['f1']:.4f}  "
              f"Type: {gap['gap_type']:12s}  "
              f"Priority: {gap['priority_score']:.2f}")
        if gap['train_samples'] is not None:
            print(f"   Training samples: {gap['train_samples']}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect performance and data gaps from evaluation metrics"
    )
    parser.add_argument("--metrics_json", type=str, required=True,
                        help="Path to per-label metrics JSON")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to training CSV (optional, for data gap detection)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save gap report JSON")
    parser.add_argument("--min_f1_threshold", type=float, default=0.82,
                        help="F1 threshold below which topic is underperforming")
    parser.add_argument("--min_samples_threshold", type=int, default=50,
                        help="Minimum training samples per topic")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top gaps to return")

    args = parser.parse_args()

    gaps = detect_gaps(
        metrics_json_path=args.metrics_json,
        train_csv_path=args.train_csv,
        min_f1_threshold=args.min_f1_threshold,
        min_samples_threshold=args.min_samples_threshold,
        top_k=args.top_k,
    )

    generate_gap_report(gaps, args.output)
