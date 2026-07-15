#!/usr/bin/env python3
"""
Data manager for active learning pipeline.

Handles merging new data with existing training set, running the multilabel labeller,
filtering by length, and managing the frozen eval split.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def merge_csv_files(
    existing_train_csv: Optional[str],
    new_data_csv: str,
    output_csv: str,
) -> None:
    """
    Merge new data CSV with existing training CSV.

    Parameters
    ----------
    existing_train_csv : str, optional
        Path to existing training CSV (if None, just copy new_data_csv)
    new_data_csv : str
        Path to new data CSV to add
    output_csv : str
        Path to save merged CSV
    """
    if existing_train_csv and Path(existing_train_csv).exists():
        existing_df = pd.read_csv(existing_train_csv)
        new_df = pd.read_csv(new_data_csv)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"Merged {len(existing_df)} existing + {len(new_df)} new = {len(merged_df)} total rows")
    else:
        merged_df = pd.read_csv(new_data_csv)
        print(f"No existing training data, using {len(merged_df)} new rows")

    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged dataset to {output_csv}")


async def run_multilabel_labeller(
    input_csv: str,
    output_csv: str,
    config_yaml: Optional[str] = None,
    max_concurrent: int = 10,
) -> None:
    """
    Run the multilabel labeller on a CSV file.

    Parameters
    ----------
    input_csv : str
        Path to input CSV (must have "original topic" and "message" columns)
    output_csv : str
        Path to save labeled CSV
    config_yaml : str, optional
        Path to YAML config file for labeller
    max_concurrent : int
        Maximum concurrent labeling tasks
    """
    print(f"Running multilabel labeller on {input_csv}...")
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "dataset_generator" / "multilabel_labeller.py"),
        "--input", input_csv,
        "--output", output_csv,
        "--max-concurrent", str(max_concurrent),
    ]
    
    if config_yaml:
        cmd.extend(["--config", config_yaml])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Multilabel labeller failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    
    print(f"Labeling complete. Output saved to {output_csv}")


def filter_dataset(
    input_csv: str,
    output_csv: str,
    base_model: str,
    max_context_length: int,
    input_col: str = "message",
) -> None:
    """
    Filter dataset by length - truncate samples that exceed max_context_length.

    Parameters
    ----------
    input_csv : str
        Path to input CSV
    output_csv : str
        Path to save filtered CSV
    base_model : str
        Base model name for tokenizer
    max_context_length : int
        Maximum context length in tokens
    input_col : str
        Name of column with input text
    """
    print(f"Filtering dataset {input_csv}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load CSV
    df = pd.read_csv(input_csv)
    
    print(f"Original dataset size: {len(df)}")
    
    # Truncate messages that are too long
    truncated_count = 0
    for idx, row in df.iterrows():
        text = str(row[input_col])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) > max_context_length - 2:
            # Truncate to fit
            truncated_tokens = tokens[:max_context_length - 2]
            df.at[idx, input_col] = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            truncated_count += 1
    
    if truncated_count > 0:
        print(f"Truncated {truncated_count} samples that exceeded {max_context_length} tokens")
    
    # Save filtered CSV
    df.to_csv(output_csv, index=False)
    print(f"Filtering complete. Output saved to {output_csv}")


def split_train_eval(
    input_csv: str,
    train_csv: str,
    eval_csv: str,
    test_size: float = 0.15,
    random_state: int = 42,
    frozen_eval_csv: Optional[str] = None,
    input_col: str = "message",
) -> None:
    """
    Split dataset into train/eval sets.

    If frozen_eval_csv is provided, use that as eval set and put all new data in train.
    Otherwise, do a fresh split.

    Ensures all label columns have positive samples in both train and eval sets.

    Parameters
    ----------
    input_csv : str
        Path to input CSV
    train_csv : str
        Path to save training CSV
    eval_csv : str
        Path to save eval CSV
    test_size : float
        Fraction of data for eval set
    random_state : int
        Random seed for reproducibility
    frozen_eval_csv : str, optional
        Path to frozen eval CSV (if provided, use this as eval and put all input in train)
    input_col : str
        Name of column with input text (default: message)
    """
    df = pd.read_csv(input_csv)
    
    # Identify label columns (all columns except input_col)
    label_cols = [col for col in df.columns if col != input_col]
    
    if frozen_eval_csv and Path(frozen_eval_csv).exists():
        # Use frozen eval set
        print(f"Using frozen eval set from {frozen_eval_csv}")
        eval_df = pd.read_csv(frozen_eval_csv)
        
        # All input data goes to training
        train_df = df.copy()
        
        print(f"Train: {len(train_df)} rows, Eval: {len(eval_df)} rows (frozen)")
        
        # Verify all labels are present in eval set
        eval_label_cols = [col for col in eval_df.columns if col != input_col]
        missing_in_eval = [col for col in label_cols if col not in eval_label_cols]
        if missing_in_eval:
            print(f"Warning: Labels missing in frozen eval set: {missing_in_eval}")
            # Add missing labels with zeros
            for label in missing_in_eval:
                eval_df[label] = 0
        
        # Check for labels with no positive samples in eval
        labels_without_positives = []
        for label in label_cols:
            if label in eval_df.columns and eval_df[label].sum() == 0:
                labels_without_positives.append(label)
        
        if labels_without_positives:
            error_msg = (
                f"ERROR: Labels with no positive samples in frozen eval set: {labels_without_positives}. "
                f"This will cause evaluation to fail. The frozen eval set must have at least one "
                f"positive sample for each label."
            )
            print(error_msg)
            raise ValueError(error_msg)
    else:
        # Fresh split - ensure all labels have positive samples in both sets
        max_attempts = 100
        for attempt in range(max_attempts):
            train_df, eval_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state + attempt,  # Try different seeds
            )
            
            # Check if all labels have positive samples in both sets
            all_labels_represented = True
            missing_labels = []
            for label in label_cols:
                train_positive = train_df[label].sum() > 0 if label in train_df.columns else False
                eval_positive = eval_df[label].sum() > 0 if label in eval_df.columns else False
                
                if not train_positive:
                    missing_labels.append(f"{label} (train)")
                    all_labels_represented = False
                if not eval_positive:
                    missing_labels.append(f"{label} (eval)")
                    all_labels_represented = False
            
            if all_labels_represented:
                break
            
            if attempt == max_attempts - 1:
                # Last attempt failed - check if it's because some labels have no positives at all
                labels_with_no_positives = []
                for label in label_cols:
                    if df[label].sum() == 0:
                        labels_with_no_positives.append(label)
                
                if labels_with_no_positives:
                    error_msg = (
                        f"ERROR: Labels with no positive samples in entire dataset: {labels_with_no_positives}. "
                        f"Cannot create valid train/eval split. Check your data generation and labeling."
                    )
                    print(error_msg)
                    raise ValueError(error_msg)
                else:
                    error_msg = (
                        f"ERROR: Failed to create train/eval split with all labels represented after {max_attempts} attempts. "
                        f"Labels missing: {missing_labels}. This may indicate insufficient data or poor label distribution."
                    )
                    print(error_msg)
                    raise ValueError(error_msg)
        
        print(f"Split {len(df)} rows -> Train: {len(train_df)}, Eval: {len(eval_df)}")
        print(f"All {len(label_cols)} labels have positive samples in both train and eval sets")
    
    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    
    print(f"Saved train set to {train_csv}")
    print(f"Saved eval set to {eval_csv}")


def process_new_data(
    new_raw_csv: str,
    existing_train_csv: Optional[str],
    frozen_eval_csv: Optional[str],
    config_yaml: Optional[str],
    base_model: str,
    max_context_length: int,
    test_size: float,
    output_dir: str,
    iteration: int,
    input_col: str = "message",
) -> Dict[str, str]:
    """
    Process new data through the full pipeline:
    1. Merge with existing training data
    2. Run multilabel labeller
    3. Filter by length
    4. Split into train/eval (using frozen eval if available)

    Parameters
    ----------
    new_raw_csv : str
        Path to new raw CSV (from topic_search_generator or gap_analysis_generator)
    existing_train_csv : str, optional
        Path to existing training CSV to merge with
    frozen_eval_csv : str, optional
        Path to frozen eval CSV (from iteration 0)
    config_yaml : str, optional
        Path to YAML config for multilabel labeller
    base_model : str
        Base model name for tokenizer
    max_context_length : int
        Maximum context length
    test_size : float
        Test split size (only used if no frozen eval)
    output_dir : str
        Directory to save outputs
    iteration : int
        Current iteration number
    input_col : str
        Name of column with input text (default: message)

    Returns
    -------
    dict
        Dictionary with paths to:
        - labeled_csv: Labeled CSV (before filtering)
        - filtered_csv: Filtered CSV (after length filtering)
        - train_csv: Training CSV
        - eval_csv: Eval CSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Merge with existing training data (if this is not iteration 0)
    if iteration > 0 and existing_train_csv:
        merged_raw_csv = output_path / f"iter_{iteration}_merged_raw.csv"
        merge_csv_files(existing_train_csv, new_raw_csv, str(merged_raw_csv))
        input_for_labeler = str(merged_raw_csv)
    else:
        input_for_labeler = new_raw_csv
    
    # Step 2: Run multilabel labeller
    labeled_csv = output_path / f"iter_{iteration}_labeled.csv"
    asyncio.run(run_multilabel_labeller(
        input_csv=input_for_labeler,
        output_csv=str(labeled_csv),
        config_yaml=config_yaml,
    ))
    
    # Step 3: Filter by length
    filtered_csv = output_path / f"iter_{iteration}_filtered.csv"
    filter_dataset(
        input_csv=str(labeled_csv),
        output_csv=str(filtered_csv),
        base_model=base_model,
        max_context_length=max_context_length,
    )
    
    # Step 4: Split into train/eval
    train_csv = output_path / f"iter_{iteration}_train.csv"
    eval_csv = output_path / f"iter_{iteration}_eval.csv"
    split_train_eval(
        input_csv=str(filtered_csv),
        train_csv=str(train_csv),
        eval_csv=str(eval_csv),
        test_size=test_size,
        frozen_eval_csv=frozen_eval_csv,
        input_col=input_col,
    )
    
    return {
        "labeled_csv": str(labeled_csv),
        "filtered_csv": str(filtered_csv),
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process new data through the active learning pipeline"
    )
    parser.add_argument("--new_raw_csv", type=str, required=True)
    parser.add_argument("--existing_train_csv", type=str, default=None)
    parser.add_argument("--frozen_eval_csv", type=str, default=None)
    parser.add_argument("--config_yaml", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--max_context_length", type=int, default=1024)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)

    args = parser.parse_args()

    results = process_new_data(
        new_raw_csv=args.new_raw_csv,
        existing_train_csv=args.existing_train_csv,
        frozen_eval_csv=args.frozen_eval_csv,
        config_yaml=args.config_yaml,
        base_model=args.base_model,
        max_context_length=args.max_context_length,
        test_size=args.test_size,
        output_dir=args.output_dir,
        iteration=args.iteration,
    )

    print("\nProcessing complete. Output files:")
    for key, path in results.items():
        print(f"  {key}: {path}")
