#!/usr/bin/env python3
"""
Active Learning Pipeline Orchestrator

Main loop that orchestrates:
1. Initial dataset generation (iteration 0)
2. Training
3. Evaluation
4. Gap detection
5. Targeted data generation (iterations 1+)
6. Convergence checking

State is saved to pipeline/runs/<run_id>/ for resumability.
"""

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load active learning config YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_iteration_state(
    run_dir: Path,
    iteration: int,
    state: Dict[str, Any],
) -> None:
    """Save iteration state to JSON."""
    state_file = run_dir / f"iter_{iteration}" / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_iteration_state(run_dir: Path, iteration: int) -> Optional[Dict[str, Any]]:
    """Load iteration state from JSON."""
    state_file = run_dir / f"iter_{iteration}" / "state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


def generate_initial_dataset(
    config: Dict[str, Any],
    output_csv: str,
    base_config_yaml: str,
) -> None:
    """Generate initial dataset using topic_search_generator."""
    # Check if dataset already exists
    output_path = Path(output_csv)
    if output_path.exists():
        print(f"\n{'='*80}")
        print("ITERATION 0: Dataset already exists, skipping generation")
        print(f"Existing dataset: {output_csv}")
        print(f"{'='*80}")
        return
    
    print(f"\n{'='*80}")
    print("ITERATION 0: Generating initial dataset")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "dataset_generator" / "topic_search_generator.py"),
        "--config", base_config_yaml,
        "--output", output_csv.replace(".csv", ""),  # Script adds .csv automatically
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Initial dataset generation failed with code {result.returncode}")


def generate_gap_filling_data(
    config: Dict[str, Any],
    existing_train_csv: str,
    gap_report_json: str,
    output_csv: str,
    base_config_yaml: str,
    metrics_json: str,
    messages_per_topic: int,
) -> None:
    """Generate gap-filling data using gap_analysis_generator."""
    # Check if dataset already exists
    output_path = Path(output_csv)
    if output_path.exists():
        print(f"\n{'='*80}")
        print("Gap-filling dataset already exists, skipping generation")
        print(f"Existing dataset: {output_csv}")
        print(f"{'='*80}")
        return
    
    print(f"\n{'='*80}")
    print("Generating gap-filling data")
    print(f"{'='*80}")
    
    # Load gap report to get topics to target
    with open(gap_report_json, 'r') as f:
        gap_report = json.load(f)
    
    gaps = gap_report.get("gaps", [])
    if not gaps:
        print("No gaps detected. Skipping data generation.")
        return
    
    # Extract topics from gaps
    topics = [gap["topic"] for gap in gaps]
    
    print(f"Targeting {len(topics)} topics: {', '.join(topics[:5])}...")
    
    # Create a temporary YAML config for gap generation
    import tempfile
    
    # Load base config to merge with
    with open(base_config_yaml, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create gap config that extends base config
    gap_config = base_config.copy()
    gap_config["topics"] = topics
    gap_config["generation"] = gap_config.get("generation", {})
    gap_config["generation"]["num_primary_topics"] = 1  # One primary per gap topic
    gap_config["generation"]["num_topics"] = 5  # 5 secondary topics per primary
    gap_config["generation"]["messages_per_topic"] = messages_per_topic
    gap_config["output"] = gap_config.get("output", {})
    gap_config["output"]["file"] = output_csv.replace(".csv", "")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(gap_config, f)
        gap_config_yaml = f.name
    
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "dataset_generator" / "gap_analysis_generator.py"),
            "--config", gap_config_yaml,
            "--existing-messages", existing_train_csv,
            "--metrics-json", metrics_json,
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            raise RuntimeError(f"Gap-filling data generation failed with code {result.returncode}")
    finally:
        # Clean up temp config
        Path(gap_config_yaml).unlink()


def get_training_config_for_iteration(
    config: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    """
    Get training configuration for a specific iteration.
    
    Supports per-iteration overrides via training.iterations[iteration] or
    training.iterations[default] for fallback.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Full config dictionary
    iteration : int
        Current iteration number
    
    Returns
    -------
    Dict[str, Any]
        Training config for this iteration (merged with base config)
    """
    training_config = config.get("training", {}).copy()
    
    # Check for per-iteration overrides
    iterations_config = training_config.pop("iterations", {})
    
    # Try to get iteration-specific config
    iter_config = None
    if isinstance(iterations_config, dict):
        # Check for exact iteration match (handle both int and str keys)
        if iteration in iterations_config:
            iter_config = iterations_config[iteration]
        elif str(iteration) in iterations_config:
            iter_config = iterations_config[str(iteration)]
        # Check for default fallback
        elif "default" in iterations_config:
            iter_config = iterations_config["default"]
    
    # Merge iteration-specific config with base config
    if iter_config:
        # Deep merge: base config first, then iteration-specific overrides
        merged_config = training_config.copy()
        merged_config.update(iter_config)
        return merged_config
    
    return training_config


def train_model(
    train_csv: str,
    output_dir: str,
    config: Dict[str, Any],
    iteration: int = 0,
) -> str:
    """
    Train model using train.py.
    
    Parameters
    ----------
    train_csv : str
        Path to training CSV file
    output_dir : str
        Directory to save model outputs
    config : Dict[str, Any]
        Full config dictionary
    iteration : int
        Current iteration number (for per-iteration training params)
    
    Returns
    -------
    str
        Path to trained model directory
    """
    print(f"\n{'='*80}")
    print(f"Training model (iteration {iteration})")
    print(f"{'='*80}")
    
    training_config = get_training_config_for_iteration(config, iteration)
    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "train" / "train.py"),
        "--train_file", train_csv,
        "--base_model", training_config.get("base_model", "answerdotai/ModernBERT-base"),
        "--max_context_length", str(training_config.get("max_context_length", 1024)),
        "--num_epochs", str(training_config.get("num_epochs", 5)),
        "--batch_size", str(training_config.get("batch_size", 4)),
        "--learning_rate", str(training_config.get("learning_rate", 2e-5)),
        "--weight_decay", str(training_config.get("weight_decay", 0.01)),
        "--output_dir", str(model_dir),
        "--checkpoint_dir", str(model_dir / "checkpoints"),
    ]
    
    # Add memory optimization parameters if specified
    if training_config.get("gradient_accumulation_steps"):
        cmd.extend(["--gradient_accumulation_steps", str(training_config["gradient_accumulation_steps"])])
    if training_config.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")
    if training_config.get("fp16", False):
        cmd.append("--fp16")
    if training_config.get("bf16", False):
        cmd.append("--bf16")
    if training_config.get("dataloader_pin_memory", False):
        cmd.append("--dataloader_pin_memory")
    if training_config.get("dataloader_num_workers") is not None:
        cmd.extend(["--dataloader_num_workers", str(training_config["dataloader_num_workers"])])
    
    print(f"Training parameters for iteration {iteration}:")
    print(f"  base_model: {training_config.get('base_model', 'answerdotai/ModernBERT-base')}")
    print(f"  num_epochs: {training_config.get('num_epochs', 5)}")
    print(f"  batch_size: {training_config.get('batch_size', 4)}")
    print(f"  learning_rate: {training_config.get('learning_rate', 2e-5)}")
    print(f"  weight_decay: {training_config.get('weight_decay', 0.01)}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")
    
    return str(model_dir)


def evaluate_model(
    model_dir: str,
    eval_csv: str,
    metrics_json: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate model and save per-label metrics."""
    print(f"\n{'='*80}")
    print("Evaluating model")
    print(f"{'='*80}")
    
    training_config = config.get("training", {})
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "train" / "evaluate.py"),
        "--model_dir", model_dir,
        "--test_file", eval_csv,
        "--max_context_length", str(training_config.get("max_context_length", 1024)),
        "--output_metrics_json", metrics_json,
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with code {result.returncode}")
    
    # Load metrics
    with open(metrics_json, 'r') as f:
        return json.load(f)


def detect_gaps(
    metrics_json: str,
    train_csv: str,
    gap_report_json: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect gaps using gap_detector."""
    print(f"\n{'='*80}")
    print("Detecting gaps")
    print(f"{'='*80}")
    
    al_config = config.get("active_learning", {})
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "gap_detector.py"),
        "--metrics_json", metrics_json,
        "--train_csv", train_csv,
        "--output", gap_report_json,
        "--min_f1_threshold", str(al_config.get("min_f1_threshold", 0.82)),
        "--min_samples_threshold", str(al_config.get("min_samples_threshold", 50)),
        "--top_k", str(al_config.get("top_k_gaps", 10)),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Gap detection failed with code {result.returncode}")
    
    # Load gap report
    with open(gap_report_json, 'r') as f:
        return json.load(f)


def check_convergence(
    metrics_json: str,
    config: Dict[str, Any],
    previous_metrics: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """
    Check if we've converged.

    Returns:
        (converged, reason)
    """
    with open(metrics_json, 'r') as f:
        metrics = json.load(f)
    
    al_config = config.get("active_learning", {})
    min_f1_threshold = al_config.get("min_f1_threshold", 0.82)
    convergence_delta = al_config.get("convergence_delta", 0.01)
    
    # Check if all labels exceed threshold
    all_above_threshold = all(
        m.get("f1", 0) >= min_f1_threshold
        for m in metrics.values()
    )
    
    if all_above_threshold:
        return True, f"All labels exceed F1 threshold ({min_f1_threshold})"
    
    # Check improvement delta if we have previous metrics
    if previous_metrics:
        improvements = []
        for label, current_metrics in metrics.items():
            if label in previous_metrics:
                prev_f1 = previous_metrics[label].get("f1", 0)
                curr_f1 = current_metrics.get("f1", 0)
                improvements.append(curr_f1 - prev_f1)
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            if avg_improvement < convergence_delta:
                return True, f"Average improvement ({avg_improvement:.4f}) below convergence delta ({convergence_delta})"
    
    return False, ""


def update_summary(
    run_dir: Path,
    iteration: int,
    metrics: Dict[str, Any],
) -> None:
    """Update summary.json with metrics for this iteration."""
    summary_file = run_dir / "summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {"iterations": []}
    
    summary["iterations"].append({
        "iteration": iteration,
        "metrics": metrics,
    })
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def run_active_learning(
    config_path: str,
    run_id: Optional[str] = None,
    resume_from: Optional[int] = None,
) -> None:
    """
    Run the active learning pipeline.

    Parameters
    ----------
    config_path : str
        Path to active_learning_config.yaml
    run_id : str, optional
        Unique run ID (auto-generated if None)
    resume_from : int, optional
        Iteration to resume from (if None, starts from 0)
    """
    config = load_config(config_path)
    
    # Generate run ID if not provided
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    run_dir = Path(__file__).parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Active Learning Pipeline - Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*80}\n")
    
    # Load base generation config
    base_config_yaml = config.get("base_generation_config")
    if not base_config_yaml:
        raise ValueError("base_generation_config must be specified in config")
    
    base_config_yaml = str(Path(config_path).parent / base_config_yaml)
    
    al_config = config.get("active_learning", {})
    training_config = config.get("training", {})
    max_iterations = al_config.get("max_iterations", 6)
    min_iterations = al_config.get("min_iterations", 0)
    
    # Determine starting iteration
    start_iter = resume_from if resume_from is not None else 0
    
    previous_metrics = None
    frozen_eval_csv = None
    train_csv = None
    raw_csv_paths = []  # Track raw CSV paths (before multilabel labeling)
    
    # Restore raw CSV paths from all previous completed iterations
    for prev_iter in range(start_iter):
        prev_state = load_iteration_state(run_dir, prev_iter)
        if prev_state and prev_state.get("completed") and prev_state.get("raw_csv"):
            raw_csv_paths.append(prev_state.get("raw_csv"))
    
    for iteration in range(start_iter, max_iterations):
        iter_dir = run_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}\n")
        
        # Check if we can resume this iteration
        state = load_iteration_state(run_dir, iteration)
        if state and state.get("completed"):
            print(f"Iteration {iteration} already completed. Skipping.")
            previous_metrics = state.get("metrics")
            if iteration == 0:
                frozen_eval_csv = state.get("eval_csv")
            train_csv = state.get("train_csv")
            # Restore raw CSV path if available
            if state.get("raw_csv"):
                raw_csv_paths.append(state.get("raw_csv"))
            continue
        
        try:
            if iteration == 0:
                # Initial dataset generation
                raw_csv = iter_dir / "generated_raw.csv"
                generate_initial_dataset(
                    config=config,
                    output_csv=str(raw_csv),
                    base_config_yaml=base_config_yaml,
                )
                
                # Process data
                sys.path.insert(0, str(Path(__file__).parent))
                from data_manager import process_new_data
                
                existing_train_csv = None  # No existing data in iteration 0
                results = process_new_data(
                    new_raw_csv=str(raw_csv),
                    existing_train_csv=existing_train_csv,
                    frozen_eval_csv=None,  # Will be created
                    config_yaml=base_config_yaml,
                    base_model=training_config.get("base_model", "answerdotai/ModernBERT-base"),
                    max_context_length=training_config.get("max_context_length", 1024),
                    test_size=training_config.get("test_size", 0.15),
                    output_dir=str(iter_dir),
                    iteration=iteration,
                )
                
                frozen_eval_csv = results["eval_csv"]
                train_csv = results["train_csv"]
                raw_csv_paths.append(str(raw_csv))  # Track raw CSV path
                
            else:
                # Gap-filling data generation
                gap_report_json = run_dir / f"iter_{iteration-1}" / "gap_report.json"
                if not gap_report_json.exists():
                    raise FileNotFoundError(f"Gap report not found: {gap_report_json}")
                
                raw_csv = iter_dir / "generated_raw.csv"
                previous_metrics_json = run_dir / f"iter_{iteration-1}" / "metrics.json"
                
                # Merge all raw CSVs (before multilabel labeling) for gap analysis
                # gap_analysis_generator expects CSV with columns: message, primary topic, original topic, secondary topic
                merged_raw_csv = iter_dir / "merged_raw_for_gap_analysis.csv"
                if raw_csv_paths:
                    import pandas as pd
                    dfs = []
                    for raw_path in raw_csv_paths:
                        if Path(raw_path).exists():
                            dfs.append(pd.read_csv(raw_path))
                    if dfs:
                        merged_df = pd.concat(dfs, ignore_index=True)
                        merged_df.to_csv(merged_raw_csv, index=False)
                        print(f"Merged {len(dfs)} raw CSV files ({len(merged_df)} rows) for gap analysis")
                        existing_raw_csv = str(merged_raw_csv)
                    else:
                        raise RuntimeError(
                            f"No valid raw CSV files found in {raw_csv_paths}. "
                            "Cannot perform gap analysis without existing data."
                        )
                else:
                    raise RuntimeError(
                        "No previous raw CSV files found. "
                        "Gap analysis requires existing data from previous iterations."
                    )
                
                generate_gap_filling_data(
                    config=config,
                    existing_train_csv=existing_raw_csv,  # Use raw CSV format, not multilabel
                    gap_report_json=str(gap_report_json),
                    output_csv=str(raw_csv),
                    base_config_yaml=base_config_yaml,
                    metrics_json=str(previous_metrics_json),
                    messages_per_topic=al_config.get("gap_fill_messages_per_topic", 150),
                )
                
                # Process new data
                sys.path.insert(0, str(Path(__file__).parent))
                from data_manager import process_new_data
                
                results = process_new_data(
                    new_raw_csv=str(raw_csv),
                    existing_train_csv=train_csv,
                    frozen_eval_csv=frozen_eval_csv,
                    config_yaml=base_config_yaml,
                    base_model=training_config.get("base_model", "answerdotai/ModernBERT-base"),
                    max_context_length=training_config.get("max_context_length", 1024),
                    test_size=training_config.get("test_size", 0.15),
                    output_dir=str(iter_dir),
                    iteration=iteration,
                )
                
                train_csv = results["train_csv"]  # Accumulated training set
                raw_csv_paths.append(str(raw_csv))  # Track raw CSV path
            
            # Train model
            model_dir = train_model(
                train_csv=train_csv,
                output_dir=str(iter_dir),
                config=config,
                iteration=iteration,
            )
            
            # Evaluate model
            metrics_json = iter_dir / "metrics.json"
            metrics = evaluate_model(
                model_dir=model_dir,
                eval_csv=frozen_eval_csv or results["eval_csv"],
                metrics_json=str(metrics_json),
                config=config,
            )
            
            # Update summary
            update_summary(run_dir, iteration, metrics)
            
            # Detect gaps
            gap_report_json = iter_dir / "gap_report.json"
            gap_report = detect_gaps(
                metrics_json=str(metrics_json),
                train_csv=train_csv,
                gap_report_json=str(gap_report_json),
                config=config,
            )
            
            # Check convergence
            converged, reason = check_convergence(
                metrics_json=str(metrics_json),
                config=config,
                previous_metrics=previous_metrics,
            )
            
            # Save iteration state
            # raw_csv is defined in both branches (iteration 0 and else)
            save_iteration_state(run_dir, iteration, {
                "completed": True,
                "train_csv": train_csv,
                "eval_csv": frozen_eval_csv or results["eval_csv"],
                "model_dir": model_dir,
                "metrics": metrics,
                "gap_report": gap_report,
                "converged": converged,
                "convergence_reason": reason,
                "raw_csv": str(raw_csv),  # Save raw CSV path for gap analysis
            })
            
            previous_metrics = metrics
            
            if converged:
                if iteration + 1 >= min_iterations:
                    print(f"\n{'='*80}")
                    print(f"CONVERGED: {reason}")
                    print(f"{'='*80}\n")
                    break
                else:
                    print(f"\nConvergence detected but continuing to reach min_iterations ({min_iterations})")
                    print(f"Current iteration: {iteration + 1}, need at least: {min_iterations}")
            
            print(f"\nIteration {iteration} complete. {len(gap_report.get('gaps', []))} gaps detected.")
        
        except Exception as e:
            print(f"\nERROR in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error state
            save_iteration_state(run_dir, iteration, {
                "completed": False,
                "error": str(e),
            })
            
            raise
    
    print(f"\n{'='*80}")
    print("Active Learning Pipeline Complete")
    print(f"Run ID: {run_id}")
    print(f"Results: {run_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run active learning pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to active_learning_config.yaml",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--resume_from",
        type=int,
        default=None,
        help="Iteration to resume from",
    )
    
    args = parser.parse_args()
    
    run_active_learning(
        config_path=args.config,
        run_id=args.run_id,
        resume_from=args.resume_from,
    )
