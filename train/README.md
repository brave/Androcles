# Training

Fine-tune and evaluate the Androcles multi-label classifier (ModernBERT by default).

## Installation

```bash
pip install -r requirements.txt
```

From the repo root:

```bash
pip install -r train/requirements.txt
```

## Train

The training CSV needs a `message` column and one binary column per label:

```bash
python train.py \
  --train_file PATH/TO/TRAIN_FILE.csv \
  --base_model answerdotai/ModernBERT-base \
  --output_dir test_model \
  --eval_output eval.csv
```

Useful flags: `--max_context_length`, `--batch_size`, `--num_epochs`, `--learning_rate`, `--split_seed`.

## Quick inference

```bash
python check_outputs.py \
  --model_dir test_model \
  --test_string 'generate a python script to open a CSV file'
```

## Evaluate

```bash
python evaluate.py \
  --model_dir test_model \
  --test_file eval.csv \
  --output_metrics_json metrics.json
```
