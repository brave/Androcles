# Active Learning Pipeline

Orchestrates iterative dataset generation, training, evaluation, and gap-filling for Androcles.

## Loop

1. **Iteration 0**: Generate initial dataset from seed topics  
2. **Train**: Fine-tune the multilabel classifier  
3. **Evaluate**: Per-label F1 / precision / recall  
4. **Detect gaps**: Low F1 or insufficient samples  
5. **Generate**: Targeted data for underperforming topics  
6. **Repeat** until convergence or max iterations  

## Usage

```bash
# From repo root
pip install -r pipeline/requirements.txt

# Edit models.* in dataset_generator/configs/androcles_chat_dataset.yaml first
python pipeline/orchestrator.py \
  --config pipeline/configs/active_learning_config.example.yaml
```

Resume:

```bash
python pipeline/orchestrator.py \
  --config pipeline/configs/active_learning_config.example.yaml \
  --run_id <previous_run_id> \
  --resume_from <iteration_number>
```

## Configuration

[`configs/active_learning_config.example.yaml`](configs/active_learning_config.example.yaml):

- `base_generation_config`: path to Androcles generation YAML (21 topics)
- `training`: ModernBERT / hyperparameters
- `active_learning`: F1 thresholds, gap fill size, max iterations

## Output

Runs are written under `pipeline/runs/<run_id>/`:

```
runs/<run_id>/
├── iter_0/
│   ├── generated_raw.csv
│   ├── labeled.csv
│   ├── train.csv
│   ├── eval.csv          # frozen after iter 0
│   ├── model/
│   ├── metrics.json
│   ├── gap_report.json
│   └── state.json
└── summary.json
```

## Dependencies

- `dataset_generator/topic_search_generator.py`
- `dataset_generator/gap_analysis_generator.py`
- `dataset_generator/multilabel_labeller.py`
- `train/train.py`
- `train/evaluate.py`
