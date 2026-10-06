# Androcles

## About

**Dataset generation, training, and evaluation code for Brave's Androcles model**

Androcles is a multi-label text classifier that powers **Auto** model selection in [Leo](https://brave.com/leo/). Given a user prompt, it returns independent probabilities for each routing label so the backend server can choose an appropriate model or tool path.

Androcles 2 is a fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) model trained on a multilabel chat dataset (synthetic search-grounded prompts plus additional labelled open-source chat data). The full labelled training set is not published; the open **[Diverse LLM Prompts 34k](https://huggingface.co/datasets/bravesoftware/diverse-llm-prompts-34k)** release is the synthetic portion of that pipeline.

### Download

| Artifact | Hugging Face |
|---|---|
| Synthetic prompts (unlabelled) | [`bravesoftware/diverse-llm-prompts-34k`](https://huggingface.co/datasets/bravesoftware/diverse-llm-prompts-34k) |
| Model | [`bravesoftware/Androcles-2`](https://huggingface.co/bravesoftware/Androcles-2) |

The 34k dataset has columns `original topic`, `primary topic`, `secondary topic`, and `message`. Run the multilabel labeller (below) to produce a CSV with `message` plus one binary column per routing label, then train.


## Repository layout

```
androcles/
├── dataset_generator/   # Topic-search generation + multilabel labelling
├── train/               # Fine-tune + evaluate ModernBERT classifier
├── pipeline/            # Active learning loop (generate → train → gap-fill)
```

---

## Getting started

Pinned dependency lockfiles: `train/requirements.txt` and `pipeline/requirements.txt` (from `*.in` via `pip-compile --strip-extras --allow-unsafe`). Dev/test: `requirements-dev.txt`.

It is recommended to run training on a suitable GPU. Data generation needs an OpenAI-compatible LLM endpoint (or AWS Bedrock) and optionally an MCP search server.

### Train from the open synthetic dataset

```bash
pip install -r train/requirements.txt
pip install -r pipeline/requirements.txt

hf download bravesoftware/diverse-llm-prompts-34k --repo-type dataset --local-dir ./data/diverse-llm-prompts-34k

python -c "
from datasets import load_dataset
load_dataset('bravesoftware/diverse-llm-prompts-34k', split='train').to_csv(
    'data/diverse-llm-prompts-34k.csv', index=False)
"

python dataset_generator/multilabel_labeller.py \
  --config dataset_generator/configs/labeller_config.yaml \
  --input data/diverse-llm-prompts-34k.csv \
  --output data/diverse-llm-prompts-34k-labelled.csv

python train/train.py \
  --train_file data/diverse-llm-prompts-34k-labelled.csv \
  --base_model answerdotai/ModernBERT-base \
  --output_dir androcles2_model \
  --eval_output data/eval.csv
```

Labelling requires a configured LLM (see [`dataset_generator/README.md`](dataset_generator/README.md)). If the dataset repo is private, run `hf auth login` first.

To add more prompts (e.g. from public chat corpora such as [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)), prepare CSV rows the labeller accepts (`original topic` and `message`), label them, and merge with the synthetic CSV before training.

### Inference

```bash
python train/check_outputs.py \
  --model_dir bravesoftware/Androcles-2 \
  --test_string "Generate an image of a cat" "Write a Python script to parse CSV"
```

### Evaluate

```bash
python train/evaluate.py \
  --model_dir bravesoftware/Androcles-2 \
  --test_file data/eval.csv \
  --output_metrics_json metrics.json
```

### Regenerate or extend data

```bash
pip install -r pipeline/requirements.txt

# Edit models.* in dataset_generator/configs/androcles_chat_dataset.yaml first
python dataset_generator/topic_search_generator.py \
  --config dataset_generator/configs/androcles_chat_dataset.yaml

python dataset_generator/multilabel_labeller.py \
  --config dataset_generator/configs/labeller_config.yaml \
  --input generated.csv \
  --output labeled.csv
```

### Full active learning pipeline

```bash
python pipeline/orchestrator.py \
  --config pipeline/configs/active_learning_config.example.yaml
```


## License

This code is made available under the Mozilla Public License 2.0. Please see the [`LICENSE`](LICENSE) for more information.

Model and dataset cards use Apache-2.0 metadata on Hugging Face; the training/data code in this repository is MPL-2.0.


## Disclaimer

Please note that AI was used to pull this repo out from our ML monorepo - while we've double checked to ensure nothing was missing, if anything looks out of place please let us know and we can update
