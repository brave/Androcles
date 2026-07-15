# Androcles

## About

**Dataset generation, training, and evaluation code for Brave's Androcles model**

Androcles is a multi-label text classifier that powers **Auto** model selection in [Leo](https://brave.com/leo/). Given a user prompt, it returns independent probabilities for each routing label so the backend server can choose an appropriate model or tool path.

Androcles 2 is a fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) model trained on a synthetic + search-augmented chat dataset with 21 distinct labels. It was not designed to be a perfectly accurate model with no mistakes, moreso to act as a general helper to guide the assistant

### Download

| Artifact | Hugging Face |
|---|---|
| Dataset | [`bravesoftware/androcles-2-chat-50k`](https://huggingface.co/datasets/bravesoftware/androcles-2-chat-50k) |
| Model | [`bravesoftware/Androcles-2`](https://huggingface.co/bravesoftware/Androcles-2) |


## Repository layout

```
androcles/
├── dataset_generator/   # Topic-search generation + multilabel labelling
├── train/               # Fine-tune + evaluate ModernBERT classifier
├── pipeline/            # Active learning loop (generate → train → gap-fill)
```

---

## Getting started

It is recommended to run training on a suitable GPU. Data generation needs an OpenAI-compatible LLM endpoint (or AWS Bedrock) and optionally an MCP search server.

### Train on the published dataset

```bash
pip install -r train/requirements.txt

huggingface-cli download bravesoftware/androcles-2-chat-50k --repo-type dataset --local-dir ./data

python train/train.py \
  --train_file ./data/train.csv \
  --base_model answerdotai/ModernBERT-base \
  --output_dir androcles2_model
```

CSV must have a `message` column plus one binary column per label.

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
  --test_file ./data/eval.csv \
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