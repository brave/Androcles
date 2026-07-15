# Dataset Generator

Topic-search message generation and multilabel labelling used to build Androcles 2 training data.

## Overview

1. **`topic_search_generator.py`** — expand seed topics, optionally search via MCP, generate realistic user messages  
2. **`multilabel_labeller.py`** — assign binary labels for all taxonomy topics  
3. **`gap_analysis_generator.py`** — generate more data for underperforming topics (used by the active learning pipeline)

## Canonical Androcles config

Use [`configs/androcles_chat_dataset.yaml`](configs/androcles_chat_dataset.yaml) for the 21-topic Androcles taxonomy and topic contexts.

Few-shot labelling examples: [`configs/labeller_config.yaml`](configs/labeller_config.yaml) (extends `androcles_chat_dataset.yaml`).

Sanitized templates:

- [`configs/config.example.yaml`](configs/config.example.yaml) — local vLLM + MCP placeholders  
- [`configs/config_bedrock.example.yaml`](configs/config_bedrock.example.yaml) — AWS Bedrock backend  

Edit `models.vllm.api_base` / `models.mcp.server_url` (or use Bedrock) before running.

## Quick start

```bash
# From repo root
pip install -r pipeline/requirements.txt

python dataset_generator/topic_search_generator.py \
  --config dataset_generator/configs/androcles_chat_dataset.yaml

python dataset_generator/multilabel_labeller.py \
  --config dataset_generator/configs/labeller_config.yaml \
  --input your_messages.csv \
  --output labeled.csv
```

CLI flags override YAML values. See `--help` on each script.

## Topic contexts

`topic_contexts` in the YAML guide generation so e.g. "Image Generation" means *ask the LLM to create an image*, not tutorials about image models.

## Search dependency

Search via MCP improves diversity. Without a search server, generation still works but enrichment quality may degrade. Point `models.mcp.server_url` at your MCP-compatible search endpoint. Internal tests showed that this also allows for stronger dataset diversity even with smaller models
