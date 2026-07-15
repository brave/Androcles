"""
Configuration for Topic Search Generator.

Defaults are public placeholders. Override via YAML configs under configs/
or CLI flags. See configs/config.example.yaml.
"""

# ---------------------------------------------------------------------------
# LLM (self-hosted vLLM, OpenAI-compatible API)
# ---------------------------------------------------------------------------
VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-14B"
VLLM_API_KEY = "EMPTY"  # vLLM ignores this but litellm/OpenAI clients require it
VLLM_MAX_TOKENS = 800
VLLM_TEMPERATURE = 0.7
# Labelling benefits from deterministic output; default 0.0 for multilabel_labeller
VLLM_TEMPERATURE_LABELLER = 0.0
# Qwen3 defaults to chain-of-thought "thinking" mode, which burns tokens before
# the actual answer and causes garbage output with a low max_tokens budget.
VLLM_ENABLE_THINKING = False

# ---------------------------------------------------------------------------
# AWS Bedrock (alternative to vLLM)
# ---------------------------------------------------------------------------
LLM_PROVIDER = "vllm"  # "vllm" or "bedrock"
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-east-1"

# ---------------------------------------------------------------------------
# MCP Server (web search, e.g. Brave Search MCP)
# ---------------------------------------------------------------------------
# Search improves message diversity. Point this at your own MCP search endpoint.
MCP_SERVER_URL = "http://localhost:8080/mcp"

# ---------------------------------------------------------------------------
# Topic Search Generator - Rate Limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_LLM_CALLS_PER_MINUTE: int = 60
RATE_LIMIT_SEARCH_CALLS_PER_MINUTE: int = 30
DEFAULT_MAX_CONCURRENT_TOPICS: int = 5
DEFAULT_MAX_CONCURRENT_PRIMARY_TOPICS: int = 2
DEFAULT_NUM_PRIMARY_TOPICS: int = 1

# ---------------------------------------------------------------------------
# Topic Contexts - Descriptions for topics to guide message generation
# ---------------------------------------------------------------------------
# Prefer setting these in YAML (e.g. configs/androcles_chat_dataset.yaml).
TOPIC_CONTEXTS: dict[str, str] = {
    "image generation": "when users want the LLM to generate or create images",
}
