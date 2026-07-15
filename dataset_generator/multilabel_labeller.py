#!/usr/bin/env python3
"""
Multilabel Data Labelling Agent

Converts single-label datasets to multilabel format by using an LLM to determine
which topics apply to each message. Outputs binary labels (1/0) for all topics
in the dataset.

Usage:
    # Using command-line arguments
    python multilabel_labeller.py --input dataset.csv --output multilabel.csv

    # Using YAML config file
    python multilabel_labeller.py --config config.yaml --input dataset.csv --output multilabel.csv

    # Using labeller-specific config (extends base, adds samples)
    python multilabel_labeller.py --config labeller_config.yaml --input dataset.csv --output multilabel.csv

    # Config file with CLI overrides
    python multilabel_labeller.py --config config.yaml --input dataset.csv --output multilabel.csv --max-concurrent 10
"""

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from tqdm import tqdm

import config
from llm_factory import create_llm
from rate_limiter import RateLimiter, retry_with_backoff

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunConfig:
    """Configuration for multilabel labelling run."""

    input_file: str
    output_file: str
    max_concurrent: int
    verbose: bool
    resume: bool
    dry_run: bool
    per_topic: bool
    samples: list[dict[str, Any]]  # [{"message": str, "labels": {topic: 0|1}}, ...]
    topic_contexts: dict[str, str]
    topics: (
        list[str] | None
    )  # If provided, use these topics instead of extracting from CSV
    llm_provider: str  # "vllm" or "bedrock"
    vllm_api_base: str
    vllm_model: str
    vllm_api_key: str
    vllm_temperature: float
    vllm_max_tokens: int
    vllm_enable_thinking: bool
    bedrock_model_id: str | None
    bedrock_region: str
    bedrock_max_pool_connections: int
    rate_limit_llm: int


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overrides into base. Overrides take precedence."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.
    Supports 'extends: other.yaml' to inherit from a base config.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails or config is invalid
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required for YAML config support. Install with: pip install pyyaml"
        )

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError("Config file must contain a YAML object/dictionary")

        # Resolve extends: load base config and deep merge
        if "extends" in config_data:
            extends_path = config_data.pop("extends")
            base_path = config_file.parent / extends_path
            if not base_path.exists():
                base_path = Path(extends_path)
            if not base_path.exists():
                raise FileNotFoundError(f"Extended config not found: {extends_path}")
            base_config = load_yaml_config(str(base_path))
            config_data = _deep_merge(base_config, config_data)

        return config_data
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config: {e}") from e


def apply_config_overrides(
    cfg: dict[str, Any], config_module: Any, args: argparse.Namespace | None = None
) -> dict[str, Any]:
    """
    Apply configuration from YAML and command-line args, with proper precedence.

    Precedence: command-line args > YAML config > config.py defaults

    Args:
        cfg: YAML config dictionary
        config_module: The config module (config.py)
        args: Optional argparse namespace with command-line arguments

    Returns:
        Dictionary with all configuration values resolved
    """
    result = {}

    # Start with config.py defaults
    result["llm_provider"] = getattr(config_module, "LLM_PROVIDER", "vllm")
    result["bedrock_model_id"] = getattr(config_module, "BEDROCK_MODEL_ID", None)
    result["bedrock_region"] = getattr(config_module, "BEDROCK_REGION", "us-east-1")
    result["bedrock_max_pool_connections"] = 50
    result["vllm_api_base"] = config_module.VLLM_API_BASE
    result["vllm_model"] = config_module.VLLM_MODEL
    result["vllm_api_key"] = config_module.VLLM_API_KEY
    result["vllm_max_tokens"] = getattr(config_module, "VLLM_MAX_TOKENS", 800)
    result["vllm_temperature"] = getattr(
        config_module, "VLLM_TEMPERATURE_LABELLER", 0.0
    )
    result["vllm_enable_thinking"] = getattr(
        config_module, "VLLM_ENABLE_THINKING", False
    )
    result["rate_limit_llm"] = config_module.RATE_LIMIT_LLM_CALLS_PER_MINUTE
    result["max_concurrent"] = config_module.DEFAULT_MAX_CONCURRENT_TOPICS
    result["topic_contexts"] = (
        config_module.TOPIC_CONTEXTS.copy()
        if hasattr(config_module, "TOPIC_CONTEXTS")
        else {}
    )
    result["topics"] = None  # Will be set from config if available
    result["input_file"] = None
    result["output_file"] = None
    result["verbose"] = False
    result["resume"] = False
    result["dry_run"] = False
    result["per_topic"] = True
    result["samples"] = []

    # Override with YAML config if present
    if "models" in cfg:
        models = cfg["models"]
        if "bedrock" in models:
            bedrock = models["bedrock"]
            result["llm_provider"] = "bedrock"
            result["bedrock_model_id"] = bedrock.get(
                "model_id", result["bedrock_model_id"]
            )
            result["bedrock_region"] = bedrock.get(
                "region_name", result["bedrock_region"]
            )
            result["vllm_max_tokens"] = bedrock.get(
                "max_tokens", result["vllm_max_tokens"]
            )
            result["vllm_temperature"] = bedrock.get(
                "temperature", result["vllm_temperature"]
            )
            result["bedrock_max_pool_connections"] = bedrock.get(
                "max_pool_connections", 50
            )
        elif "vllm" in models:
            vllm = models["vllm"]
            result["vllm_api_base"] = vllm.get("api_base", result["vllm_api_base"])
            result["vllm_model"] = vllm.get("model", result["vllm_model"])
            result["vllm_api_key"] = vllm.get("api_key", result["vllm_api_key"])
            result["vllm_max_tokens"] = vllm.get(
                "max_tokens", result["vllm_max_tokens"]
            )
            result["vllm_temperature"] = vllm.get(
                "temperature", result["vllm_temperature"]
            )
            result["vllm_enable_thinking"] = vllm.get(
                "enable_thinking", result["vllm_enable_thinking"]
            )

    if "topic_contexts" in cfg:
        result["topic_contexts"].update(cfg["topic_contexts"])

    # Extract topics from config if available (from "topics" or "labels" field)
    if "topics" in cfg and isinstance(cfg["topics"], list):
        result["topics"] = cfg["topics"]
    elif "labels" in cfg and isinstance(cfg["labels"], list):
        result["topics"] = cfg["labels"]

    if "rate_limiting" in cfg:
        rate_limit = cfg["rate_limiting"]
        result["rate_limit_llm"] = rate_limit.get(
            "llm_calls_per_minute", result["rate_limit_llm"]
        )

    if "concurrency" in cfg:
        concurrency = cfg["concurrency"]
        result["max_concurrent"] = concurrency.get(
            "max_concurrent", result["max_concurrent"]
        )

    if "logging" in cfg:
        result["verbose"] = cfg["logging"].get("verbose", result["verbose"])

    if "labelling" in cfg:
        labelling = cfg["labelling"]
        result["per_topic"] = labelling.get("per_topic", result["per_topic"])
        result["samples"] = labelling.get("samples", result["samples"])

    # Override with command-line args if present (highest precedence)
    if args:
        if hasattr(args, "input") and args.input:
            result["input_file"] = args.input
        if hasattr(args, "output") and args.output:
            result["output_file"] = args.output
        if hasattr(args, "max_concurrent") and args.max_concurrent is not None:
            result["max_concurrent"] = args.max_concurrent
        if hasattr(args, "verbose"):
            result["verbose"] = args.verbose
        if hasattr(args, "resume"):
            result["resume"] = args.resume
        if hasattr(args, "dry_run"):
            result["dry_run"] = args.dry_run
        if hasattr(args, "per_topic"):
            result["per_topic"] = args.per_topic

    return result


def read_input_csv(input_file: str) -> tuple[list[dict[str, str]], list[str]]:
    """
    Read input CSV and extract messages and unique topics.

    Args:
        input_file: Path to input CSV file

    Returns:
        Tuple of (list of message rows, sorted list of unique topics)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If CSV format is invalid
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    messages = []
    topics_set = set()

    try:
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required_columns = {"original topic", "message"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain columns: {required_columns}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                original_topic = row.get("original topic", "").strip()
                message = row.get("message", "").strip()

                if not message:
                    logger.warning("Skipping row with empty message")
                    continue

                if original_topic:
                    topics_set.add(original_topic)

                messages.append(
                    {
                        "original_topic": original_topic,
                        "message": message,
                    }
                )

        topics = sorted(list(topics_set))
        logger.info(
            "Read %d messages from %s, found %d unique topics",
            len(messages),
            input_file,
            len(topics),
        )
        return messages, topics

    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e


def read_existing_output(output_path: Path) -> set[str]:
    """
    Read already-processed messages from an existing output CSV (for resume).

    Args:
        output_path: Path to output CSV file

    Returns:
        Set of message strings already in the output (for deduplication)
    """
    if not output_path.exists():
        return set()

    existing_messages = set()
    try:
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = row.get("message", "").strip()
                if msg:
                    existing_messages.add(msg)
    except Exception as e:
        logger.warning("Could not read existing output for resume: %s", e)
        return set()

    return existing_messages


def _format_per_topic_samples(samples: list[dict[str, Any]], topic: str) -> str:
    """Format samples relevant to a single topic for few-shot in per-topic mode."""
    if not samples:
        return ""
    pos = []
    neg = []
    for s in samples:
        labels = s.get("labels", {})
        msg = s.get("message", "")
        val = labels.get(topic, 0)
        if val:
            pos.append(msg)
        else:
            neg.append(msg)
        if len(pos) >= 2 and len(neg) >= 2:
            break
    lines = []
    if pos:
        lines.append("Positive examples:")
        for m in pos[:2]:
            lines.append(f"  {json.dumps(m)} -> 1")
    if neg:
        lines.append("Negative examples:")
        for m in neg[:2]:
            lines.append(f"  {json.dumps(m)} -> 0")
    if lines:
        return "\n".join(lines) + "\n\n"
    return ""


async def label_single_topic(
    llm: BaseChatModel,
    message: str,
    topic: str,
    context: str,
    rate_limiter: RateLimiter | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> int:
    """
    Binary classification: does this message apply to this topic?

    Args:
        llm: Chat model instance (BaseChatModel)
        message: Message text
        topic: Topic name
        context: Optional context description for the topic
        rate_limiter: Optional rate limiter
        samples: Optional few-shot examples (filtered by topic)

    Returns:
        1 if topic applies, 0 otherwise
    """
    context_part = f" {context}" if context else ""
    message_escaped = json.dumps(message)
    samples_block = _format_per_topic_samples(samples or [], topic)

    prompt = f"""
    You are a data labelling assistant. Your job is to build datasets capable of training a multilabel classification model.
    These datasets should be very high quality and accurate.
    You will be given a topic and a context.
    You will be given a user message.
    You need to determine whether the user message should be labelled with this topic.
    Topic: {topic}
    Topic context: {context_part}

    {samples_block}
    
    User message: {message_escaped}

    Reply with 1 (the message should be labelled with this topic) or 0 (the message should not be labelled with this topic) only."""

    async def _generate():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        content = await retry_with_backoff(_generate)
        if not content:
            return 0
        # Parse: accept "1", "0", " 1 ", "1.", etc.
        stripped = content.strip().split()[0] if content.strip() else ""
        if stripped.startswith("1"):
            return 1
        return 0
    except Exception as e:
        logger.debug("Failed to label topic %r: %s", topic, e)
        return 0


def _format_samples_for_prompt(samples: list[dict[str, Any]], topics: list[str]) -> str:
    """Format few-shot samples for inclusion in the labelling prompt."""
    if not samples:
        return ""
    lines = ["Examples:"]
    for s in samples:
        msg = s.get("message", "")
        labels = s.get("labels", {})
        # Build full label dict (all topics, 0 or 1)
        full = {t: labels.get(t, 0) for t in topics}
        lines.append(f"  Message: {json.dumps(msg)}")
        lines.append(f"  Labels: {json.dumps(full)}")
    return "\n".join(lines) + "\n\n"


def _strip_markdown_json(text: str) -> str:
    """
    Extract JSON from markdown code blocks if present.

    Args:
        text: Text that may contain JSON in markdown code blocks

    Returns:
        Text with markdown code blocks stripped
    """
    if "```json" in text:
        json_start = text.find("```json") + 7
        json_end = text.find("```", json_start)
        return text[json_start:json_end].strip()
    elif "```" in text:
        json_start = text.find("```") + 3
        json_end = text.find("```", json_start)
        return text[json_start:json_end].strip()
    return text


async def label_message(
    llm: BaseChatModel,
    message: str,
    topics: list[str],
    topic_contexts: dict[str, str],
    rate_limiter: RateLimiter | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """
    Use LLM to determine which topics apply to a message (multilabel classification).

    Args:
        llm: Chat model instance (BaseChatModel)
        message: Message text to label
        topics: List of all possible topics
        topic_contexts: Dictionary mapping topic names to context descriptions
        rate_limiter: Optional rate limiter for LLM calls
        samples: Optional few-shot examples [{"message": str, "labels": {topic: 0|1}}]

    Returns:
        Dictionary mapping topic names to binary labels (1 or 0)

    Raises:
        RuntimeError: If LLM call fails or returns invalid response
    """
    # Build topic list with contexts
    topics_list = []
    for topic in topics:
        context = topic_contexts.get(topic, "")
        if context:
            topics_list.append(f"- {topic}: {context}")
        else:
            topics_list.append(f"- {topic}")

    topics_text = "\n".join(topics_list)
    message_escaped = json.dumps(message)
    samples_block = _format_samples_for_prompt(samples or [], topics)

    prompt = f"""You are a data labelling assistant. Label which topics apply to a user message.

CRITICAL: A topic applies ONLY if the user is explicitly asking the LLM to perform that type of task or provide that type of response. Do NOT label a topic just because the message mentions or discusses that subject—the user must be requesting that specific kind of output from the LLM. When uncertain, label 0.

Topics and their meanings:
{topics_text}
{samples_block}User message: {message_escaped}

For each topic, output 1 only if the user is clearly asking for that type of LLM response. Output 0 otherwise. Most messages will have 0-3 topics; avoid overlabeling.

Return ONLY a JSON object with all topics as keys and 0 or 1 as values. No explanations."""

    async def _generate():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        content = await retry_with_backoff(_generate)

        if not content:
            raise RuntimeError("LLM returned empty response for message labelling")

        # Extract JSON from response
        content = _strip_markdown_json(content)
        labels = json.loads(content)

        if not isinstance(labels, dict):
            raise ValueError("LLM response is not a JSON object")

        # Build normalized lookup map: lowercase key -> (original_key, value)
        # Enables case-insensitive matching when LLM returns "coding" instead of "Coding"
        labels_normalized = {k.strip().lower(): (k, v) for k, v in labels.items()}

        # Validate labels: exact match first, then normalized (case-insensitive)
        result = {}
        for topic in topics:
            value = None
            if topic in labels:
                value = labels[topic]
            else:
                key_normalized = topic.strip().lower()
                if key_normalized in labels_normalized:
                    _, value = labels_normalized[key_normalized]

            if value is not None:
                # Convert to int and ensure it's 0 or 1
                if isinstance(value, bool):
                    result[topic] = 1 if value else 0
                elif isinstance(value, (int, float)):
                    result[topic] = 1 if int(value) != 0 else 0
                else:
                    result[topic] = 0
            else:
                # Topic not in response, default to 0
                result[topic] = 0

        return result

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON from LLM response: %s", e)
        logger.debug("LLM response was: %s", content)
        # Fallback: return all zeros
        return {topic: 0 for topic in topics}
    except Exception as e:
        logger.error("Failed to label message: %s", e)
        # Fallback: return all zeros
        return {topic: 0 for topic in topics}


async def process_message(
    llm: BaseChatModel,
    message_data: dict[str, str],
    topics: list[str],
    topic_contexts: dict[str, str],
    rate_limiter: RateLimiter | None = None,
    per_topic: bool = False,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Process a single message and return labels.

    Args:
        llm: Chat model instance (BaseChatModel)
        message_data: Dictionary with 'message' and 'original_topic' keys
        topics: List of all possible topics
        topic_contexts: Dictionary mapping topic names to context descriptions
        rate_limiter: Optional rate limiter for LLM calls
        per_topic: If True, send one request per topic (more accurate for large taxonomies)

    Returns:
        Dictionary with 'message' and labels for each topic
    """
    message = message_data["message"]
    original_topic = message_data.get("original_topic", "")

    try:
        if per_topic:
            # One request per topic, run concurrently
            async def label_one(topic: str) -> tuple[str, int]:
                ctx = topic_contexts.get(topic, "")
                val = await label_single_topic(
                    llm,
                    message,
                    topic,
                    ctx,
                    rate_limiter,
                    samples=samples,
                )
                return (topic, val)

            results = await asyncio.gather(
                *[label_one(t) for t in topics],
                return_exceptions=True,
            )
            labels = {}
            for r in results:
                if isinstance(r, Exception):
                    logger.debug("Per-topic label failed: %s", r)
                    continue
                topic, val = r
                labels[topic] = val
            # Fill missing with 0
            for t in topics:
                if t not in labels:
                    labels[t] = 0
        else:
            labels = await label_message(
                llm,
                message,
                topics,
                topic_contexts,
                rate_limiter,
                samples=samples,
            )

        result = {"message": message}
        result.update(labels)

        # Always preserve the original topic label - don't let LLM override it
        # If a message was generated for a specific topic, that topic should always be labeled
        if original_topic and original_topic in topics:
            if labels.get(original_topic, 0) == 0:
                logger.debug(
                    "LLM did not label original topic '%s', forcing it to 1 for message: %s",
                    original_topic,
                    message[:50],
                )
            result[original_topic] = 1  # Always set original topic to 1

        return result

    except Exception as e:
        logger.error("Failed to process message: %s", e)
        # Return message with all zeros on error
        result = {"message": message}
        result.update({topic: 0 for topic in topics})
        return result


async def main_async(cfg: RunConfig) -> None:
    """
    Main async orchestration function.

    Args:
        cfg: RunConfig dataclass with all configuration parameters
    """
    # Setup logging
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting multilabel data labelling agent")
    logger.info("Input file: %s", cfg.input_file)
    logger.info("Output file: %s", cfg.output_file)

    csv_file_handle = None

    # Read input CSV
    logger.info("Reading input CSV...")
    messages, csv_topics = read_input_csv(cfg.input_file)

    # Determine which topics to use for labeling
    if cfg.topics:
        # Use topics from config file
        topics = sorted(cfg.topics)
        logger.info("Using %d topics from config: %s", len(topics), ", ".join(topics))
        # Warn if CSV topics don't match config topics
        csv_topics_set = set(csv_topics)
        config_topics_set = set(topics)
        missing_in_csv = config_topics_set - csv_topics_set
        if missing_in_csv:
            logger.info(
                "Note: %d config topics not found in CSV 'original topic' column: %s",
                len(missing_in_csv),
                ", ".join(sorted(missing_in_csv)),
            )
    else:
        # Fall back to topics from CSV (original behavior)
        topics = csv_topics
        if not topics:
            logger.error(
                "No topics found in input file and no topics specified in config"
            )
            sys.exit(1)
        logger.info(
            "Found %d unique topics from CSV: %s", len(topics), ", ".join(topics[:10])
        )

    if cfg.dry_run:
        logger.info("Dry-run: skipping LLM calls and output")
        logger.info(
            "Would process %d messages with %d topics",
            len(messages),
            len(topics),
        )
        logger.info(
            "Model: %s (provider=%s), temperature=%.2f",
            cfg.bedrock_model_id if cfg.llm_provider == "bedrock" else cfg.vllm_model,
            cfg.llm_provider,
            cfg.vllm_temperature,
        )
        if cfg.per_topic:
            logger.info(
                "Per-topic mode: %d LLM requests per message",
                len(topics),
            )
        if cfg.samples:
            logger.info("Loaded %d few-shot samples", len(cfg.samples))
        else:
            logger.warning(
                "No samples in config. Use labeller_config.yaml or add labelling.samples for better accuracy."
            )
        return

    if cfg.samples:
        logger.info("Loaded %d few-shot samples", len(cfg.samples))
    elif cfg.per_topic:
        logger.warning(
            "No samples in config. Use labeller_config.yaml or add labelling.samples for better accuracy."
        )

    # Initialize rate limiter
    llm_rate_limiter = RateLimiter(cfg.rate_limit_llm, 60.0)

    # Initialize LLM
    if cfg.llm_provider == "bedrock":
        if not cfg.bedrock_model_id:
            raise ValueError(
                "bedrock_model_id is required when using Bedrock. "
                "Set models.bedrock.model_id in config."
            )
        llm_kwargs = {
            "model_id": cfg.bedrock_model_id,
            "region_name": cfg.bedrock_region,
            "temperature": cfg.vllm_temperature,
            "max_tokens": cfg.vllm_max_tokens,
            "max_pool_connections": cfg.bedrock_max_pool_connections,
        }
    else:
        llm_kwargs = {
            "base_url": cfg.vllm_api_base,
            "temperature": cfg.vllm_temperature,
            "model": cfg.vllm_model,
            "api_key": cfg.vllm_api_key,
            "max_tokens": cfg.vllm_max_tokens,
        }
        if cfg.vllm_enable_thinking:
            llm_kwargs["enable_thinking"] = True
    llm = create_llm(cfg.llm_provider, **llm_kwargs)

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    # Resume: skip messages already in output
    output_path = Path(cfg.output_file)
    if cfg.resume:
        existing_messages = read_existing_output(output_path)
        if existing_messages:
            messages = [m for m in messages if m["message"] not in existing_messages]
            logger.info(
                "Resume: skipping %d already-processed messages, %d remaining",
                len(existing_messages),
                len(messages),
            )
        else:
            logger.info("Resume: no existing output found, processing all messages")

    # Initialize output CSV file
    append_mode = cfg.resume and output_path.exists()
    if append_mode:
        csv_file_handle = open(output_path, "a", encoding="utf-8", newline="")
    else:
        csv_file_handle = open(output_path, "w", encoding="utf-8", newline="")

    csv_writer = csv.writer(csv_file_handle)

    # Write header only when not resuming/appending
    if not append_mode:
        header = ["message"] + topics
        csv_writer.writerow(header)
        csv_file_handle.flush()

    try:
        # Initialize progress bar
        pbar = tqdm(total=len(messages), desc="Labelling messages", unit="message")

        try:

            async def process_with_semaphore(message_data: dict[str, str]):
                """Process a message with semaphore control."""
                async with semaphore:
                    result = await process_message(
                        llm,
                        message_data,
                        topics,
                        cfg.topic_contexts,
                        llm_rate_limiter,
                        per_topic=cfg.per_topic,
                        samples=cfg.samples,
                    )

                    # Write row to CSV
                    row = [result["message"]] + [
                        result.get(topic, 0) for topic in topics
                    ]
                    csv_writer.writerow(row)
                    csv_file_handle.flush()

                    # Update progress bar
                    pbar.update(1)

                    return result

            # Process all messages concurrently with progress bar
            logger.info("Processing %d messages...", len(messages))
            tasks = [process_with_semaphore(msg) for msg in messages]
            results = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            # Count successes and failures
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count

            if failure_count > 0:
                logger.warning(
                    "Completed with %d successes and %d failures",
                    success_count,
                    failure_count,
                )
            else:
                logger.info("Successfully processed %d messages", success_count)

            csv_file_handle.close()
            logger.info("Results written to %s", cfg.output_file)
        finally:
            # Always close progress bar
            pbar.close()

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        if csv_file_handle is not None:
            csv_file_handle.close()
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=("Convert single-label dataset to multilabel format using LLM."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file path (must contain 'original topic' and 'message' columns)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path (will contain 'message' column and one column per topic)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to YAML configuration file. This file can contain model settings, "
            "topic contexts, rate limiting, and concurrency settings. "
            "Command-line arguments override values in the config file."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help=f"Maximum concurrent message processing (default: {config.DEFAULT_MAX_CONCURRENT_TOPICS})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-processed messages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and input without calling LLM or writing output",
    )
    parser.add_argument(
        "--no-per-topic",
        action="store_false",
        dest="per_topic",
        help="Use single request per message (default is one request per topic)",
    )

    args = parser.parse_args()

    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        try:
            yaml_config = load_yaml_config(args.config)
            logger.info("Loaded configuration from %s", args.config)
        except Exception as e:
            logger.error("Failed to load config file: %s", e)
            sys.exit(1)

    # Apply config with proper precedence: CLI args > YAML > config.py
    cfg_dict = apply_config_overrides(yaml_config, config, args)

    # Validate required fields
    if not cfg_dict["input_file"]:
        logger.error("Input file is required. Use --input or provide in config file.")
        sys.exit(1)
    if not cfg_dict["output_file"]:
        logger.error("Output file is required. Use --output or provide in config file.")
        sys.exit(1)

    # Setup basic logging
    level = logging.DEBUG if cfg_dict["verbose"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("langchain_aws").setLevel(logging.WARNING)

    # Create RunConfig
    run_cfg = RunConfig(
        input_file=cfg_dict["input_file"],
        output_file=cfg_dict["output_file"],
        max_concurrent=cfg_dict["max_concurrent"],
        verbose=cfg_dict["verbose"],
        resume=cfg_dict.get("resume", False),
        dry_run=cfg_dict.get("dry_run", False),
        per_topic=cfg_dict.get("per_topic", False),
        samples=cfg_dict.get("samples", []),
        topic_contexts=cfg_dict["topic_contexts"],
        topics=cfg_dict.get("topics"),
        llm_provider=cfg_dict["llm_provider"],
        vllm_api_base=cfg_dict["vllm_api_base"],
        vllm_model=cfg_dict["vllm_model"],
        vllm_api_key=cfg_dict["vllm_api_key"],
        vllm_temperature=cfg_dict["vllm_temperature"],
        vllm_max_tokens=cfg_dict["vllm_max_tokens"],
        vllm_enable_thinking=cfg_dict["vllm_enable_thinking"],
        bedrock_model_id=cfg_dict.get("bedrock_model_id"),
        bedrock_region=cfg_dict.get("bedrock_region", "us-east-1"),
        bedrock_max_pool_connections=cfg_dict.get("bedrock_max_pool_connections", 50),
        rate_limit_llm=cfg_dict["rate_limit_llm"],
    )

    asyncio.run(main_async(run_cfg))


if __name__ == "__main__":
    main()
