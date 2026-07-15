#!/usr/bin/env python3
"""
Gap Analysis Dataset Generator

Extends the topic search generator to analyze existing messages and identify gaps,
then generates new messages to fill those gaps.

Usage:
    # Analyze gaps in existing CSV and generate new messages
    python gap_analysis_generator.py --topic "food" --existing-messages existing.csv

    # Using YAML config file
    python gap_analysis_generator.py --config config.yaml --existing-messages existing.csv

    # Config file with CLI overrides
    python gap_analysis_generator.py --config config.yaml --existing-messages existing.csv --num-topics 30
"""

import argparse
import asyncio
import csv
import dataclasses
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

import config
from llm_factory import create_llm
from rate_limiter import RateLimiter, retry_with_backoff

# Import reusable functions from topic_search_generator
from topic_search_generator import (
    RunConfig,
    _extract_agent_content,
    _parse_topic_list,
    generate_topics,
    generate_search_queries,
    search_topic,
    generate_messages_per_topic,
    create_message_generator_agent,
    load_yaml_config,
    apply_config_overrides,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GapAnalysisConfig(RunConfig):
    """Extended configuration for gap analysis generator."""

    existing_messages_file: str | None = None
    min_messages_per_topic: int = 5  # Threshold for considering a topic well-covered
    metrics_json: str | None = (
        None  # Path to per-label metrics JSON for F1-based weighting
    )


def load_existing_messages(csv_path: str) -> dict[str, Any]:
    """
    Load existing messages from CSV and extract topic statistics.

    Args:
        csv_path: Path to CSV file with columns: original topic, primary topic, secondary topic, message

    Returns:
        Dictionary with:
        - messages: List of all message rows
        - primary_topics: Dict mapping primary topic -> message count
        - secondary_topics: Dict mapping (primary, secondary) -> message count
        - original_topics: Set of original topics
        - total_messages: Total number of messages
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    messages = []
    primary_topics = defaultdict(int)
    secondary_topics = defaultdict(int)
    original_topics = set()

    try:
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Validate columns
            expected_columns = {
                "original topic",
                "primary topic",
                "secondary topic",
                "message",
            }
            if not expected_columns.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV must contain columns: {expected_columns}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                original_topic = row.get("original topic", "").strip()
                primary_topic = row.get("primary topic", "").strip()
                secondary_topic = row.get("secondary topic", "").strip()
                message = row.get("message", "").strip()

                if not message:  # Skip empty messages
                    continue

                messages.append(
                    {
                        "original_topic": original_topic,
                        "primary_topic": primary_topic,
                        "secondary_topic": secondary_topic,
                        "message": message,
                    }
                )

                if original_topic:
                    original_topics.add(original_topic)
                if primary_topic:
                    primary_topics[primary_topic] += 1
                if primary_topic and secondary_topic:
                    secondary_topics[(primary_topic, secondary_topic)] += 1

        logger.info(
            "Loaded %d messages from %s: %d primary topics, %d secondary topics",
            len(messages),
            csv_path,
            len(primary_topics),
            len(secondary_topics),
        )

        return {
            "messages": messages,
            "primary_topics": dict(primary_topics),
            "secondary_topics": dict(secondary_topics),
            "original_topics": original_topics,
            "total_messages": len(messages),
        }

    except Exception as e:
        logger.error("Failed to load CSV file %s: %s", csv_path, e)
        raise


def create_gap_analysis_agent(llm):
    """
    Create a LangChain agent for analyzing dataset gaps.

    Args:
        llm: LangChain LLM instance

    Returns:
        Agent graph instance (from create_agent)
    """
    logger.info("Creating gap analysis agent...")

    # Create agent without tools - just LLM for gap analysis
    agent = create_agent(
        model=llm,
        system_prompt=(
            "You are a helpful assistant that analyzes datasets to identify "
            "coverage gaps. You examine existing topics and messages to find "
            "areas that are underrepresented or missing, then suggest topics "
            "that would fill those gaps."
        ),
        debug=logger.isEnabledFor(logging.DEBUG),
    )

    return agent


async def identify_gaps(
    gap_agent: Any,
    topic: str,
    topic_context: str | None,
    existing_data: dict[str, Any],
    num_gaps: int = 10,
    rate_limiter: RateLimiter | None = None,
    min_messages_per_topic: int = 5,
    metrics_json: str | None = None,
) -> list[str]:
    """
    Identify gaps in existing dataset coverage.

    Args:
        gap_agent: Agent executor for gap analysis
        topic: Topic to analyze gaps for
        topic_context: Optional context/description about what the topic means
        existing_data: Dictionary from load_existing_messages()
        num_gaps: Number of gap topics to identify
        rate_limiter: Optional rate limiter for LLM calls
        min_messages_per_topic: Minimum messages per topic to consider it well-covered

    Returns:
        List of topic strings representing gaps (to be used as primary topics)
    """
    logger.info("Identifying gaps for topic: %r", topic)

    # Load model metrics if provided
    model_metrics = {}
    if metrics_json and Path(metrics_json).exists():
        try:
            with open(metrics_json, "r") as f:
                model_metrics = json.load(f)
            logger.info("Loaded model metrics for %d labels", len(model_metrics))
        except Exception as e:
            logger.warning("Failed to load metrics JSON %s: %s", metrics_json, e)

    # Prepare summary of existing coverage
    primary_topics = existing_data["primary_topics"]
    secondary_topics = existing_data["secondary_topics"]
    total_messages = existing_data["total_messages"]

    # Find underrepresented primary topics
    underrepresented_primary = [
        pt for pt, count in primary_topics.items() if count < min_messages_per_topic
    ]

    # Find underrepresented secondary topics
    underrepresented_secondary = [
        f"{pt} - {st}"
        for (pt, st), count in secondary_topics.items()
        if count < min_messages_per_topic
    ]

    # Build model performance summary if metrics available
    model_performance_section = ""
    if model_metrics:
        # Find topics with low F1 scores
        low_f1_topics = [
            label
            for label, metrics in model_metrics.items()
            if metrics.get("f1", 1.0) < 0.82  # Default threshold
        ]
        if low_f1_topics:
            model_performance_section = f"""
MODEL PERFORMANCE ANALYSIS:
The following topics have low F1 scores (model gaps, not just data gaps):
{chr(10).join(f"- {label}: F1={model_metrics[label].get('f1', 0):.3f}, "
               f"Precision={model_metrics[label].get('precision', 0):.3f}, "
               f"Recall={model_metrics[label].get('recall', 0):.3f}"
               for label in low_f1_topics[:10])}

PRIORITY: Focus on generating data for topics with low F1 scores, as these represent
model confusion/underperformance rather than just insufficient training examples.
"""

    # Build coverage summary
    primary_summary = "\n".join(
        f"- {pt}: {count} messages"
        for pt, count in sorted(primary_topics.items(), key=lambda x: x[1])
    )

    secondary_summary = "\n".join(
        f"- {pt} > {st}: {count} messages"
        for (pt, st), count in sorted(secondary_topics.items(), key=lambda x: x[1])[
            :20
        ]  # Limit to top 20
    )

    context_section = ""
    if topic_context:
        context_section = f"""
IMPORTANT CONTEXT: "{topic}" means: {topic_context}

All identified gap topics MUST align with this context. For example, if the topic \
is "image generation" and the context says "when users want the LLM to generate \
images", then identify gaps related to users asking the LLM to create/generate \
images, NOT gaps about image classification, image editing, or other unrelated \
image tasks.
"""

    prompt = f"""Analyze the existing dataset for topic "{topic}" and identify coverage gaps.

{context_section}
{model_performance_section}
EXISTING COVERAGE SUMMARY:
- Total messages: {total_messages}
- Primary topics covered: {len(primary_topics)}
- Secondary topics covered: {len(secondary_topics)}

PRIMARY TOPICS COVERAGE:
{primary_summary if primary_summary else "(none)"}

SECONDARY TOPICS COVERAGE (sample):
{secondary_summary if secondary_summary else "(none)"}

UNDERREPRESENTED AREAS:
- Primary topics with <{min_messages_per_topic} messages: {len(underrepresented_primary)}
- Secondary topics with <{min_messages_per_topic} messages: {len(underrepresented_secondary)}

TASK:
Identify {num_gaps} topic gaps that would expand coverage for "{topic}". These gaps should:
1. Be areas NOT well-covered in the existing dataset
2. Be directly related to "{topic}"
3. Represent broad categories (primary topics) that could have many subtopics
4. Be distinct from existing primary topics (or be underrepresented existing topics)
{f"5. MUST align with the context provided above" if topic_context else ""}

Return ONLY a list of gap topics, one per line. No numbering, no explanations, no extra text.
You can use commas to separate multiple topics on the same line, or put one topic per line.

Example format:
gap topic 1
gap topic 2
gap topic 3

Now identify {num_gaps} gap topics for "{topic}":
"""

    async def _analyze():
        result = await gap_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        result = await retry_with_backoff(_analyze)
        response_text = _extract_agent_content(result)

        # Parse the response
        gap_topics = _parse_topic_list(response_text, num_gaps)

        if len(gap_topics) < num_gaps:
            logger.warning(
                "Identified only %d gap topics (requested %d)",
                len(gap_topics),
                num_gaps,
            )

        logger.info("Identified gap topics: %s", gap_topics)
        return gap_topics

    except Exception as e:
        logger.error("Failed to identify gaps: %s", e)
        return []


async def main_async(cfg: GapAnalysisConfig, write_csv_header: bool = True) -> None:
    """
    Main async orchestration function for gap analysis and generation.

    Args:
        cfg: GapAnalysisConfig dataclass with all configuration parameters
        write_csv_header: Whether to write CSV header (for multi-topic runs)
    """
    # Setup logging
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting gap analysis dataset generator")
    logger.info(
        "Topic: %r, Existing messages: %s",
        cfg.initial_topic,
        cfg.existing_messages_file,
    )

    # Load existing messages
    if not cfg.existing_messages_file:
        logger.error("--existing-messages is required for gap analysis")
        sys.exit(1)

    try:
        existing_data = load_existing_messages(cfg.existing_messages_file)
    except Exception as e:
        logger.error("Failed to load existing messages: %s", e)
        sys.exit(1)

    # Initialize rate limiters
    llm_rate_limiter = RateLimiter(cfg.rate_limit_llm, 60.0)
    search_rate_limiter = RateLimiter(cfg.rate_limit_search, 60.0)

    # Initialize LLM for agents
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

    # Setup MCP client
    logger.info("Setting up MCP client...")
    mcp_client = MultiServerMCPClient(
        {
            "brave_search": {
                "transport": "http",
                "url": cfg.mcp_server_url,
            }
        }
    )
    tools = await mcp_client.get_tools()
    logger.info("Retrieved %d tools from MCP client", len(tools))

    # Create agents
    gap_agent = create_gap_analysis_agent(llm)
    message_agent = create_message_generator_agent(llm)

    # Semaphores for concurrency control
    secondary_semaphore = asyncio.Semaphore(cfg.max_concurrent)
    primary_semaphore = asyncio.Semaphore(cfg.max_concurrent_primary)

    # Message counter for max_messages tracking
    message_counter: list[int] = [0]
    message_counter_lock = asyncio.Lock()

    # Initialize CSV file (append mode for multiple topics)
    csv_path = Path(cfg.output_file)
    write_header = write_csv_header and not csv_path.exists()
    csv_file_handle = open(cfg.output_file, "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file_handle)
    if write_header:
        csv_writer.writerow(
            ["original topic", "primary topic", "secondary topic", "message"]
        )
        csv_file_handle.flush()

    try:
        # Look up topic context
        context = None
        if cfg.topic_contexts:
            context = cfg.topic_contexts.get(cfg.initial_topic)

        # Identify gaps
        logger.info("Analyzing gaps in existing dataset...")
        gap_topics = await identify_gaps(
            gap_agent,
            cfg.initial_topic,
            context,
            existing_data,
            num_gaps=cfg.num_primary_topics,
            rate_limiter=llm_rate_limiter,
            min_messages_per_topic=cfg.min_messages_per_topic,
            metrics_json=cfg.metrics_json,
        )

        if not gap_topics:
            logger.warning("No gaps identified. Exiting.")
            csv_file_handle.close()
            return

        logger.info("Identified %d gap topics to process", len(gap_topics))

        async def process_secondary_topic(
            original_topic: str, primary_topic: str, secondary_topic: str
        ) -> None:
            """Process a single secondary topic."""
            async with secondary_semaphore:
                try:
                    # Look up topic context (check secondary, then primary, then original)
                    context = None
                    if cfg.topic_contexts:
                        context = (
                            cfg.topic_contexts.get(secondary_topic)
                            or cfg.topic_contexts.get(primary_topic)
                            or cfg.topic_contexts.get(original_topic)
                        )

                    # Generate search queries
                    queries = await generate_search_queries(
                        llm,
                        secondary_topic,
                        num_searches=3,
                        rate_limiter=llm_rate_limiter,
                    )
                    if not queries:
                        queries = [secondary_topic]

                    # Execute searches
                    search_data = await search_topic(
                        None,  # search_agent not used when tools/llm provided
                        secondary_topic,
                        num_searches=len(queries),
                        tools=tools,
                        llm=llm,
                        rate_limiter=search_rate_limiter,
                    )

                    searches = search_data.get("searches", [])

                    # Generate messages
                    messages = await generate_messages_per_topic(
                        message_agent,
                        secondary_topic,
                        searches,
                        messages_per_topic=cfg.messages_per_topic,
                        rate_limiter=llm_rate_limiter,
                        topic_context=context,
                    )

                    # Write messages to CSV and track count
                    message_count = len(messages)
                    for message in messages:
                        csv_writer.writerow(
                            [original_topic, primary_topic, secondary_topic, message]
                        )
                        csv_file_handle.flush()

                    # Update message counter
                    if cfg.max_messages is not None:
                        async with message_counter_lock:
                            message_counter[0] += message_count

                    logger.info(
                        "Completed secondary topic %r (%d messages saved)",
                        secondary_topic,
                        message_count,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process secondary topic %r: %s", secondary_topic, e
                    )

        async def process_primary_topic(primary_topic: str) -> None:
            """Process a single primary topic (gap topic)."""
            async with primary_semaphore:
                try:
                    # Look up topic context (check primary, then original)
                    context = None
                    if cfg.topic_contexts:
                        context = cfg.topic_contexts.get(
                            primary_topic
                        ) or cfg.topic_contexts.get(cfg.initial_topic)

                    # Generate secondary topics
                    logger.info(
                        "Generating secondary topics for gap topic: %r",
                        primary_topic,
                    )
                    secondary_topics = await generate_topics(
                        llm, primary_topic, cfg.num_topics, llm_rate_limiter, context
                    )

                    # Process secondary topics in parallel
                    await asyncio.gather(
                        *[
                            process_secondary_topic(
                                cfg.initial_topic, primary_topic, st
                            )
                            for st in secondary_topics
                        ]
                    )

                    logger.info("Completed gap topic: %r", primary_topic)

                except Exception as e:
                    logger.error("Failed to process gap topic %r: %s", primary_topic, e)

        # Process gap topics
        await asyncio.gather(*[process_primary_topic(gt) for gt in gap_topics])

        # Check if we've reached max_messages
        if cfg.max_messages is not None:
            async with message_counter_lock:
                current_count = message_counter[0]
            if current_count >= cfg.max_messages:
                logger.info(
                    "Reached max_messages limit (%d/%d)",
                    current_count,
                    cfg.max_messages,
                )

        csv_file_handle.close()
        logger.info("Completed successfully. Results written to %s", cfg.output_file)

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        csv_file_handle.close()
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze existing messages to identify gaps, then generate new messages "
            "to fill those gaps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topic",
        required=False,
        nargs="+",
        help=(
            "Topic(s) to analyze gaps for (e.g., 'food', 'sports'). "
            "Can specify multiple topics. Multi-word topics must be quoted. "
            "Not required if topics are specified in --config file."
        ),
    )
    parser.add_argument(
        "--existing-messages",
        type=str,
        required=True,
        help="Path to CSV file with existing messages to analyze",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=None,
        help="Number of secondary topics to generate per gap topic (default: from config or 20)",
    )
    parser.add_argument(
        "--num-primary-topics",
        "--num-gaps",
        type=int,
        dest="num_primary_topics",
        default=None,
        help="Number of gap topics to identify (default: from config or 1)",
    )
    parser.add_argument(
        "--min-messages-per-topic",
        type=int,
        default=5,
        help="Minimum messages per topic to consider it well-covered (default: 5)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: auto-generated CSV)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        help="Maximum number of messages to generate (default: unlimited)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help=f"Maximum concurrent secondary topics (default: {config.DEFAULT_MAX_CONCURRENT_TOPICS})",
    )
    parser.add_argument(
        "--max-concurrent-primary",
        type=int,
        help=f"Maximum concurrent primary topics (default: {config.DEFAULT_MAX_CONCURRENT_PRIMARY_TOPICS})",
    )
    parser.add_argument(
        "--messages-per-topic",
        type=int,
        default=15,
        help="Number of messages to generate per secondary topic (default: 15)",
    )
    parser.add_argument(
        "--topic-contexts",
        type=str,
        help=(
            "Path to JSON file mapping topic names to context descriptions, "
            'or JSON string. Example: \'{"image generation": "when users want '
            "the LLM to generate images\"}'"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to YAML configuration file. This file can contain all settings "
            "including models, topics, contexts, and parameters. Command-line "
            "arguments override values in the config file."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        dest="metrics_json",
        default=None,
        help=(
            "Path to per-label metrics JSON file from model evaluation. "
            "When provided, gap analysis will prioritize topics with low F1 scores."
        ),
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

    # Setup basic logging
    level = logging.DEBUG if cfg_dict["verbose"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Get topics from config or CLI args
    topics = cfg_dict["topics"]
    if not topics:
        if not args.topic:
            logger.error(
                "No topics specified. Use --topic or provide topics in config file."
            )
            sys.exit(1)
        topics = args.topic if isinstance(args.topic, list) else [args.topic]

    # Determine CSV output filename
    if cfg_dict["output_file"]:
        csv_file = (
            cfg_dict["output_file"]
            if cfg_dict["output_file"].endswith(".csv")
            else cfg_dict["output_file"] + ".csv"
        )
    else:
        if len(topics) == 1:
            csv_file = f"{topics[0]}_gaps.csv"
        else:
            csv_file = "topics_gaps.csv"

    logger.info("Processing %d topic(s): %s", len(topics), ", ".join(topics))
    if cfg_dict["topic_contexts"]:
        logger.info("Using %d topic context(s)", len(cfg_dict["topic_contexts"]))

    # Process each topic sequentially
    for i, topic in enumerate(topics):
        logger.info("Processing topic %d/%d: %r", i + 1, len(topics), topic)

        # Create GapAnalysisConfig for this topic
        gap_cfg = GapAnalysisConfig(
            initial_topic=topic,
            num_topics=cfg_dict["num_topics"],
            num_primary_topics=cfg_dict["num_primary_topics"],
            messages_per_topic=cfg_dict["messages_per_topic"],
            max_messages=cfg_dict["max_messages"],
            max_concurrent=cfg_dict["max_concurrent"],
            max_concurrent_primary=cfg_dict["max_concurrent_primary"],
            continuous=False,  # Gap analysis doesn't support continuous mode
            resume=cfg_dict.get("resume", False),
            output_file=csv_file,
            verbose=cfg_dict["verbose"],
            topic_contexts=cfg_dict["topic_contexts"],
            llm_provider=cfg_dict["llm_provider"],
            vllm_api_base=cfg_dict["vllm_api_base"],
            vllm_model=cfg_dict["vllm_model"],
            vllm_api_key=cfg_dict["vllm_api_key"],
            vllm_temperature=cfg_dict["vllm_temperature"],
            vllm_max_tokens=cfg_dict["vllm_max_tokens"],
            vllm_enable_thinking=cfg_dict["vllm_enable_thinking"],
            bedrock_model_id=cfg_dict.get("bedrock_model_id"),
            bedrock_region=cfg_dict.get("bedrock_region", "us-east-1"),
            bedrock_max_pool_connections=cfg_dict.get(
                "bedrock_max_pool_connections", 50
            ),
            mcp_server_url=cfg_dict["mcp_server_url"],
            rate_limit_llm=cfg_dict["rate_limit_llm"],
            rate_limit_search=cfg_dict["rate_limit_search"],
            existing_messages_file=args.existing_messages,
            min_messages_per_topic=getattr(args, "min_messages_per_topic", 5),
            metrics_json=getattr(args, "metrics_json", None),
        )

        asyncio.run(main_async(gap_cfg, write_csv_header=i == 0))


if __name__ == "__main__":
    main()
