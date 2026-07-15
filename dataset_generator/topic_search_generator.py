#!/usr/bin/env python3
"""
Topic Search Message Generator

Generates subtopics from an initial topic, searches them via brave-search MCP,
and generates realistic user messages about those topics.

Usage:
    # Using command-line arguments
    python topic_search_generator.py --topic "food" --num-topics 20
    python topic_search_generator.py --topic "sports" \\
        --num-topics 10 --output results.json
    
    # Using YAML config file (recommended for full runs)
    python topic_search_generator.py --config config.yaml
    
    # Config file with CLI overrides
    python topic_search_generator.py --config config.yaml --num-topics 30

Configuration File:
    Use --config to load all settings from a YAML file. This includes:
    - Model settings (VLLM API or AWS Bedrock, MCP server URLs)
    - Topics to process
    - Topic contexts
    - Generation parameters
    - Rate limiting and concurrency settings
    - Output settings
    
    See config_example.yaml for a complete example. For Bedrock, use
    models.bedrock: { model_id: "anthropic.claude-3-sonnet-20240229-v1:0",
    region_name: "us-east-1", ... } instead of models.vllm.

    Command-line arguments override values in the config file.

Topic Contexts:
    You can provide context about what a topic means to ensure generated messages
    align with the intended use case. For example, "image generation" should
    generate messages where users ask the LLM to create images, not messages
    about image classification or editing.
    
    Topic contexts can be provided via:
    1. config.py: Add entries to TOPIC_CONTEXTS dictionary
    2. Command line: --topic-contexts path/to/contexts.json
    3. Command line: --topic-contexts '{"topic": "description"}'
    
    Contexts are checked in order: secondary topic -> primary topic -> original topic
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

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

import config
from llm_factory import create_llm
from rate_limiter import RateLimiter, retry_with_backoff

logger = logging.getLogger(__name__)

# Fallback query terms for when LLM query generation fails
_FALLBACK_QUERY_TERMS = ["guide", "tips", "best practices", "examples", "resources"]


def load_completed_topics(csv_path: Path) -> set[tuple[str, str, str]]:
    """
    Load (original_topic, primary_topic, secondary_topic) tuples from an
    existing CSV output file for resume support.

    Args:
        csv_path: Path to CSV file

    Returns:
        Set of completed topic tuples
    """
    completed: set[tuple[str, str, str]] = set()
    if not csv_path.exists():
        return completed
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and header[0] != "original topic":
                return completed
            for row in reader:
                if len(row) >= 3:
                    completed.add((row[0].strip(), row[1].strip(), row[2].strip()))
    except OSError as e:
        logger.warning("Could not read existing CSV for resume: %s", e)
    return completed


@dataclasses.dataclass
class RunConfig:
    """Configuration for a single topic processing run."""

    initial_topic: str
    num_topics: int
    num_primary_topics: int
    messages_per_topic: int
    max_messages: int | None
    max_concurrent: int
    max_concurrent_primary: int
    continuous: bool
    resume: bool
    output_file: str
    verbose: bool
    topic_contexts: dict[str, str]
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
    mcp_server_url: str
    rate_limit_llm: int
    rate_limit_search: int


def _extract_agent_content(result: Any) -> str:
    """
    Extract content from a LangChain agent result.

    Args:
        result: Agent result (dict or object with messages attribute)

    Returns:
        Extracted content as string
    """
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get("content", "")
            else:
                return str(last_message)
        else:
            return str(result)
    else:
        return str(result)


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


def _parse_topic_list(content: str, limit: int) -> list[str]:
    """
    Parse a list of topics from LLM response.

    Handles various formats: numbered lists, comma-separated, one per line.

    Args:
        content: Raw LLM response content
        limit: Maximum number of topics to return

    Returns:
        List of parsed topic strings
    """
    topics = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering if present (e.g., "1. topic" -> "topic")
        if line and line[0].isdigit():
            line = line.split(".", 1)[-1].strip()
        # Split by comma if multiple topics on one line
        for topic in line.split(","):
            topic = topic.strip()
            if topic:
                topics.append(topic)
    return topics[:limit]


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

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
    result["vllm_temperature"] = getattr(config_module, "VLLM_TEMPERATURE", 0.7)
    result["vllm_enable_thinking"] = getattr(
        config_module, "VLLM_ENABLE_THINKING", False
    )
    result["mcp_server_url"] = config_module.MCP_SERVER_URL
    result["rate_limit_llm"] = config_module.RATE_LIMIT_LLM_CALLS_PER_MINUTE
    result["rate_limit_search"] = config_module.RATE_LIMIT_SEARCH_CALLS_PER_MINUTE
    result["max_concurrent"] = config_module.DEFAULT_MAX_CONCURRENT_TOPICS
    result["max_concurrent_primary"] = (
        config_module.DEFAULT_MAX_CONCURRENT_PRIMARY_TOPICS
    )
    result["num_primary_topics"] = config_module.DEFAULT_NUM_PRIMARY_TOPICS
    result["topics"] = []
    result["topic_contexts"] = (
        config_module.TOPIC_CONTEXTS.copy()
        if hasattr(config_module, "TOPIC_CONTEXTS")
        else {}
    )
    result["num_topics"] = 20
    result["messages_per_topic"] = 15
    result["max_messages"] = None
    result["continuous"] = False
    result["resume"] = False
    result["output_file"] = None
    result["verbose"] = False

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
        if "mcp" in models:
            result["mcp_server_url"] = models["mcp"].get(
                "server_url", result["mcp_server_url"]
            )

    if "topics" in cfg:
        result["topics"] = cfg["topics"]

    if "topic_contexts" in cfg:
        result["topic_contexts"].update(cfg["topic_contexts"])

    if "generation" in cfg:
        gen = cfg["generation"]
        result["num_primary_topics"] = gen.get(
            "num_primary_topics", result["num_primary_topics"]
        )
        result["num_topics"] = gen.get("num_topics", result["num_topics"])
        result["messages_per_topic"] = gen.get(
            "messages_per_topic", result["messages_per_topic"]
        )
        # Handle None/null explicitly for max_messages
        if "max_messages" in gen:
            result["max_messages"] = gen["max_messages"]  # Can be None/null
        result["continuous"] = gen.get("continuous", result["continuous"])
        result["resume"] = gen.get("resume", result["resume"])

    if "concurrency" in cfg:
        concurrency = cfg["concurrency"]
        result["max_concurrent"] = concurrency.get(
            "max_concurrent", result["max_concurrent"]
        )
        result["max_concurrent_primary"] = concurrency.get(
            "max_concurrent_primary", result["max_concurrent_primary"]
        )

    if "rate_limiting" in cfg:
        rate_limit = cfg["rate_limiting"]
        result["rate_limit_llm"] = rate_limit.get(
            "llm_calls_per_minute", result["rate_limit_llm"]
        )
        result["rate_limit_search"] = rate_limit.get(
            "search_calls_per_minute", result["rate_limit_search"]
        )

    if "output" in cfg and cfg["output"].get("file"):
        result["output_file"] = cfg["output"]["file"]

    if "logging" in cfg:
        result["verbose"] = cfg["logging"].get("verbose", result["verbose"])

    # Override with command-line args if present (highest precedence)
    if args:
        if hasattr(args, "topic") and args.topic:
            result["topics"] = (
                args.topic if isinstance(args.topic, list) else [args.topic]
            )
        if hasattr(args, "num_topics") and args.num_topics is not None:
            result["num_topics"] = args.num_topics
        if hasattr(args, "num_primary_topics") and args.num_primary_topics is not None:
            result["num_primary_topics"] = args.num_primary_topics
        if hasattr(args, "messages_per_topic") and args.messages_per_topic is not None:
            result["messages_per_topic"] = args.messages_per_topic
        if hasattr(args, "max_messages") and args.max_messages is not None:
            result["max_messages"] = args.max_messages
        if hasattr(args, "max_concurrent") and args.max_concurrent is not None:
            result["max_concurrent"] = args.max_concurrent
        if (
            hasattr(args, "max_concurrent_primary")
            and args.max_concurrent_primary is not None
        ):
            result["max_concurrent_primary"] = args.max_concurrent_primary
        if hasattr(args, "continuous"):
            result["continuous"] = args.continuous
        if hasattr(args, "resume"):
            result["resume"] = args.resume
        if hasattr(args, "output") and args.output:
            result["output_file"] = args.output
        if hasattr(args, "verbose"):
            result["verbose"] = args.verbose
        if hasattr(args, "topic_contexts") and args.topic_contexts:
            # Load topic contexts from CLI (JSON file or string)
            try:
                try:
                    cli_contexts = json.loads(args.topic_contexts)
                except json.JSONDecodeError:
                    context_path = Path(args.topic_contexts)
                    if context_path.exists():
                        with open(context_path, "r", encoding="utf-8") as f:
                            cli_contexts = json.load(f)
                    else:
                        raise FileNotFoundError(
                            f"Topic contexts file not found: {args.topic_contexts}"
                        )
                result["topic_contexts"].update(cli_contexts)
            except Exception as e:
                logger.warning("Failed to load topic contexts from CLI: %s", e)

    return result


async def generate_primary_topics(
    llm: BaseChatModel,
    seed_topic: str,
    num_primary_topics: int,
    rate_limiter: RateLimiter | None = None,
) -> list[str]:
    """
    Generate multiple primary topics from a seed topic.

    Args:
        llm: Chat model instance (BaseChatModel)
        seed_topic: Seed topic (e.g., 'food')
        num_primary_topics: Number of primary topics to generate
        rate_limiter: Optional rate limiter for LLM calls

    Returns:
        List of primary topic strings
    """
    if num_primary_topics == 1:
        return [seed_topic]

    logger.info(
        "Generating %d primary topics from seed: %r", num_primary_topics, seed_topic
    )

    prompt = f"""You are generating primary topics for the seed topic: "{seed_topic}"

Generate exactly {num_primary_topics} distinct primary topics related to "{seed_topic}".

CRITICAL: All primary topics MUST be directly related to "{seed_topic}".

Each primary topic should be:
- A broad category or domain (1-3 words)
- Directly related to the seed topic "{seed_topic}"
- Different from the others
- Something that could have many subtopics

Return ONLY a comma-separated list of topics, one per line. \
No numbering, no explanations, no extra text.

Example format (for reference - these examples are for DIFFERENT topics, NOT for "{seed_topic}"):
If seed topic was "food": food, cooking, recipes, restaurants, nutrition
If seed topic was "sports": basketball, soccer, tennis, golf, baseball
If seed topic was "technology": ai, machine learning, blockchain, cybersecurity, cloud computing

Now generate {num_primary_topics} primary topics specifically related to "{seed_topic}":
"""

    async def _generate():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        content = await retry_with_backoff(_generate)

        if not content:
            raise RuntimeError(
                "LLM returned empty response for primary topic generation"
            )

        # Parse the response
        topics = _parse_topic_list(content, num_primary_topics)
        if seed_topic not in topics and len(topics) < num_primary_topics:
            topics.insert(0, seed_topic)
        topics = topics[:num_primary_topics]

        if len(topics) < num_primary_topics:
            logger.warning(
                "Generated only %d primary topics (requested %d)",
                len(topics),
                num_primary_topics,
            )

        logger.info("Generated primary topics: %s", topics)
        return topics

    except Exception as e:
        logger.error("Failed to generate primary topics: %s", e)
        # Fallback: return seed topic
        return [seed_topic]


async def generate_topics(
    llm: BaseChatModel,
    initial_topic: str,
    num_topics: int,
    rate_limiter: RateLimiter | None = None,
    topic_context: str | None = None,
) -> list[str]:
    """
    Generate subtopics from an initial topic using the self-hosted LLM.

    Args:
        llm: Chat model instance (BaseChatModel)
        initial_topic: The initial topic (e.g., 'food')
        num_topics: Number of subtopics to generate
        rate_limiter: Optional rate limiter for LLM calls
        topic_context: Optional context/description about what the topic means
                      and what kind of subtopics should be generated

    Returns:
        List of subtopic strings

    Raises:
        RuntimeError: If LLM call fails or returns invalid response
    """
    logger.info("Generating %d subtopics for topic: %r", num_topics, initial_topic)

    context_section = ""
    if topic_context:
        context_section = f"""
IMPORTANT CONTEXT: "{initial_topic}" means: {topic_context}

All generated subtopics MUST align with this context. For example, if the topic \
is "image generation" and the context says "when users want the LLM to generate \
images", then generate subtopics about users asking the LLM to create/generate \
images, NOT subtopics about image classification, image editing tools, or other \
unrelated image tasks.

"""

    prompt = f"""You are generating subtopics for the topic: "{initial_topic}"

{context_section}Generate exactly {num_topics} distinct subtopics related to "{initial_topic}".

IMPORTANT: All subtopics MUST be directly related to "{initial_topic}".
{f"All subtopics MUST align with the context provided above." if topic_context else ""}

Each subtopic should be:
- A short phrase (2-4 words)
- Specific and actionable
- Directly related to "{initial_topic}"
- Different from the others

Return ONLY a list of subtopics, one per line. \
No numbering, no explanations, no extra text. You can use commas to separate multiple \
subtopics on the same line, or put one subtopic per line.

For example, if the initial topic is "food", the subtopics could be:
recipe ideas
restaurant reviews
cooking techniques
meal planning

For example, if the initial topic is "sports", the subtopics could be:
basketball tips
soccer drills
tennis training
golf practice

For example, if the initial topic is "technology", the subtopics could be:
ai applications
cybersecurity tips
cloud computing solutions
blockchain use cases

Now generate {num_topics} subtopics specifically related to "{initial_topic}":
"""

    async def _generate():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        content = await retry_with_backoff(_generate)
        if not content:
            raise RuntimeError("LLM returned empty response for topic generation")

        # Parse the response
        topics = _parse_topic_list(content, num_topics)

        if len(topics) < num_topics:
            logger.warning(
                "Generated only %d topics (requested %d)", len(topics), num_topics
            )

        logger.info("Generated topics: %s", topics)
        return topics

    except Exception as e:
        logger.error("Failed to generate topics: %s", e)
        raise RuntimeError(f"Topic generation failed: {e}") from e


def create_search_agent(tools: list, llm: BaseChatModel):
    """
    Create a LangChain agent for generating and executing searches.

    Args:
        tools: List of LangChain tools from MCP client
        llm: LangChain LLM instance

    Returns:
        Agent graph instance (from create_agent)
    """
    logger.info("Creating search agent...")

    # Create agent with search tools using LangChain 1.2+ API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant that generates search queries "
            "and executes them using the available search tools."
        ),
        debug=logger.isEnabledFor(logging.DEBUG),
    )

    return agent


async def generate_search_queries(
    llm: BaseChatModel,
    topic: str,
    num_searches: int = 5,
    rate_limiter: RateLimiter | None = None,
) -> list[str]:
    """
    Generate search queries for a topic without executing them.

    Args:
        llm: Chat model instance (BaseChatModel)
        topic: Topic to generate queries for
        num_searches: Number of queries to generate

    Returns:
        List of search query strings
    """
    prompt = f"""Generate exactly {num_searches} specific search queries related to "{topic}".

Each search query should be:
- Specific and actionable
- Something someone would actually search for
- Related to "{topic}"
- Different from the others

Return ONLY a JSON array of search queries, one per line. No explanations, no extra text.

Example:
["query 1", "query 2", "query 3"]
"""

    async def _generate():
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        content = await retry_with_backoff(_generate)

        # Extract JSON array
        content = _strip_markdown_json(content)
        queries = json.loads(content)
        if isinstance(queries, list):
            return queries[:num_searches]
        return []

    except Exception as e:
        logger.warning("Failed to generate search queries via LLM: %s", e)
        # Fallback: generate simple queries
        return [f"{topic} {term}" for term in _FALLBACK_QUERY_TERMS][:num_searches]


async def execute_single_search(search_agent: Any, query: str) -> dict[str, Any]:
    """
    Execute a single search query and extract summary information.

    Args:
        search_agent: Agent executor with search capabilities
        query: Search query to execute

    Returns:
        Dictionary with query and summary of results
    """
    prompt = f"""Execute a search for: "{query}"

After executing the search, provide a brief summary (2-3 sentences) of what you found.
Focus on the main topics and key information, not all the details.
"""

    try:
        # Create a fresh agent for each search to avoid context buildup.
        # The MCP SSE stream can disconnect mid-request and never deliver a
        # response, so enforce a hard timeout to prevent an indefinite hang.
        result = await asyncio.wait_for(
            search_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
            timeout=60.0,
        )

        # Extract the last message content
        response_text = _extract_agent_content(result)

        # Truncate response if too long (keep first 500 chars as summary)
        summary = response_text[:500] if len(response_text) > 500 else response_text

        return {
            "query": query,
            "summary": summary,
        }

    except asyncio.TimeoutError:
        logger.warning(
            "Search timed out for query %r (MCP stream may have disconnected)", query
        )
        return {
            "query": query,
            "summary": "Search timed out",
            "error": "Timeout waiting for MCP search response",
        }
    except Exception as e:
        error_msg = str(e)
        # Check if it's a token limit error
        if "maximum context length" in error_msg or "input_tokens" in error_msg:
            logger.error(
                "Token limit exceeded for query %r. This may indicate search results are too large.",
                query,
            )
            return {
                "query": query,
                "summary": "Search results too large to process",
                "error": "Token limit exceeded",
            }
        logger.warning("Search failed for query %r: %s", query, e)
        return {
            "query": query,
            "summary": f"Search error: {error_msg[:200]}",
            "error": error_msg,
        }


async def search_topic(
    search_agent: Any,
    topic: str,
    num_searches: int = 5,
    tools: list | None = None,
    llm: BaseChatModel | None = None,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    """
    Use search agent to generate and execute searches for a topic.

    Args:
        search_agent: Agent executor with search capabilities (may be None if tools/llm provided)
        topic: Topic to search for
        num_searches: Number of searches to generate (3-5)
        tools: List of tools for creating fresh agents (optional)
        llm: LLM instance for generating queries (optional)

    Returns:
        Dictionary with 'searches' (list of query strings) and \
'results' (list of search results)
    """
    logger.info("Searching topic: %r", topic)

    try:
        # Step 1: Generate search queries without executing them
        if llm is None:
            # Fallback: use simple queries
            queries = [f"{topic} {term}" for term in _FALLBACK_QUERY_TERMS][
                :num_searches
            ]
        else:
            queries = await generate_search_queries(
                llm, topic, num_searches, rate_limiter
            )

        if not queries:
            logger.warning("No queries generated for topic %r", topic)
            queries = [topic]  # Fallback to topic itself

        logger.info("Generated %d search queries for topic %r", len(queries), topic)

        # Step 2: Execute searches in parallel
        async def execute_search(query: str):
            """Execute a single search with rate limiting."""
            if rate_limiter:
                await rate_limiter.acquire()
            if tools is not None and llm is not None:
                fresh_agent = create_search_agent(tools, llm)
            else:
                fresh_agent = search_agent

            async def _execute():
                return await execute_single_search(fresh_agent, query)

            return await retry_with_backoff(_execute)

        # Execute all searches in parallel
        search_results = await asyncio.gather(
            *[execute_search(query) for query in queries], return_exceptions=True
        )

        searches = []
        results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error("Search failed for query %r: %s", queries[i], result)
                results.append(
                    {
                        "query": queries[i],
                        "summary": f"Search error: {str(result)[:200]}",
                        "error": str(result),
                    }
                )
            else:
                results.append(result)
            searches.append(queries[i])

        return {
            "searches": searches,
            "results": results,
        }

    except Exception as e:
        logger.error("Search failed for topic %r: %s", topic, e)
        return {
            "searches": [],
            "results": [],
            "error": str(e),
        }


def create_message_generator_agent(llm: BaseChatModel):
    """
    Create a LangChain agent for generating user messages.

    Args:
        llm: LangChain LLM instance

    Returns:
        Agent graph instance (from create_agent)
    """
    logger.info("Creating message generator agent...")

    # Create agent without tools - just LLM for message generation
    agent = create_agent(
        model=llm,
        system_prompt=(
            "You are a helpful assistant that generates realistic user "
            "messages that people would send to an LLM. Generate natural, "
            "conversational queries."
        ),
        debug=logger.isEnabledFor(logging.DEBUG),
    )

    return agent


async def generate_messages_per_topic(
    message_agent: Any,
    topic: str,
    searches: list[str],
    messages_per_topic: int = 15,
    rate_limiter: RateLimiter | None = None,
    topic_context: str | None = None,
) -> list[str]:
    """
    Generate user messages for a single topic.

    Args:
        message_agent: Agent executor for message generation
        topic: Topic name
        searches: List of search queries for this topic
        messages_per_topic: Number of messages to generate
        topic_context: Optional context/description about what the topic means
                      and what kind of messages should be generated

    Returns:
        List of user messages
    """
    context_section = ""
    if topic_context:
        context_section = f"""
IMPORTANT CONTEXT: "{topic}" means: {topic_context}

All generated messages MUST align with this context. For example, if the topic \
is "image generation" and the context says "when users want the LLM to generate \
images", then generate messages where users are asking the LLM to create/generate \
images, NOT messages about image classification, image editing, or other unrelated \
image tasks.
"""

    prompt = f"""Generate {messages_per_topic} realistic user messages \
that someone would send to an LLM about "{topic}".

{context_section}
Based on these search queries: {', '.join(searches)}

The messages should:
- Be natural and conversational
- Be specific and actionable
- Sound like real questions or requests someone would make
- Cover different aspects of "{topic}"
{f"- MUST align with the context provided above" if topic_context else ""}

Return ONLY a JSON array of messages, one per line. No explanations, \
no extra text.

Example:
["find me a vegan recipe for burritos", \
"what's the best cookbook for beginners", ...]
"""

    async def _generate():
        result = await message_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return result

    try:
        if rate_limiter:
            await rate_limiter.acquire()
        # LangChain 1.2+ agents use messages format
        result = await retry_with_backoff(_generate)
        # Extract the last message content
        response_text = _extract_agent_content(result)

        # Try to parse JSON array
        try:
            response_text = _strip_markdown_json(response_text)
            messages = json.loads(response_text)
            if isinstance(messages, list):
                return messages[:messages_per_topic]
            else:
                return []
        except json.JSONDecodeError:
            # Fallback: try to extract messages from text
            messages = []
            for line in response_text.split("\n"):
                line = line.strip()
                if line and (line.startswith('"') or line.startswith("'")):
                    # Remove quotes
                    msg = line.strip("\"'")
                    if msg:
                        messages.append(msg)
            return messages[:messages_per_topic]

    except Exception as e:
        logger.error("Failed to generate messages for topic %r: %s", topic, e)
        return []


async def main_async(cfg: RunConfig, write_csv_header: bool = True) -> None:
    """
    Main async orchestration function with scaling support.

    Args:
        cfg: RunConfig dataclass with all configuration parameters
        write_csv_header: Whether to write CSV header (for multi-topic runs)
    """
    # Setup logging
    level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Starting topic search message generator")
    logger.info(
        "Initial topic: %r, Primary topics: %d, Secondary topics per primary: %d",
        cfg.initial_topic,
        cfg.num_primary_topics,
        cfg.num_topics,
    )

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

    # Create message agent (search agents are created fresh per search)
    message_agent = create_message_generator_agent(llm)

    # Semaphores for concurrency control
    secondary_semaphore = asyncio.Semaphore(cfg.max_concurrent)
    primary_semaphore = asyncio.Semaphore(cfg.max_concurrent_primary)

    # Message counter for max_messages tracking
    message_counter: list[int] = [0]
    message_counter_lock = asyncio.Lock()

    # Load completed topics for resume support
    completed_topics: set[tuple[str, str, str]] = set()
    csv_path = Path(cfg.output_file)
    if cfg.resume and csv_path.exists():
        completed_topics = load_completed_topics(csv_path)
        completed_original_topics = {orig for (orig, _, _) in completed_topics}
        logger.info(
            "Resume enabled: skipping %d already-completed (original, primary, secondary) topic combinations",
            len(completed_topics),
        )
        if cfg.initial_topic in completed_original_topics:
            logger.info(
                "Skipping topic %r entirely (already in output file)",
                cfg.initial_topic,
            )
            return

    # Initialize CSV file (append mode for multiple topics)
    write_header = write_csv_header and not csv_path.exists()
    csv_file_handle = open(cfg.output_file, "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file_handle)
    if write_header:
        csv_writer.writerow(
            ["original topic", "primary topic", "secondary topic", "message"]
        )
        csv_file_handle.flush()

    try:
        # Generate primary topics
        logger.info("Generating primary topics...")
        primary_topics = await generate_primary_topics(
            llm, cfg.initial_topic, cfg.num_primary_topics, llm_rate_limiter
        )

        logger.info("Processing %d primary topics", len(primary_topics))

        async def process_secondary_topic(
            original_topic: str, primary_topic: str, secondary_topic: str
        ) -> None:
            """Process a single secondary topic."""
            if (original_topic, primary_topic, secondary_topic) in completed_topics:
                logger.info(
                    "Skipping %r / %r / %r (already completed)",
                    original_topic,
                    primary_topic,
                    secondary_topic,
                )
                return
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

                    # Generate search queries (rate limiting handled inside function)
                    queries = await generate_search_queries(
                        llm,
                        secondary_topic,
                        num_searches=3,
                        rate_limiter=llm_rate_limiter,
                    )
                    if not queries:
                        queries = [secondary_topic]

                    # Execute searches (already parallel in search_topic)
                    search_data = await search_topic(
                        None,  # search_agent not used when tools/llm provided
                        secondary_topic,
                        num_searches=len(queries),
                        tools=tools,
                        llm=llm,
                        rate_limiter=search_rate_limiter,
                    )

                    searches = search_data.get("searches", [])

                    # Generate messages (rate limiting handled inside function)
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
                        csv_file_handle.flush()  # Ensure incremental writes

                    # Update message counter for max_messages check
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
            """Process a single primary topic."""
            async with primary_semaphore:
                try:
                    # Look up topic context (check primary, then original)
                    context = None
                    if cfg.topic_contexts:
                        context = cfg.topic_contexts.get(
                            primary_topic
                        ) or cfg.topic_contexts.get(cfg.initial_topic)

                    # Generate secondary topics (rate limiting handled inside function)
                    logger.info(
                        "Generating secondary topics for primary topic: %r",
                        primary_topic,
                    )
                    secondary_topics = await generate_topics(
                        llm, primary_topic, cfg.num_topics, llm_rate_limiter, context
                    )

                    # Process secondary topics in parallel (with semaphore)
                    await asyncio.gather(
                        *[
                            process_secondary_topic(
                                cfg.initial_topic, primary_topic, st
                            )
                            for st in secondary_topics
                        ]
                    )

                    logger.info("Completed primary topic: %r", primary_topic)

                except Exception as e:
                    logger.error(
                        "Failed to process primary topic %r: %s", primary_topic, e
                    )

        # Process primary topics
        while True:
            # Process primary topics in parallel (with semaphore)
            await asyncio.gather(*[process_primary_topic(pt) for pt in primary_topics])

            # Check if we've reached max_messages (actual count)
            if cfg.max_messages is not None:
                async with message_counter_lock:
                    current_count = message_counter[0]
                if current_count >= cfg.max_messages:
                    logger.info(
                        "Reached max_messages limit (%d/%d), stopping",
                        current_count,
                        cfg.max_messages,
                    )
                    break

            if not cfg.continuous:
                break

            # Generate more primary topics for continuous mode
            logger.info("Generating more primary topics...")
            primary_topics = await generate_primary_topics(
                llm, cfg.initial_topic, cfg.num_primary_topics, llm_rate_limiter
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
        description=("Generate subtopics, search them, and create user messages."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--topic",
        required=False,
        nargs="+",
        help=(
            "Initial topic(s) (e.g., 'food', 'sports', 'food safety'). "
            "Can specify multiple topics. Multi-word topics must be quoted "
            "(e.g., --topic food 'food safety' sports). "
            "Not required if topics are specified in --config file."
        ),
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=None,
        help="Number of secondary topics (subtopics) to generate per primary topic (default: from config file or 20)",
    )
    parser.add_argument(
        "--num-primary-topics",
        "--depth",
        type=int,
        dest="num_primary_topics",
        default=None,
        help="Number of primary topics to generate (default: from config file or 1)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout for JSON, auto-generated for CSV)",
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
        "--continuous",
        action="store_true",
        help="Run continuously, generating more topics when queue is empty",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing output file: skip (original, primary, secondary) "
            "topic combinations that are already present in the CSV"
        ),
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
    cfg = apply_config_overrides(yaml_config, config, args)

    # Setup basic logging
    level = logging.DEBUG if cfg["verbose"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Log final config values for debugging
    logger.info(
        "Final config: num_primary_topics=%d, num_topics=%d",
        cfg["num_primary_topics"],
        cfg["num_topics"],
    )

    # Get topics from config or CLI args
    topics = cfg["topics"]
    if not topics:
        # Fallback: require --topic if not in config
        if not args.topic:
            logger.error(
                "No topics specified. Use --topic or provide topics in config file."
            )
            sys.exit(1)
        topics = args.topic if isinstance(args.topic, list) else [args.topic]

    # Determine CSV output filename
    if cfg["output_file"]:
        csv_file = (
            cfg["output_file"]
            if cfg["output_file"].endswith(".csv")
            else cfg["output_file"] + ".csv"
        )
    else:
        if len(topics) == 1:
            csv_file = f"{topics[0]}_questions.csv"
        else:
            csv_file = "topics_questions.csv"

    logger.info("Processing %d topic(s): %s", len(topics), ", ".join(topics))
    if cfg["topic_contexts"]:
        logger.info("Using %d topic context(s)", len(cfg["topic_contexts"]))

    # Process each topic sequentially
    for i, topic in enumerate(topics):
        logger.info("Processing topic %d/%d: %r", i + 1, len(topics), topic)

        # Create RunConfig for this topic
        run_cfg = RunConfig(
            initial_topic=topic,
            num_topics=cfg["num_topics"],
            num_primary_topics=cfg["num_primary_topics"],
            messages_per_topic=cfg["messages_per_topic"],
            max_messages=cfg["max_messages"],
            max_concurrent=cfg["max_concurrent"],
            max_concurrent_primary=cfg["max_concurrent_primary"],
            continuous=cfg["continuous"],
            resume=cfg["resume"],
            output_file=csv_file,
            verbose=cfg["verbose"],
            topic_contexts=cfg["topic_contexts"],
            llm_provider=cfg["llm_provider"],
            vllm_api_base=cfg["vllm_api_base"],
            vllm_model=cfg["vllm_model"],
            vllm_api_key=cfg["vllm_api_key"],
            vllm_temperature=cfg["vllm_temperature"],
            vllm_max_tokens=cfg["vllm_max_tokens"],
            vllm_enable_thinking=cfg["vllm_enable_thinking"],
            bedrock_model_id=cfg.get("bedrock_model_id"),
            bedrock_region=cfg.get("bedrock_region", "us-east-1"),
            bedrock_max_pool_connections=cfg.get("bedrock_max_pool_connections", 50),
            mcp_server_url=cfg["mcp_server_url"],
            rate_limit_llm=cfg["rate_limit_llm"],
            rate_limit_search=cfg["rate_limit_search"],
        )

        asyncio.run(main_async(run_cfg, write_csv_header=i == 0))


if __name__ == "__main__":
    main()
