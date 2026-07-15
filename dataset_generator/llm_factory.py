"""
LLM factory for dataset generator scripts.

Creates ChatOpenAI (vLLM) or ChatBedrock instances based on provider config.
"""

from langchain_core.language_models.chat_models import BaseChatModel


def create_llm(provider: str, **kwargs) -> BaseChatModel:
    """
    Create LLM by provider: 'vllm' -> ChatOpenAI, 'bedrock' -> ChatBedrock.

    Args:
        provider: Either "vllm" (OpenAI-compatible) or "bedrock" (AWS Bedrock)
        **kwargs: Provider-specific args (base_url/model for vllm, model_id/region for bedrock)

    Returns:
        BaseChatModel instance (ChatOpenAI or ChatBedrock)
    """
    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrock

        # Increase connection pool for high-concurrency workloads (e.g. per-topic labelling)
        max_pool = kwargs.get("max_pool_connections", 50)
        config = Config(max_pool_connections=max_pool)
        return ChatBedrock(
            model_id=kwargs["model_id"],
            region_name=kwargs.get("region_name", "us-east-1"),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800),
            config=config,
        )
    elif provider == "vllm":
        from langchain_openai import ChatOpenAI

        llm_kwargs = {
            "base_url": kwargs["base_url"],
            "model": kwargs["model"],
            "api_key": kwargs.get("api_key", "EMPTY"),
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 800),
        }
        if kwargs.get("enable_thinking"):
            llm_kwargs["enable_thinking"] = True
        return ChatOpenAI(**llm_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
