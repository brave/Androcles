#!/usr/bin/env python3
"""
Rate Limiter for Topic Search Generator

Provides token bucket rate limiting and exponential backoff retry logic.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_refill

                # Refill tokens based on elapsed time
                if elapsed > 0:
                    tokens_to_add = (elapsed / self.period) * self.max_calls
                    self.tokens = min(self.max_calls, self.tokens + tokens_to_add)
                    self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                # Calculate wait time and release the lock before sleeping so
                # other tasks are not blocked while this task waits for tokens.
                wait_time = (1 - self.tokens) * self.period / self.max_calls

            logger.debug("Rate limit reached, waiting %.2f seconds", wait_time)
            await asyncio.sleep(wait_time)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays

    Returns:
        Result of function call

    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Check if it's a rate limit error
            is_rate_limit = (
                "rate limit" in error_msg
                or "429" in error_msg
                or "too many requests" in error_msg
            )

            # Don't retry on last attempt
            if attempt == max_retries:
                break

            # Calculate delay
            delay = min(base_delay * (exponential_base**attempt), max_delay)

            # Add jitter
            if jitter:
                jitter_amount = delay * 0.1 * random.random()
                delay += jitter_amount

            if is_rate_limit:
                logger.warning(
                    "Rate limit error (attempt %d/%d), retrying in %.2f seconds: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    e,
                )
            else:
                logger.warning(
                    "Error (attempt %d/%d), retrying in %.2f seconds: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    e,
                )

            await asyncio.sleep(delay)

    # All retries failed
    logger.error("All retries exhausted, raising exception")
    raise last_exception
