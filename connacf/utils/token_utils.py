"""
Token counting and truncation utilities.

This module provides utilities for counting tokens and truncating text
to fit within token limits. Uses a simple word-based estimation when
tiktoken is not available.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken not available, using word-based token estimation")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string.
    
    Uses tiktoken if available, otherwise falls back to word-based estimation
    (approximately 1.3 tokens per word for English text).
    
    Args:
        text: The text to count tokens for
        model: The tokenizer model to use (default: cl100k_base for GPT-4/Claude)
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, using word estimation")
    
    # Fallback: word-based estimation
    # Average English word is ~1.3 tokens
    words = text.split()
    return int(len(words) * 1.3)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "cl100k_base",
    suffix: str = "..."
) -> str:
    """Truncate text to fit within a token limit.
    
    Truncates text to approximately max_tokens, adding a suffix to indicate
    truncation. Allows 10% tolerance for tokenizer variance.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The tokenizer model to use
        suffix: Suffix to add when truncating (default: "...")
        
    Returns:
        Truncated text if over limit, original text otherwise
    """
    if not text:
        return text
    
    current_tokens = count_tokens(text, model)
    
    # Allow 10% tolerance
    if current_tokens <= max_tokens * 1.1:
        return text
    
    # Binary search for the right truncation point
    words = text.split()
    low, high = 0, len(words)
    
    while low < high:
        mid = (low + high + 1) // 2
        truncated = ' '.join(words[:mid]) + suffix
        if count_tokens(truncated, model) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    
    if low == 0:
        # Even one word is too long, just return suffix
        return suffix
    
    return ' '.join(words[:low]) + suffix


def estimate_tokens_for_prompt(
    template: str,
    variables: dict,
    model: str = "cl100k_base"
) -> int:
    """Estimate tokens for a prompt template with variables.
    
    Args:
        template: The prompt template string
        variables: Dictionary of variable values to substitute
        model: The tokenizer model to use
        
    Returns:
        Estimated total tokens after substitution
    """
    from string import Template
    
    try:
        filled = Template(template).safe_substitute(variables)
        return count_tokens(filled, model)
    except Exception as e:
        logger.warning(f"Failed to estimate prompt tokens: {e}")
        # Rough estimate: template tokens + sum of variable tokens
        total = count_tokens(template, model)
        for value in variables.values():
            if isinstance(value, str):
                total += count_tokens(value, model)
        return total
