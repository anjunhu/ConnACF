"""
Error handling and robustness utilities for MACF.

This module provides error handling mechanisms for MACF including:
- LLM call retry logic with exponential backoff
- Empty retrieval handling
- Response parsing with fallback
- Ranked list padding for insufficient items

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from connacf.macf.data_models import ItemSuggestion, ToolCall, AgentResponse

if TYPE_CHECKING:
    from connacf.agentverse.llms.base import BaseLLM
    from connacf.macf.toolkit import MACFToolkit

logger = logging.getLogger(__name__)


class LLMCallHandler:
    """
    Handles LLM call failures with exponential backoff.
    
    Implements retry logic with exponential backoff for rate-limited
    or temporarily failing LLM calls. After all retries are exhausted,
    returns None to allow the caller to use fallback mechanisms.
    
    Attributes:
        MAX_RETRIES: Maximum number of retry attempts (5)
        BASE_DELAY: Initial delay between retries in seconds (1.0)
        MAX_DELAY: Maximum delay between retries in seconds (32.0)
    
    Requirements: 10.1, 10.5
    """
    
    MAX_RETRIES: int = 5
    BASE_DELAY: float = 1.0  # seconds
    MAX_DELAY: float = 32.0  # seconds
    
    @staticmethod
    async def call_with_retry(llm: 'BaseLLM', prompt: str) -> Optional[str]:
        """
        Call LLM with exponential backoff retry.
        
        Attempts to call the LLM up to MAX_RETRIES times, with exponentially
        increasing delays between attempts. The delay doubles after each
        failure, up to MAX_DELAY.
        
        Args:
            llm: The LLM instance to call (must have agenerate_response method)
            prompt: The prompt string to send to the LLM
            
        Returns:
            Optional[str]: The LLM response content if successful,
                          None if all retries fail (allowing caller to use fallback)
        
        Example:
            >>> result = await LLMCallHandler.call_with_retry(llm, "Hello")
            >>> if result is None:
            ...     # Use fallback mechanism
            ...     result = "Default response"
        
        Requirements: 10.1, 10.5
        """
        delay = LLMCallHandler.BASE_DELAY
        
        for attempt in range(LLMCallHandler.MAX_RETRIES):
            try:
                result = await llm.agenerate_response(prompt)
                return result.content
            except Exception as e:
                if attempt == LLMCallHandler.MAX_RETRIES - 1:
                    # Final attempt failed
                    logger.error(
                        f"LLM call failed after {LLMCallHandler.MAX_RETRIES} retries: {e}"
                    )
                    return None
                
                # Log warning and retry
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                
                # Exponential backoff with max cap
                delay = min(delay * 2, LLMCallHandler.MAX_DELAY)
        
        # Should not reach here, but return None for safety
        return None


def handle_empty_retrieval(
    retrieval_type: str,
    target_user_id: int,
    query: Optional[str] = None
) -> List[int]:
    """
    Handle empty retrieval results.
    
    Logs a warning and returns an empty list, allowing the orchestrator
    to proceed with available agents rather than failing completely.
    
    Args:
        retrieval_type: Type of retrieval that returned empty results
                       (e.g., 'similar_users', 'relevant_items', 'candidates')
        target_user_id: The ID of the target user for the retrieval
        query: Optional query string if applicable
        
    Returns:
        List[int]: Empty list, allowing caller to proceed with available data
    
    Example:
        >>> similar_users = toolkit.get_similar_users(user_id, n=5)
        >>> if not similar_users:
        ...     similar_users = handle_empty_retrieval(
        ...         'similar_users', user_id
        ...     )
    
    Requirements: 10.2
    """
    message = f"Empty {retrieval_type} retrieval for user {target_user_id}"
    if query:
        message += f" with query '{query}'"
    
    logger.warning(message)
    return []


@dataclass
class FallbackResponse:
    """
    Fallback response when parsing fails.
    
    Used when an agent's raw response cannot be parsed into a proper
    AgentResponse. Contains empty suggestions and a default rationale
    indicating the parsing failure.
    
    Attributes:
        agent_id: Unique identifier for the agent
        suggestions: Empty list of suggestions (default)
        rationale: Default message indicating parsing failure
        tool_calls: Empty list of tool calls (default)
        raw_response: The original unparseable response
    
    Requirements: 10.3
    """
    agent_id: str
    suggestions: List[ItemSuggestion] = field(default_factory=list)
    rationale: str = "Unable to parse response, using fallback."
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: str = ""


def parse_agent_response(raw: str, agent_id: str) -> AgentResponse:
    """
    Parse agent response with fallback on failure.
    
    Attempts to parse a raw LLM response into a structured AgentResponse.
    If parsing fails for any reason, returns a FallbackResponse with
    empty suggestions and logs a warning.
    
    Args:
        raw: The raw text response from the LLM
        agent_id: The ID of the agent that generated the response
        
    Returns:
        AgentResponse: Parsed response if successful, or FallbackResponse
                      if parsing fails
    
    Example:
        >>> response = parse_agent_response(raw_text, "user_agent_123")
        >>> if isinstance(response, FallbackResponse):
        ...     # Handle fallback case
        ...     pass
    
    Requirements: 10.3
    """
    try:
        # Attempt to parse structured response using AgentResponse.from_raw
        # This method should be implemented in AgentResponse class
        if hasattr(AgentResponse, 'from_raw'):
            return AgentResponse.from_raw(raw, agent_id)
        
        # If from_raw is not available, attempt basic parsing
        # This is a fallback parsing strategy
        return _basic_parse_response(raw, agent_id)
        
    except Exception as e:
        logger.warning(f"Failed to parse response from {agent_id}: {e}")
        return FallbackResponse(agent_id=agent_id, raw_response=raw)


def _basic_parse_response(raw: str, agent_id: str) -> AgentResponse:
    """
    Basic parsing strategy for agent responses.
    
    Attempts to extract suggestions from raw text using simple heuristics.
    This is used when AgentResponse.from_raw is not available.
    
    Args:
        raw: The raw text response from the LLM
        agent_id: The ID of the agent that generated the response
        
    Returns:
        AgentResponse: Parsed response with extracted suggestions
        
    Raises:
        ValueError: If parsing fails completely
    """
    # Simple parsing: look for item IDs in the response
    # This is a basic implementation that can be enhanced
    import re
    
    suggestions = []
    
    # Try to find item IDs mentioned in the response
    # Pattern: "item" followed by numbers, or just numbers in context
    item_pattern = r'item[_\s]*(\d+)|#(\d+)|ID[:\s]*(\d+)'
    matches = re.findall(item_pattern, raw, re.IGNORECASE)
    
    for match in matches:
        # Get the first non-empty group
        item_id_str = next((m for m in match if m), None)
        if item_id_str:
            try:
                item_id = int(item_id_str)
                suggestions.append(ItemSuggestion(
                    item_id=item_id,
                    score=1.0,
                    reason="Extracted from response"
                ))
            except ValueError:
                continue
    
    if not suggestions:
        # If no items found, raise to trigger fallback
        raise ValueError("No item suggestions found in response")
    
    return AgentResponse(
        agent_id=agent_id,
        suggestions=suggestions,
        rationale=raw[:500] if len(raw) > 500 else raw,
        tool_calls=[],
        raw_response=raw
    )


def pad_ranked_list(
    draft_list: List[int],
    target_size: int,
    toolkit: 'MACFToolkit',
    query: str
) -> List[int]:
    """
    Pad ranked list to target size using retrieval-based candidates.
    
    Called when the discussion fails to produce enough items. Uses the
    toolkit's retrieve_by_query method to find additional candidates
    and adds them to the list (avoiding duplicates).
    
    Args:
        draft_list: Current list of recommended item IDs
        target_size: Desired size of the final list
        toolkit: MACFToolkit instance for retrieval
        query: Query string for candidate retrieval
        
    Returns:
        List[int]: Padded list with up to target_size items.
                  May contain fewer items if not enough candidates available.
    
    Example:
        >>> draft = [1, 2, 3]  # Only 3 items
        >>> padded = pad_ranked_list(draft, 10, toolkit, "action movies")
        >>> len(padded)  # Up to 10 items
    
    Requirements: 10.4
    """
    # If already at or above target size, truncate and return
    if len(draft_list) >= target_size:
        return draft_list[:target_size]
    
    # Calculate how many more items we need
    needed = target_size - len(draft_list)
    
    # Get additional candidates via query retrieval
    # Request more than needed to account for duplicates
    candidates = toolkit.retrieve_by_query(query, k=needed * 2)
    
    # Create a set for O(1) lookup of existing items
    existing_items = set(draft_list)
    
    # Add candidates not already in list
    for item_id in candidates:
        if item_id not in existing_items:
            draft_list.append(item_id)
            existing_items.add(item_id)
            
            if len(draft_list) >= target_size:
                break
    
    # If still not enough, log error but return what we have
    if len(draft_list) < target_size:
        logger.error(
            f"Could not pad list to {target_size} items, "
            f"returning {len(draft_list)} items"
        )
    
    return draft_list
