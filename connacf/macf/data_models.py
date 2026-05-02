"""
Data models for MACF (Multi-Agent Collaborative Filtering).

This module contains dataclasses for representing agent responses, suggestions,
discussion state, ranked lists, evaluation results, and tool invocations.

Requirements: 3.6, 4.6, 5.3, 5.7, 8.1
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ToolCall:
    """
    Represents a single tool invocation by an agent.
    
    Tracks which retrieval tool was called, with what arguments,
    and what results were returned.
    
    Attributes:
        tool_name: Name of the tool that was called (e.g., 'retrieve_by_query')
        arguments: Dictionary of arguments passed to the tool
        result: List of item IDs returned by the tool
        timestamp: Optional timestamp of when the call was made
    """
    tool_name: str
    arguments: Dict[str, Any]
    result: List[int]
    timestamp: Optional[str] = None


@dataclass
class ItemSuggestion:
    """
    A single item suggestion from an agent.
    
    Represents one item that an agent recommends, along with
    the agent's confidence score and reasoning.
    
    Attributes:
        item_id: The ID of the suggested item
        score: Agent's confidence/relevance score (higher is better)
        reason: Explanation of why this item is suggested
    
    Requirements: 3.6, 4.6
    """
    item_id: int
    score: float
    reason: str


@dataclass
class AgentResponse:
    """
    Response from a MACF agent.
    
    Contains the agent's suggestions, overall rationale, any tool calls
    made during response generation, and the raw LLM response.
    
    Attributes:
        agent_id: Unique identifier for the agent (e.g., 'user_agent_123')
        suggestions: List of item suggestions with scores and reasons
        rationale: Overall reasoning for the suggestions
        tool_calls: List of retrieval tool invocations made
        raw_response: The raw text response from the LLM
    
    Requirements: 3.6, 4.6
    """
    agent_id: str
    suggestions: List[ItemSuggestion]
    rationale: str
    tool_calls: List[ToolCall]
    raw_response: str


@dataclass
class RankedList:
    """
    Final ranked recommendation list.
    
    Contains the ordered list of recommended items along with their
    aggregated scores, rationales, and the full discussion history.
    
    Attributes:
        items: Ordered list of item IDs (most relevant first)
        scores: Aggregated scores for each item (same order as items)
        rationales: Aggregated rationales per item (same order as items)
        discussion_log: Full discussion history from all rounds
    
    Requirements: 5.3, 5.7
    """
    items: List[int]
    scores: List[float]
    rationales: List[str]
    discussion_log: List[Dict[str, Any]]
    
    def to_item_ids(self) -> List[int]:
        """Return just the item IDs."""
        return self.items
    
    def __len__(self) -> int:
        """Return the number of items in the ranked list."""
        return len(self.items)


@dataclass
class DiscussionState:
    """
    State of a multi-round discussion.
    
    Tracks the current state of the MACF discussion including
    round number, draft recommendations, active agents, and
    whether convergence has been reached.
    
    Attributes:
        round_idx: Current round index (0-based)
        draft_list: Current draft of recommended item IDs
        active_agents: List of agent IDs currently participating
        message_history: Messages from all previous rounds
        conflicts: List of identified conflicts between agent suggestions
        converged: Whether the discussion has converged
    
    Requirements: 5.3
    """
    round_idx: int
    draft_list: List[int]
    active_agents: List[str]
    message_history: List[Dict[str, Any]]
    conflicts: List[str]
    converged: bool


@dataclass
class EvaluationResult:
    """
    Evaluation result for a single query.
    
    Contains the recommendation results and computed metrics
    for a single user-query evaluation.
    
    Attributes:
        user_id: The target user ID
        query: The natural language query
        ranked_list: The recommended item IDs in order
        ground_truth: The actual relevant item IDs
        hit_at_10: Hit@10 metric (1.0 if any ground truth in top 10, else 0.0)
        ndcg_at_10: NDCG@10 metric for ranking quality
        num_rounds: Number of discussion rounds used
        discussion_log: Full discussion history for debugging
    
    Requirements: 8.1
    """
    user_id: int
    query: str
    ranked_list: List[int]
    ground_truth: List[int]
    hit_at_10: float
    ndcg_at_10: float
    num_rounds: int
    discussion_log: List[Dict[str, Any]]
