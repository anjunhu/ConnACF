"""
Audience context for audience-aware attacker manipulation.

This module provides the AudienceContext dataclass that is passed to attackers
so they can craft targeted manipulation strategies based on who is listening.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class InteractionType(str, Enum):
    """Types of interactions where audience context is relevant."""
    UU_OPINION = "uu_opinion"           # User giving opinion to another user
    UI_PITCH = "ui_pitch"               # Item pitching to user
    UI_CONCERN_RESPONSE = "ui_concern_response"  # Item addressing user concerns
    DISCUSSION = "discussion"           # General multi-agent discussion
    SYNTHESIS = "synthesis"             # User synthesizing friend opinions


class ListenerType(str, Enum):
    """Types of agents that can be listeners."""
    USER_AGENT = "UserAgent"
    ITEM_AGENT = "ItemAgent"
    REC_AGENT = "RecAgent"
    ORCHESTRATOR = "Orchestrator"


@dataclass
class AudienceContext:
    """Context about the audience for a message.
    
    Passed to attackers to enable audience-aware manipulation strategies.
    Per Requirement 14, this provides full context about who will receive
    a message so attackers can craft targeted content.
    
    Attributes:
        target_user_id: The primary user being targeted
        listener_ids: List of all agent IDs that will receive the message
        listener_types: Map of listener IDs to their types
        interaction_type: The type of interaction (U-U opinion, U-I pitch, etc.)
        conversation_history: Previous messages in the current interaction
        user_preferences: Target user's preferences (for U-I pitches)
        concerns_text: Specific concerns raised (for addressing concerns)
        round_idx: Current training round index
        batch_idx: Current batch index
    """
    
    target_user_id: int
    listener_ids: List[str]
    listener_types: Dict[str, str]
    interaction_type: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Additional context for specific interaction types
    user_preferences: Optional[str] = None
    concerns_text: Optional[str] = None
    
    # Metadata for logging/tracing
    round_idx: int = 0
    batch_idx: int = 0
    
    def __post_init__(self):
        """Validate required fields are present."""
        if self.target_user_id is None:
            raise ValueError("target_user_id is required")
        if not self.listener_ids:
            raise ValueError("listener_ids cannot be empty")
        if not self.listener_types:
            raise ValueError("listener_types cannot be empty")
        if not self.interaction_type:
            raise ValueError("interaction_type is required")
    
    @classmethod
    def for_uu_opinion(
        cls,
        target_user_id: int,
        friend_id: int,
        other_friend_ids: List[int],
        user_preferences: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> 'AudienceContext':
        """Create context for U-U opinion generation.
        
        Args:
            target_user_id: User requesting the opinion
            friend_id: Friend generating the opinion
            other_friend_ids: Other friends also participating
            user_preferences: Target user's known preferences
            conversation_history: Previous opinions in this consultation
            round_idx: Training round index
            batch_idx: Batch index
        """
        listener_ids = [f"user_{target_user_id}"] + [f"user_{fid}" for fid in other_friend_ids]
        listener_types = {f"user_{target_user_id}": ListenerType.USER_AGENT.value}
        for fid in other_friend_ids:
            listener_types[f"user_{fid}"] = ListenerType.USER_AGENT.value
        
        return cls(
            target_user_id=target_user_id,
            listener_ids=listener_ids,
            listener_types=listener_types,
            interaction_type=InteractionType.UU_OPINION.value,
            conversation_history=conversation_history or [],
            user_preferences=user_preferences,
            round_idx=round_idx,
            batch_idx=batch_idx
        )
    
    @classmethod
    def for_ui_pitch(
        cls,
        target_user_id: int,
        item_id: int,
        user_preferences: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> 'AudienceContext':
        """Create context for U-I pitch generation.
        
        Args:
            target_user_id: User receiving the pitch
            item_id: Item generating the pitch
            user_preferences: Target user's preferences
            conversation_history: Previous dialogue turns
            round_idx: Training round index
            batch_idx: Batch index
        """
        return cls(
            target_user_id=target_user_id,
            listener_ids=[f"user_{target_user_id}"],
            listener_types={f"user_{target_user_id}": ListenerType.USER_AGENT.value},
            interaction_type=InteractionType.UI_PITCH.value,
            conversation_history=conversation_history or [],
            user_preferences=user_preferences,
            round_idx=round_idx,
            batch_idx=batch_idx
        )
    
    @classmethod
    def for_ui_concern_response(
        cls,
        target_user_id: int,
        item_id: int,
        user_preferences: str,
        concerns_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> 'AudienceContext':
        """Create context for addressing user concerns.
        
        Args:
            target_user_id: User who raised concerns
            item_id: Item addressing the concerns
            user_preferences: Target user's preferences
            concerns_text: The specific concerns raised
            conversation_history: Previous dialogue turns
            round_idx: Training round index
            batch_idx: Batch index
        """
        return cls(
            target_user_id=target_user_id,
            listener_ids=[f"user_{target_user_id}"],
            listener_types={f"user_{target_user_id}": ListenerType.USER_AGENT.value},
            interaction_type=InteractionType.UI_CONCERN_RESPONSE.value,
            conversation_history=conversation_history or [],
            user_preferences=user_preferences,
            concerns_text=concerns_text,
            round_idx=round_idx,
            batch_idx=batch_idx
        )
    
    def add_to_history(self, speaker_id: str, message: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            'speaker_id': speaker_id,
            'message': message
        })
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'target_user_id': self.target_user_id,
            'listener_ids': self.listener_ids,
            'listener_types': self.listener_types,
            'interaction_type': self.interaction_type,
            'conversation_history': self.conversation_history,
            'user_preferences': self.user_preferences,
            'concerns_text': self.concerns_text,
            'round_idx': self.round_idx,
            'batch_idx': self.batch_idx
        }
