"""
Data models for natural language conversations.

This module provides dataclasses for representing conversation turns,
dialogue summaries, and friend opinions in U-U and U-I interactions.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import time


class SpeakerType(str, Enum):
    """Types of speakers in conversations."""
    USER = "user"
    ITEM = "item"
    FRIEND = "friend"
    SYSTEM = "system"


class Sentiment(str, Enum):
    """Overall sentiment of a dialogue."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class ConversationTurn:
    """A single turn in a conversation.
    
    Represents one message in a U-U or U-I dialogue, including
    metadata about interception by attack hooks.
    
    Attributes:
        speaker_id: ID of the agent speaking
        speaker_type: Type of speaker (user, item, friend)
        message: The message content
        timestamp: Unix timestamp of the message
        was_intercepted: Whether the message was modified by attack hooks
        original_message: Original message before interception (if intercepted)
    """
    speaker_id: str
    speaker_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    was_intercepted: bool = False
    original_message: Optional[str] = None
    
    @classmethod
    def from_user(cls, user_id: int, message: str, is_friend: bool = False) -> 'ConversationTurn':
        """Create a turn from a user agent."""
        return cls(
            speaker_id=f"user_{user_id}",
            speaker_type=SpeakerType.FRIEND.value if is_friend else SpeakerType.USER.value,
            message=message
        )
    
    @classmethod
    def from_item(cls, item_id: int, message: str) -> 'ConversationTurn':
        """Create a turn from an item agent."""
        return cls(
            speaker_id=f"item_{item_id}",
            speaker_type=SpeakerType.ITEM.value,
            message=message
        )
    
    def mark_intercepted(self, original: str) -> None:
        """Mark this turn as having been intercepted."""
        self.was_intercepted = True
        self.original_message = original
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'speaker_id': self.speaker_id,
            'speaker_type': self.speaker_type,
            'message': self.message,
            'timestamp': self.timestamp,
            'was_intercepted': self.was_intercepted,
            'original_message': self.original_message
        }


@dataclass
class FriendOpinion:
    """Opinion from a friend user in U-U interaction.
    
    Represents a friend's opinion about candidate items,
    including similarity score and adversarial status.
    
    Attributes:
        friend_id: ID of the friend user
        opinion: The natural language opinion
        similarity_score: Embedding similarity to the target user
        is_adversarial: Whether this friend is an adversarial agent
        timestamp: When the opinion was generated
    """
    friend_id: int
    opinion: str
    similarity_score: float = 0.0
    is_adversarial: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'friend_id': self.friend_id,
            'opinion': self.opinion,
            'similarity_score': self.similarity_score,
            'is_adversarial': self.is_adversarial,
            'timestamp': self.timestamp
        }


@dataclass
class DialogueSummary:
    """Summary of a U-I dialogue for decision-making.
    
    Captures the key elements of a user-item dialogue including
    the pitch, concerns raised, and responses.
    
    Attributes:
        item_id: ID of the item
        item_title: Title of the item
        pitch: The item's initial pitch
        user_concerns: List of concerns raised by the user
        item_responses: List of item responses to concerns
        overall_sentiment: Overall sentiment of the dialogue
        dialogue_turns: Full list of conversation turns
    """
    item_id: int
    item_title: str
    pitch: str
    user_concerns: List[str] = field(default_factory=list)
    item_responses: List[str] = field(default_factory=list)
    overall_sentiment: str = Sentiment.NEUTRAL.value
    dialogue_turns: List[ConversationTurn] = field(default_factory=list)
    
    def add_concern(self, concern: str) -> None:
        """Add a user concern."""
        self.user_concerns.append(concern)
    
    def add_response(self, response: str) -> None:
        """Add an item response."""
        self.item_responses.append(response)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn."""
        self.dialogue_turns.append(turn)
    
    def compute_sentiment(self) -> str:
        """Compute overall sentiment based on dialogue content.
        
        Simple heuristic based on positive/negative keywords.
        Can be enhanced with LLM-based sentiment analysis.
        """
        positive_keywords = ['love', 'great', 'excellent', 'perfect', 'enjoy', 'like', 'interested']
        negative_keywords = ['dislike', 'hate', 'boring', 'bad', 'not interested', 'concern', 'worry']
        
        all_text = ' '.join([self.pitch] + self.user_concerns + self.item_responses).lower()
        
        pos_count = sum(1 for kw in positive_keywords if kw in all_text)
        neg_count = sum(1 for kw in negative_keywords if kw in all_text)
        
        if pos_count > neg_count:
            self.overall_sentiment = Sentiment.POSITIVE.value
        elif neg_count > pos_count:
            self.overall_sentiment = Sentiment.NEGATIVE.value
        else:
            self.overall_sentiment = Sentiment.NEUTRAL.value
        
        return self.overall_sentiment
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'item_id': self.item_id,
            'item_title': self.item_title,
            'pitch': self.pitch,
            'user_concerns': self.user_concerns,
            'item_responses': self.item_responses,
            'overall_sentiment': self.overall_sentiment,
            'dialogue_turns': [t.to_dict() for t in self.dialogue_turns]
        }


@dataclass
class UUInteractionResult:
    """Result of a U-U interaction phase.
    
    Contains friend opinions and synthesized context for decision-making.
    
    Attributes:
        user_id: ID of the user who consulted friends
        friend_opinions: List of opinions from friends
        synthesized_context: Combined reasoning from all opinions
        timestamp: When the interaction completed
    """
    user_id: int
    friend_opinions: List[FriendOpinion] = field(default_factory=list)
    synthesized_context: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def add_opinion(self, opinion: FriendOpinion) -> None:
        """Add a friend opinion."""
        self.friend_opinions.append(opinion)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'friend_opinions': [o.to_dict() for o in self.friend_opinions],
            'synthesized_context': self.synthesized_context,
            'timestamp': self.timestamp
        }


@dataclass
class UIInteractionResult:
    """Result of a U-I interaction phase.
    
    Contains dialogue summaries for all candidate items.
    
    Attributes:
        user_id: ID of the user who engaged in dialogues
        dialogue_summaries: Map of item_id to DialogueSummary
        timestamp: When the interaction completed
    """
    user_id: int
    dialogue_summaries: dict = field(default_factory=dict)  # item_id -> DialogueSummary
    timestamp: float = field(default_factory=time.time)
    
    def add_dialogue(self, item_id: int, summary: DialogueSummary) -> None:
        """Add a dialogue summary for an item."""
        self.dialogue_summaries[item_id] = summary
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'dialogue_summaries': {k: v.to_dict() for k, v in self.dialogue_summaries.items()},
            'timestamp': self.timestamp
        }
