"""
MACF Persistent Memory Store

Provides persistent memory for MACF agents across tasks/queries.
Similar to ConnaCF's update_memory mechanism but adapted for MACF's
multi-agent discussion framework.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry for an agent."""
    content: str
    timestamp: str
    query: str  # The query that generated this memory
    round_idx: int  # Which discussion round
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        return cls(**data)


@dataclass 
class AgentMemory:
    """Memory container for a single agent.
    
    Contains both:
    - Episodic memory: Specific interaction memories (what happened)
    - Semantic memory: User portrait / item profile (who they are)
    
    Similar to ConnaCF's memory structure with role_description and update_memory.
    """
    agent_type: str  # 'user' or 'item'
    entity_id: int  # user_id or item_id
    memories: List[MemoryEntry] = field(default_factory=list)  # Episodic memories
    summary: str = ""  # Condensed summary of all memories
    
    # Semantic memory: persistent profile/portrait
    profile: str = ""  # User portrait or item description (semantic)
    preferences: List[str] = field(default_factory=list)  # Learned preferences
    personality_traits: List[str] = field(default_factory=list)  # Personality/characteristics
    
    # Original profile tracking (for contamination detection)
    _original_profile: str = ""  # Profile before any interactions (for LLM judge comparison)
    
    def add_memory(self, content: str, query: str, round_idx: int, context: Dict = None):
        """Add a new episodic memory entry."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now().isoformat(),
            query=query,
            round_idx=round_idx,
            context=context or {}
        )
        self.memories.append(entry)
        
        # Keep only last N memories to prevent unbounded growth
        max_memories = 50
        if len(self.memories) > max_memories:
            self.memories = self.memories[-max_memories:]
    
    def set_profile(self, profile: str):
        """Set the semantic profile (user portrait / item description).
        
        Also stores the original profile if not already set (for contamination detection).
        """
        # Store original profile the first time it's set
        if not self._original_profile and profile:
            self._original_profile = profile
        self.profile = profile
    
    def add_preference(self, preference: str):
        """Add a learned preference."""
        if preference not in self.preferences:
            self.preferences.append(preference)
            # Keep only last 20 preferences
            if len(self.preferences) > 20:
                self.preferences = self.preferences[-20:]
    
    def add_trait(self, trait: str):
        """Add a personality trait or characteristic."""
        if trait not in self.personality_traits:
            self.personality_traits.append(trait)
            # Keep only last 10 traits
            if len(self.personality_traits) > 10:
                self.personality_traits = self.personality_traits[-10:]
    
    def get_recent_memories(self, n: int = 5) -> List[str]:
        """Get the N most recent episodic memory contents."""
        return [m.content for m in self.memories[-n:]]
    
    def get_memory_context(self) -> str:
        """Get formatted memory context for prompts (both semantic and episodic)."""
        context_parts = []
        
        # Semantic memory: Profile/Portrait
        if self.profile:
            context_parts.append(f"Profile: {self.profile}")
        
        # Semantic memory: Preferences
        if self.preferences:
            prefs = ", ".join(self.preferences[-5:])  # Last 5 preferences
            context_parts.append(f"Known preferences: {prefs}")
        
        # Semantic memory: Traits
        if self.personality_traits:
            traits = ", ".join(self.personality_traits)
            context_parts.append(f"Characteristics: {traits}")
        
        # Episodic memory: Recent interactions
        if self.memories:
            recent = self.get_recent_memories(5)
            context_parts.append("Recent interactions:")
            for i, mem in enumerate(recent, 1):
                context_parts.append(f"  {i}. {mem[:300]}{'...' if len(mem) > 300 else ''}")
        
        # Summary
        if self.summary:
            context_parts.append(f"Summary: {self.summary}")
        
        if not context_parts:
            return "No previous interaction history."
        
        return "\n".join(context_parts)
    
    def get_self_introduction(self) -> str:
        """Get a self-introduction string for the agent (like ConnaCF's role_description)."""
        if self.agent_type == 'user':
            intro = f"I am a user (ID: {self.entity_id})."
            if self.profile:
                intro += f" {self.profile}"
            if self.preferences:
                intro += f" I prefer: {', '.join(self.preferences[-3:])}."
            if self.personality_traits:
                intro += f" I am {', '.join(self.personality_traits)}."
            return intro
        else:  # item
            intro = f"I am an item (ID: {self.entity_id})."
            if self.profile:
                intro += f" {self.profile}"
            if self.personality_traits:
                intro += f" Key features: {', '.join(self.personality_traits)}."
            return intro
    
    def to_dict(self) -> Dict:
        return {
            'agent_type': self.agent_type,
            'entity_id': self.entity_id,
            'memories': [m.to_dict() for m in self.memories],
            'summary': self.summary,
            'profile': self.profile,
            'preferences': self.preferences,
            'personality_traits': self.personality_traits,
            '_original_profile': self._original_profile
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMemory':
        memories = [MemoryEntry.from_dict(m) for m in data.get('memories', [])]
        agent = cls(
            agent_type=data['agent_type'],
            entity_id=data['entity_id'],
            memories=memories,
            summary=data.get('summary', ''),
            profile=data.get('profile', ''),
            preferences=data.get('preferences', []),
            personality_traits=data.get('personality_traits', [])
        )
        # Restore original profile if available
        agent._original_profile = data.get('_original_profile', '')
        return agent


class MACFMemoryStore:
    """
    Persistent memory store for MACF agents.
    
    Stores and retrieves agent memories across tasks/queries.
    Memories can be persisted to disk for long-term storage.
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_save: bool = True,
        max_memories_per_agent: int = 50
    ):
        """
        Initialize the memory store.
        
        Args:
            persist_path: Path to save/load memories. If None, memories are in-memory only.
            auto_save: If True, automatically save after each update
            max_memories_per_agent: Maximum memories to keep per agent
        """
        self.persist_path = persist_path
        self.auto_save = auto_save
        self.max_memories_per_agent = max_memories_per_agent
        
        # Memory storage: {agent_key: AgentMemory}
        # agent_key format: "user_{id}" or "item_{id}"
        self._memories: Dict[str, AgentMemory] = {}
        
        # Load existing memories if path provided
        if persist_path and os.path.exists(persist_path):
            self.load()
        
        logger.info(
            f"MACFMemoryStore initialized with {len(self._memories)} agents"
            + (f", persist_path={persist_path}" if persist_path else " (in-memory only)")
        )
    
    def _get_key(self, agent_type: str, entity_id: int) -> str:
        """Generate storage key for an agent."""
        return f"{agent_type}_{entity_id}"
    
    def get_user_memory(self, user_id: int) -> AgentMemory:
        """Get or create memory for a user agent."""
        key = self._get_key('user', user_id)
        if key not in self._memories:
            self._memories[key] = AgentMemory(agent_type='user', entity_id=user_id)
        return self._memories[key]
    
    def get_item_memory(self, item_id: int) -> AgentMemory:
        """Get or create memory for an item agent."""
        key = self._get_key('item', item_id)
        if key not in self._memories:
            self._memories[key] = AgentMemory(agent_type='item', entity_id=item_id)
        return self._memories[key]
    
    def update_user_memory(
        self,
        user_id: int,
        content: str,
        query: str,
        round_idx: int,
        context: Dict = None
    ):
        """Add a memory entry for a user agent."""
        memory = self.get_user_memory(user_id)
        memory.add_memory(content, query, round_idx, context)
        
        if self.auto_save and self.persist_path:
            self.save()
        
        logger.debug(f"Updated memory for user_{user_id}: {content[:50]}...")
    
    def update_item_memory(
        self,
        item_id: int,
        content: str,
        query: str,
        round_idx: int,
        context: Dict = None
    ):
        """Add a memory entry for an item agent."""
        memory = self.get_item_memory(item_id)
        memory.add_memory(content, query, round_idx, context)
        
        if self.auto_save and self.persist_path:
            self.save()
        
        logger.debug(f"Updated memory for item_{item_id}: {content[:50]}...")
    
    def update_summary(self, agent_type: str, entity_id: int, summary: str):
        """Update the condensed summary for an agent."""
        key = self._get_key(agent_type, entity_id)
        if key in self._memories:
            self._memories[key].summary = summary
            
            if self.auto_save and self.persist_path:
                self.save()
    
    def set_user_profile(self, user_id: int, profile: str):
        """Set the semantic profile (portrait) for a user agent."""
        memory = self.get_user_memory(user_id)
        memory.set_profile(profile)
        
        if self.auto_save and self.persist_path:
            self.save()
        
        logger.debug(f"Set profile for user_{user_id}: {profile[:50]}...")
    
    def set_item_profile(self, item_id: int, profile: str):
        """Set the semantic profile (description) for an item agent."""
        memory = self.get_item_memory(item_id)
        memory.set_profile(profile)
        
        if self.auto_save and self.persist_path:
            self.save()
        
        logger.debug(f"Set profile for item_{item_id}: {profile[:50]}...")
    
    def add_user_preference(self, user_id: int, preference: str):
        """Add a learned preference for a user agent."""
        memory = self.get_user_memory(user_id)
        memory.add_preference(preference)
        
        if self.auto_save and self.persist_path:
            self.save()
    
    def add_user_trait(self, user_id: int, trait: str):
        """Add a personality trait for a user agent."""
        memory = self.get_user_memory(user_id)
        memory.add_trait(trait)
        
        if self.auto_save and self.persist_path:
            self.save()
    
    def add_item_trait(self, item_id: int, trait: str):
        """Add a characteristic for an item agent."""
        memory = self.get_item_memory(item_id)
        memory.add_trait(trait)
        
        if self.auto_save and self.persist_path:
            self.save()
    
    def get_user_introduction(self, user_id: int) -> str:
        """Get self-introduction for a user agent."""
        memory = self.get_user_memory(user_id)
        return memory.get_self_introduction()
    
    def get_item_introduction(self, item_id: int) -> str:
        """Get self-introduction for an item agent."""
        memory = self.get_item_memory(item_id)
        return memory.get_self_introduction()
    
    def get_all_user_memories(self) -> Dict[int, AgentMemory]:
        """Get all user memories."""
        return {
            int(k.split('_')[1]): v 
            for k, v in self._memories.items() 
            if k.startswith('user_')
        }
    
    def get_all_item_memories(self) -> Dict[int, AgentMemory]:
        """Get all item memories."""
        return {
            int(k.split('_')[1]): v 
            for k, v in self._memories.items() 
            if k.startswith('item_')
        }
    
    def save(self):
        """Save memories to disk."""
        if not self.persist_path:
            logger.warning("No persist_path set, cannot save memories")
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.persist_path) or '.', exist_ok=True)
        
        data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'memories': {k: v.to_dict() for k, v in self._memories.items()}
        }
        
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self._memories)} agent memories to {self.persist_path}")
    
    def load(self):
        """Load memories from disk."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            logger.warning(f"Cannot load memories: path does not exist: {self.persist_path}")
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self._memories = {
                k: AgentMemory.from_dict(v) 
                for k, v in data.get('memories', {}).items()
            }
            
            logger.info(f"Loaded {len(self._memories)} agent memories from {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            self._memories = {}
    
    def clear(self):
        """Clear all memories."""
        self._memories = {}
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
        logger.info("Cleared all agent memories")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        user_count = sum(1 for k in self._memories if k.startswith('user_'))
        item_count = sum(1 for k in self._memories if k.startswith('item_'))
        total_memories = sum(len(m.memories) for m in self._memories.values())
        
        return {
            'user_agents': user_count,
            'item_agents': item_count,
            'total_memories': total_memories,
            'persist_path': self.persist_path
        }
