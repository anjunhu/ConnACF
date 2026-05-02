"""
MACF Agent Base Classes.

This module contains the abstract base class for MACF agents.
MACF agents now support persistent memory across tasks/queries,
similar to ConnaCF's update_memory mechanism.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from connacf.agentverse.llms.base import BaseLLM
from connacf.macf.data_models import AgentResponse, ToolCall
from connacf.macf.toolkit import MACFToolkit

# Import for type checking to avoid circular imports
if TYPE_CHECKING:
    from connacf.macf.memory_store import AgentMemory

logger = logging.getLogger(__name__)


class BaseMACFAgent(ABC):
    """
    Base class for MACF agents with optional persistent memory.
    
    Agents can now maintain persistent memory across tasks/queries
    when a memory object is provided. This enables learning and
    evolution similar to ConnaCF agents.
    
    Attributes:
        agent_id: Unique identifier for this agent (e.g., 'user_agent_123')
        profile: Agent-specific profile data (preferences, attributes, etc.)
        llm: LLM wrapper for generating responses
        toolkit: MACFToolkit for retrieval tool access
        target_user_context: Context about the target user for recommendations
        query: The natural language query being processed
        message_history: History of messages for this agent's conversation
        memory: Optional persistent memory from MACFMemoryStore
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5
    """
    
    def __init__(
        self,
        agent_id: str,
        profile: Dict[str, Any],
        llm: BaseLLM,
        toolkit: MACFToolkit,
        target_user_context: Dict[str, Any],
        query: str,
        memory: Optional['AgentMemory'] = None
    ):
        """
        Initialize a MACF agent.
        
        Args:
            agent_id: Unique identifier for this agent (e.g., 'user_agent_123')
            profile: Agent-specific profile containing preferences/attributes
            llm: LLM wrapper instance for generating responses
            toolkit: MACFToolkit instance for retrieval tool access
            target_user_context: Context about the target user including their
                                 preferences, history summary, and other relevant info
            query: The natural language query expressing user intent
            memory: Optional AgentMemory for persistent memory across tasks
        
        Validates: Requirements 3.2, 3.3, 4.2, 4.3
        """
        self.agent_id = agent_id
        self.profile = profile
        self.llm = llm
        self.toolkit = toolkit
        self.target_user_context = target_user_context
        self.query = query
        self.message_history: List[Dict[str, str]] = []
        self.memory = memory  # Persistent memory (optional)
        
        logger.debug(f"Initialized {self.__class__.__name__} with agent_id={agent_id}, memory={'enabled' if memory else 'disabled'}")
    
    def get_memory_context(self) -> str:
        """
        Get formatted memory context for inclusion in prompts.
        
        Returns:
            str: Formatted memory context or empty string if no memory
        """
        if self.memory is None:
            return ""
        return self.memory.get_memory_context()
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Generate system prompt based on agent type and profile.
        
        The system prompt should include:
        - Agent's role and perspective
        - Profile information (preferences, attributes)
        - Target user context
        - Current query
        - Persistent memory context (if available)
        - Instructions for generating recommendations
        
        Returns:
            str: The system prompt for this agent
        
        Validates: Requirements 3.4, 4.4
        """
        pass
    
    @abstractmethod
    def reply(
        self, 
        instruction: str, 
        discussion_history: List[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Generate response to orchestrator instruction.
        
        The agent should:
        1. Consider the instruction from the orchestrator
        2. Review the discussion history from previous rounds
        3. Optionally call retrieval tools to find relevant items
        4. Generate item suggestions with rationales
        
        Args:
            instruction: Personalized instruction from the orchestrator
                        guiding what the agent should focus on
            discussion_history: Messages from previous discussion rounds,
                               each containing agent_id, content, and round info
            
        Returns:
            AgentResponse containing:
            - agent_id: This agent's identifier
            - suggestions: List of ItemSuggestion with item_id, score, reason
            - rationale: Overall reasoning for the suggestions
            - tool_calls: List of retrieval tool invocations made
            - raw_response: The raw LLM response text
        
        Validates: Requirements 3.5, 3.6, 4.5, 4.6
        """
        pass
    
    def call_tool(self, tool_name: str, **kwargs) -> List[int]:
        """
        Call a retrieval tool and return results.
        
        Provides a unified interface for agents to invoke retrieval tools
        from the MACFToolkit. Supported tools:
        - get_similar_users: Find users similar to target
        - get_relevant_items: Find history items relevant to query
        - retrieve_by_query: Find candidates by query embedding
        - retrieve_by_item: Find candidates similar to an item
        
        Args:
            tool_name: Name of the retrieval tool to call. Must be one of:
                      'get_similar_users', 'get_relevant_items',
                      'retrieve_by_query', 'retrieve_by_item'
            **kwargs: Arguments to pass to the tool (varies by tool)
        
        Returns:
            List[int]: List of user IDs or item IDs returned by the tool.
                      Returns empty list if tool_name is invalid.
        
        Example:
            # Find similar users
            similar_users = agent.call_tool('get_similar_users', 
                                           target_user_id=123, n=5)
            
            # Find items by query
            candidates = agent.call_tool('retrieve_by_query', 
                                        query='action movies', k=15)
        
        Validates: Requirements 3.5, 4.5
        """
        # Map tool names to toolkit methods
        tool_map = {
            'get_similar_users': self.toolkit.get_similar_users,
            'get_relevant_items': self.toolkit.get_relevant_items,
            'retrieve_by_query': self.toolkit.retrieve_by_query,
            'retrieve_by_item': self.toolkit.retrieve_by_item,
        }
        
        if tool_name not in tool_map:
            logger.warning(
                f"Agent {self.agent_id}: Unknown tool '{tool_name}'. "
                f"Valid tools: {list(tool_map.keys())}"
            )
            return []
        
        try:
            result = tool_map[tool_name](**kwargs)
            logger.debug(
                f"Agent {self.agent_id}: Tool '{tool_name}' returned "
                f"{len(result)} results"
            )
            return result
        except Exception as e:
            logger.warning(
                f"Agent {self.agent_id}: Error calling tool '{tool_name}': {e}"
            )
            return []
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"agent_id='{self.agent_id}', "
            f"query='{self.query[:50]}...' if len(self.query) > 50 else self.query)"
        )


class UserAgent(BaseMACFAgent):
    """
    Agent representing a similar user (neighbor).
    
    Reasons about how the target user aligns with its own preferences
    and suggests items based on that perspective.
    
    The UserAgent is instantiated for each similar user (neighbor) identified
    during agent recruitment. It uses its knowledge of the neighbor's preferences
    and interaction history to suggest items that might appeal to the target user.
    
    Attributes:
        neighbor_user_id: The ID of the neighbor user this agent represents
        neighbor_profile: Preference patterns of the neighbor user
        neighbor_history_summary: Summary of the neighbor's interaction history
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
    """
    
    def __init__(
        self,
        neighbor_user_id: int,
        neighbor_profile: Dict[str, Any],
        neighbor_history_summary: str,
        llm: BaseLLM,
        toolkit: MACFToolkit,
        target_user_context: Dict[str, Any],
        query: str,
        memory: Optional['AgentMemory'] = None,
        neighbor_item_descriptions: Optional[Dict[int, str]] = None
    ):
        """
        Initialize a UserAgent representing a similar user.
        
        Args:
            neighbor_user_id: The ID of the neighbor user this agent represents
            neighbor_profile: Dictionary containing the neighbor's preference patterns
                             (e.g., genre preferences, rating patterns, etc.)
            neighbor_history_summary: Text summary of the neighbor's interaction history
            llm: LLM wrapper instance for generating responses
            toolkit: MACFToolkit instance for retrieval tool access
            target_user_context: Context about the target user including their
                                preferences, history summary, and other relevant info
            query: The natural language query expressing user intent
            memory: Optional AgentMemory for persistent memory across tasks
            neighbor_item_descriptions: Dict mapping item_id -> description for items
                                       in this user's history (for semantic content)
        
        Validates: Requirements 3.1, 3.2, 3.3
        """
        # Build the profile dictionary with neighbor-specific information
        profile = {
            'user_id': neighbor_user_id,
            'preference_pattern': neighbor_profile,
            'history_summary': neighbor_history_summary
        }
        
        # Initialize base class with constructed profile
        super().__init__(
            agent_id=f"user_agent_{neighbor_user_id}",
            profile=profile,
            llm=llm,
            toolkit=toolkit,
            target_user_context=target_user_context,
            query=query,
            memory=memory
        )
        
        # Store neighbor-specific attributes for easy access
        self.neighbor_user_id = neighbor_user_id
        self.neighbor_profile = neighbor_profile
        self.neighbor_history_summary = neighbor_history_summary
        self.neighbor_item_descriptions = neighbor_item_descriptions or {}
        
        logger.debug(
            f"Initialized UserAgent for neighbor user {neighbor_user_id} with {len(self.neighbor_item_descriptions)} item descriptions"
        )
    
    def get_system_prompt(self) -> str:
        """
        Generate system prompt for user agent.
        
        The system prompt includes:
        - The neighbor's preferences and history summary
        - ACTUAL ITEM DESCRIPTIONS from the neighbor's history
        - Persistent memory context (if available)
        - Context about the target user
        - The current query
        - Instructions to reason about preference alignment
        - Instructions for generating item suggestions WITH SPECIFIC CONTENT
        
        Returns:
            str: The system prompt for this user agent
        
        Validates: Requirements 3.4
        """
        # Extract target user information
        target_user_id = self.target_user_context.get('user_id', 'unknown')
        target_preferences = self.target_user_context.get('preferences', {})
        target_history_summary = self.target_user_context.get('history_summary', 'No history available')
        
        # Format neighbor preferences for the prompt
        neighbor_prefs_str = self._format_preferences(self.neighbor_profile)
        target_prefs_str = self._format_preferences(target_preferences)
        
        # Format actual item descriptions from neighbor's history
        items_section = self._format_item_descriptions()
        
        # Get persistent memory context
        memory_context = self.get_memory_context()
        memory_section = ""
        if memory_context:
            memory_section = f"""
## Your Memory (Learnings from Previous Interactions)
{memory_context}
"""
        
        system_prompt = f"""You are a User Agent representing User {self.neighbor_user_id} in a multi-agent collaborative filtering system.

## Your Role
You represent a user who has similar preferences to the target user. Your job is to reason about how the target user's preferences align with your own, and suggest items that you think would appeal to them based on your perspective.

## Your Preferences and History
{neighbor_prefs_str}

Your interaction history summary:
{self.neighbor_history_summary}

## Items You Have Interacted With (YOUR KNOWLEDGE BASE)
{items_section}
{memory_section}
## Target User Context
You are helping to generate recommendations for User {target_user_id}.

Target user's preferences:
{target_prefs_str}

Target user's history summary:
{target_history_summary}

## Current Query
The target user is looking for: "{self.query}"

## Your Task
1. Analyze how the target user's preferences align with your own
2. Consider what items from your experience would appeal to someone with similar tastes
3. Use retrieval tools if needed to find relevant items
4. Suggest items with clear rationales explaining why they would appeal to the target user
5. Learn from this interaction to improve future recommendations

## CRITICAL INSTRUCTIONS FOR YOUR RESPONSE
When making suggestions, you MUST:
- Quote ACTUAL item titles and descriptions (e.g., "I recommend 'Abbey Road' by The Beatles because...")
- Reference SPECIFIC content characteristics (genres, artists, styles, themes)
- Explain your preferences using concrete examples (e.g., "I enjoy jazz fusion albums like...")
- Connect item attributes to the user's query (e.g., "Since you're looking for relaxing music, this album featuring acoustic guitar...")

DO NOT give generic responses like "based on similar preferences" without citing specific item content.

## Response Format
Provide your suggestions in the following format:
- For each suggested item, include:
  - Item ID
  - Relevance score (0.0 to 1.0)
  - Reason why this item would appeal to the target user (MUST include specific item content/attributes)

Be specific about how your preferences inform your suggestions and how they relate to the target user's query."""

        return system_prompt
    
    def _format_item_descriptions(self) -> str:
        """
        Format item descriptions from neighbor's history into readable string.
        
        Returns:
            str: Formatted item descriptions
        """
        if not self.neighbor_item_descriptions:
            return "No detailed item information available."
        
        lines = []
        for item_id, desc in list(self.neighbor_item_descriptions.items())[:15]:  # Limit to 15 items
            if desc and desc.strip():
                # Truncate very long descriptions
                if len(desc) > 300:
                    desc = desc[:300] + "..."
                lines.append(f"- Item #{item_id}: {desc}")
        
        if not lines:
            return "No detailed item information available."
        
        return "\n".join(lines)
    
    def reply(
        self, 
        instruction: str, 
        discussion_history: List[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Generate suggestions based on neighbor perspective.
        
        The agent:
        1. Considers the instruction from the orchestrator
        2. Reviews the discussion history from previous rounds
        3. Optionally calls retrieval tools to find relevant items
        4. Generates item suggestions with rationales based on preference alignment
        
        Args:
            instruction: Personalized instruction from the orchestrator
                        guiding what the agent should focus on
            discussion_history: Messages from previous discussion rounds,
                               each containing agent_id, content, and round info
            
        Returns:
            AgentResponse containing:
            - agent_id: This agent's identifier
            - suggestions: List of ItemSuggestion with item_id, score, reason
            - rationale: Overall reasoning for the suggestions
            - tool_calls: List of retrieval tool invocations made
            - raw_response: The raw LLM response text
        
        Validates: Requirements 3.4, 3.5, 3.6
        """
        from connacf.macf.data_models import ItemSuggestion, ToolCall
        
        tool_calls: List[ToolCall] = []
        
        # Build the conversation messages
        messages = []
        
        # Add system prompt
        system_prompt = self.get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add discussion history context if available
        if discussion_history:
            history_context = self._format_discussion_history(discussion_history)
            messages.append({
                "role": "user", 
                "content": f"Previous discussion context:\n{history_context}"
            })
        
        # Call retrieval tools to find relevant items for suggestions
        retrieved_items = self._retrieve_relevant_items(tool_calls)
        
        # Build the main instruction with retrieved context
        user_message = self._build_instruction_message(instruction, retrieved_items)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using LLM
        try:
            raw_response = self._call_llm(messages)
        except Exception as e:
            logger.warning(f"UserAgent {self.agent_id}: LLM call failed: {e}")
            # Return fallback response
            return AgentResponse(
                agent_id=self.agent_id,
                suggestions=[],
                rationale=f"Unable to generate response due to error: {e}",
                tool_calls=tool_calls,
                raw_response=""
            )
        
        # Parse the response to extract suggestions
        suggestions, rationale = self._parse_response(raw_response, retrieved_items)
        
        # Update message history
        self.message_history.append({
            "role": "assistant",
            "content": raw_response
        })
        
        logger.debug(
            f"UserAgent {self.agent_id}: Generated {len(suggestions)} suggestions"
        )
        
        return AgentResponse(
            agent_id=self.agent_id,
            suggestions=suggestions,
            rationale=rationale,
            tool_calls=tool_calls,
            raw_response=raw_response
        )
    
    def _format_preferences(self, preferences: Dict[str, Any]) -> str:
        """
        Format preference dictionary into readable string.
        
        Args:
            preferences: Dictionary of preference patterns
            
        Returns:
            str: Formatted preference string
        """
        if not preferences:
            return "No specific preferences available."
        
        lines = []
        for key, value in preferences.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_discussion_history(
        self, 
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """
        Format discussion history into readable context.
        
        Args:
            discussion_history: List of messages from previous rounds
            
        Returns:
            str: Formatted discussion history
        """
        if not discussion_history:
            return "No previous discussion."
        
        lines = []
        for msg in discussion_history:
            agent_id = msg.get('agent_id', 'unknown')
            content = msg.get('content', '')
            round_idx = msg.get('round_idx', '?')
            
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            lines.append(f"[Round {round_idx}] {agent_id}: {content}")
        
        return "\n".join(lines)
    
    def _retrieve_relevant_items(
        self, 
        tool_calls: List[ToolCall]
    ) -> List[int]:
        """
        Call retrieval tools to find relevant items for suggestions.
        
        Uses CF-based retrieval via retrieve_by_query with user_id to find
        items that the target user is likely to enjoy based on collaborative
        filtering signals.
        
        Args:
            tool_calls: List to append tool call records to
            
        Returns:
            List[int]: List of retrieved item IDs
        
        Validates: Requirements 3.5
        """
        from connacf.macf.data_models import ToolCall
        
        retrieved_items = []
        
        # Get target user ID for CF-based retrieval
        target_user_id = self.target_user_context.get('user_id')
        
        # Use CF-based retrieval (passing user_id for collaborative filtering)
        try:
            query_results = self.call_tool(
                'retrieve_by_query', 
                query=self.query, 
                k=15,
                user_id=target_user_id  # Enable CF-based retrieval
            )
            
            tool_calls.append(ToolCall(
                tool_name='retrieve_by_query',
                arguments={'query': self.query, 'k': 15, 'user_id': target_user_id},
                result=query_results
            ))
            
            retrieved_items.extend(query_results)
            
        except Exception as e:
            logger.warning(
                f"UserAgent {self.agent_id}: Error calling retrieve_by_query: {e}"
            )
        
        return retrieved_items
    
    def _build_instruction_message(
        self, 
        instruction: str, 
        retrieved_items: List[int]
    ) -> str:
        """
        Build the instruction message for the LLM.
        
        Args:
            instruction: Instruction from the orchestrator
            retrieved_items: Items retrieved via tools
            
        Returns:
            str: Complete instruction message
        """
        message_parts = [f"## Orchestrator Instruction\n{instruction}"]
        
        if retrieved_items:
            # Get item descriptions for retrieved items
            items_with_desc = []
            for item_id in retrieved_items[:15]:  # Limit to 15 items
                desc = ""
                if hasattr(self.toolkit, 'index_manager'):
                    desc = self.toolkit.index_manager.get_item_description(item_id)
                if desc and desc.strip():
                    # Truncate long descriptions
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    items_with_desc.append(f"  - Item #{item_id}: {desc}")
                else:
                    items_with_desc.append(f"  - Item #{item_id}")
            
            message_parts.append(
                f"\n## Retrieved Candidate Items\n"
                f"The following items were retrieved as potentially relevant:\n" +
                "\n".join(items_with_desc)
            )
        
        message_parts.append(
            "\n## Your Response\n"
            "Based on your perspective as a similar user, provide your item suggestions. "
            "For each suggestion:\n"
            "1. Quote the ACTUAL item title/description\n"
            "2. Explain how YOUR preferences (citing specific items you've enjoyed) inform this recommendation\n"
            "3. Connect the item's content to the target user's query\n\n"
            "Format each suggestion as:\n"
            "ITEM: <item_id>\n"
            "SCORE: <0.0-1.0>\n"
            "REASON: <explanation with specific item content and your preferences>\n"
        )
        
        return "\n".join(message_parts)
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM to generate a response.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            str: The LLM's response text
        """
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nAssistant:"
        
        # Call LLM
        response = self.llm.generate_response(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _parse_response(
        self, 
        raw_response: str, 
        retrieved_items: List[int]
    ) -> tuple:
        """
        Parse LLM response to extract suggestions and rationale.
        
        Args:
            raw_response: Raw text response from LLM
            retrieved_items: Items that were retrieved via tools
            
        Returns:
            tuple: (List[ItemSuggestion], str rationale)
        
        Validates: Requirements 3.6
        """
        from connacf.macf.data_models import ItemSuggestion
        import re
        
        suggestions = []
        rationale_parts = []
        
        # Try to parse structured format: ITEM: X, SCORE: Y, REASON: Z
        item_pattern = r'ITEM:\s*(\d+)'
        score_pattern = r'SCORE:\s*([\d.]+)'
        reason_pattern = r'REASON:\s*(.+?)(?=ITEM:|$)'
        
        # Find all item mentions
        item_matches = re.finditer(item_pattern, raw_response, re.IGNORECASE)
        
        for item_match in item_matches:
            try:
                item_id = int(item_match.group(1))
                
                # Find score after this item
                remaining_text = raw_response[item_match.end():]
                score_match = re.search(score_pattern, remaining_text, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0.5
                
                # Clamp score to valid range
                score = max(0.0, min(1.0, score))
                
                # Find reason after this item
                reason_match = re.search(reason_pattern, remaining_text, re.IGNORECASE | re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else "Suggested based on preference alignment."
                
                # Truncate long reasons
                if len(reason) > 500:
                    reason = reason[:500] + "..."
                
                suggestions.append(ItemSuggestion(
                    item_id=item_id,
                    score=score,
                    reason=reason
                ))
                
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse item suggestion: {e}")
                continue
        
        # If no structured suggestions found, try to extract from retrieved items
        if not suggestions and retrieved_items:
            logger.debug(
                f"UserAgent {self.agent_id}: No structured suggestions found, "
                f"using retrieved items as fallback"
            )
            for i, item_id in enumerate(retrieved_items[:5]):
                suggestions.append(ItemSuggestion(
                    item_id=item_id,
                    score=0.5 - (i * 0.05),  # Decreasing scores
                    reason=f"Retrieved item relevant to query: {self.query}"
                ))
        
        # Extract overall rationale from response
        rationale = self._extract_rationale(raw_response)
        
        return suggestions, rationale
    
    def _extract_rationale(self, raw_response: str) -> str:
        """
        Extract overall rationale from the response.
        
        Args:
            raw_response: Raw text response from LLM
            
        Returns:
            str: Extracted rationale (FULL, not truncated)
        """
        # Look for explicit rationale section
        import re
        
        rationale_patterns = [
            r'(?:Overall|Summary|Rationale):\s*(.+?)(?=ITEM:|$)',
            r'(?:Based on|Considering|Given).+?(?:I suggest|I recommend|my suggestions)',
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            if match:
                rationale = match.group(1) if match.lastindex else match.group(0)
                rationale = rationale.strip()
                # Return FULL rationale - no truncation
                return rationale
        
        # Default rationale based on agent perspective - include full query
        return (
            f"As User {self.neighbor_user_id} with similar preferences to the target user, "
            f"I've suggested items that align with both our tastes and the query: '{self.query}'"
        )


class ItemAgent(BaseMACFAgent):
    """
    Agent representing a history item.
    
    Reasons about why the target user previously interacted with this item
    and traces relevance paths to new candidates.
    
    The ItemAgent is instantiated for each query-relevant item from the target
    user's history. It uses its knowledge of the item's attributes and the
    context of the user's interaction to find similar items that might appeal
    to the target user.
    
    Attributes:
        item_id: The ID of the history item this agent represents
        item_attributes: Attributes and metadata of the item
        interaction_context: Context about why the user interacted with this item
        memory: Optional persistent memory from MACFMemoryStore
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
    """
    
    def __init__(
        self,
        item_id: int,
        item_attributes: Dict[str, Any],
        interaction_context: str,
        llm: BaseLLM,
        toolkit: MACFToolkit,
        target_user_context: Dict[str, Any],
        query: str,
        memory: Optional['AgentMemory'] = None,
        item_description: Optional[str] = None
    ):
        """
        Initialize an ItemAgent representing a history item.
        
        Args:
            item_id: The ID of the history item this agent represents
            item_attributes: Dictionary containing the item's attributes and metadata
                            (e.g., title, genre, description, etc.)
            interaction_context: Text describing why the user previously interacted
                                with this item (e.g., rating, purchase context)
            llm: LLM wrapper instance for generating responses
            toolkit: MACFToolkit instance for retrieval tool access
            target_user_context: Context about the target user including their
                                preferences, history summary, and other relevant info
            query: The natural language query expressing user intent
            memory: Optional AgentMemory for persistent memory across tasks
            item_description: The actual text description of this item (for semantic content)
        
        Validates: Requirements 4.1, 4.2, 4.3
        """
        # Build the profile dictionary with item-specific information
        profile = {
            'item_id': item_id,
            'attributes': item_attributes,
            'interaction_context': interaction_context
        }
        
        # Initialize base class with constructed profile
        super().__init__(
            agent_id=f"item_agent_{item_id}",
            profile=profile,
            llm=llm,
            toolkit=toolkit,
            target_user_context=target_user_context,
            query=query,
            memory=memory
        )
        
        # Store item-specific attributes for easy access
        self.item_id = item_id
        self.item_attributes = item_attributes
        self.interaction_context = interaction_context
        self.item_description = item_description or f"Item {item_id}"
        
        logger.debug(
            f"Initialized ItemAgent for history item {item_id}, memory={'enabled' if memory else 'disabled'}, desc_len={len(self.item_description)}"
        )
    
    def get_system_prompt(self) -> str:
        """
        Generate system prompt for item agent.
        
        The system prompt includes:
        - The item's ACTUAL DESCRIPTION (title, content, attributes)
        - Persistent memory context (if available)
        - Context about why the user interacted with this item
        - Context about the target user
        - The current query
        - Instructions to find related candidates WITH SPECIFIC CONTENT
        
        Returns:
            str: The system prompt for this item agent
        
        Validates: Requirements 4.4
        """
        # Extract target user information
        target_user_id = self.target_user_context.get('user_id', 'unknown')
        target_preferences = self.target_user_context.get('preferences', {})
        target_history_summary = self.target_user_context.get('history_summary', 'No history available')
        
        # Format item attributes for the prompt
        item_attrs_str = self._format_attributes(self.item_attributes)
        target_prefs_str = self._format_preferences(target_preferences)
        
        # Get persistent memory context
        memory_context = self.get_memory_context()
        memory_section = ""
        if memory_context:
            memory_section = f"""
## Your Memory (Learnings from Previous Interactions)
{memory_context}
"""
        
        system_prompt = f"""You are an Item Agent representing Item {self.item_id} in a multi-agent collaborative filtering system.

## Your Role
You represent an item that the target user has previously interacted with. Your job is to reason about why the user liked or engaged with you, and trace relevance paths to find new candidate items that share similar qualities.

## WHO YOU ARE - Your Full Description
{self.item_description}

## Your Attributes
{item_attrs_str}

## Interaction Context
Why the target user interacted with you:
{self.interaction_context}
{memory_section}
## Target User Context
You are helping to generate recommendations for User {target_user_id}.

Target user's preferences:
{target_prefs_str}

Target user's history summary:
{target_history_summary}

## Current Query
The target user is looking for: "{self.query}"

## Your Task
1. Analyze why the target user previously interacted with you (this item)
2. Identify your key attributes that made you appealing to the user
3. Use retrieval tools to find new candidate items that share similar attributes or qualities
4. Trace relevance paths: explain how new candidates relate to you and why they would appeal to the user
5. Suggest items with clear rationales explaining the connection from you to the new candidates
6. Learn from this interaction to improve future recommendations

## CRITICAL INSTRUCTIONS FOR YOUR RESPONSE
When making suggestions, you MUST:
- Quote YOUR OWN title/description when explaining why you were liked (e.g., "As '{self.item_description[:100]}...', I appeal to users who...")
- Reference SPECIFIC content characteristics (genres, artists, styles, themes)
- Quote the ACTUAL descriptions of items you recommend
- Trace clear relevance paths (e.g., "Since I am a jazz album, and Item #X is also jazz with similar instrumentation...")

DO NOT give generic responses like "similar to me" without citing specific content attributes.

## Response Format
Provide your suggestions in the following format:
- For each suggested item, include:
  - Item ID
  - Relevance score (0.0 to 1.0)
  - Reason explaining the relevance path from this history item to the candidate (MUST include specific content)

Be specific about which of your attributes connect to the suggested items and how they relate to the target user's query."""

        return system_prompt
    
    def reply(
        self, 
        instruction: str, 
        discussion_history: List[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Generate suggestions by tracing relevance from history item.
        
        The agent:
        1. Considers the instruction from the orchestrator
        2. Reviews the discussion history from previous rounds
        3. Calls RetrieveByItem to find items similar to itself
        4. Optionally calls RetrieveByQuery for additional candidates
        5. Generates item suggestions with rationales explaining relevance paths
        
        Args:
            instruction: Personalized instruction from the orchestrator
                        guiding what the agent should focus on
            discussion_history: Messages from previous discussion rounds,
                               each containing agent_id, content, and round info
            
        Returns:
            AgentResponse containing:
            - agent_id: This agent's identifier
            - suggestions: List of ItemSuggestion with item_id, score, reason
            - rationale: Overall reasoning for the suggestions
            - tool_calls: List of retrieval tool invocations made
            - raw_response: The raw LLM response text
        
        Validates: Requirements 4.4, 4.5, 4.6
        """
        from connacf.macf.data_models import ItemSuggestion, ToolCall
        
        tool_calls: List[ToolCall] = []
        
        # Build the conversation messages
        messages = []
        
        # Add system prompt
        system_prompt = self.get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Add discussion history context if available
        if discussion_history:
            history_context = self._format_discussion_history(discussion_history)
            messages.append({
                "role": "user", 
                "content": f"Previous discussion context:\n{history_context}"
            })
        
        # Call retrieval tools to find relevant items for suggestions
        retrieved_items = self._retrieve_relevant_items(tool_calls)
        
        # Build the main instruction with retrieved context
        user_message = self._build_instruction_message(instruction, retrieved_items)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using LLM
        try:
            raw_response = self._call_llm(messages)
        except Exception as e:
            logger.warning(f"ItemAgent {self.agent_id}: LLM call failed: {e}")
            # Return fallback response
            return AgentResponse(
                agent_id=self.agent_id,
                suggestions=[],
                rationale=f"Unable to generate response due to error: {e}",
                tool_calls=tool_calls,
                raw_response=""
            )
        
        # Parse the response to extract suggestions
        suggestions, rationale = self._parse_response(raw_response, retrieved_items)
        
        # Update message history
        self.message_history.append({
            "role": "assistant",
            "content": raw_response
        })
        
        logger.debug(
            f"ItemAgent {self.agent_id}: Generated {len(suggestions)} suggestions"
        )
        
        return AgentResponse(
            agent_id=self.agent_id,
            suggestions=suggestions,
            rationale=rationale,
            tool_calls=tool_calls,
            raw_response=raw_response
        )
    
    def _format_attributes(self, attributes: Dict[str, Any]) -> str:
        """
        Format item attributes dictionary into readable string.
        
        Args:
            attributes: Dictionary of item attributes
            
        Returns:
            str: Formatted attributes string
        """
        if not attributes:
            return "No specific attributes available."
        
        lines = []
        for key, value in attributes.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_preferences(self, preferences: Dict[str, Any]) -> str:
        """
        Format preference dictionary into readable string.
        
        Args:
            preferences: Dictionary of preference patterns
            
        Returns:
            str: Formatted preference string
        """
        if not preferences:
            return "No specific preferences available."
        
        lines = []
        for key, value in preferences.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_discussion_history(
        self, 
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """
        Format discussion history into readable context.
        
        Args:
            discussion_history: List of messages from previous rounds
            
        Returns:
            str: Formatted discussion history
        """
        if not discussion_history:
            return "No previous discussion."
        
        lines = []
        for msg in discussion_history:
            agent_id = msg.get('agent_id', 'unknown')
            content = msg.get('content', '')
            round_idx = msg.get('round_idx', '?')
            
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            lines.append(f"[Round {round_idx}] {agent_id}: {content}")
        
        return "\n".join(lines)
    
    def _retrieve_relevant_items(
        self, 
        tool_calls: List[ToolCall]
    ) -> List[int]:
        """
        Call retrieval tools to find relevant items for suggestions.
        
        Uses CF-based retrieval:
        1. retrieve_by_item: Find items similar to this history item (CF-based)
        2. retrieve_by_query with user_id: Find items for the target user (CF-based)
        
        Args:
            tool_calls: List to append tool call records to
            
        Returns:
            List[int]: List of retrieved item IDs
        
        Validates: Requirements 4.5
        """
        from connacf.macf.data_models import ToolCall
        
        retrieved_items = []
        
        # Get target user ID for CF-based retrieval
        target_user_id = self.target_user_context.get('user_id')
        
        # Primary: Use retrieve_by_item to find items similar to this history item (CF-based)
        try:
            item_results = self.call_tool('retrieve_by_item', item_id=self.item_id, k=15)
            
            tool_calls.append(ToolCall(
                tool_name='retrieve_by_item',
                arguments={'item_id': self.item_id, 'k': 15},
                result=item_results
            ))
            
            retrieved_items.extend(item_results)
            
        except Exception as e:
            logger.warning(
                f"ItemAgent {self.agent_id}: Error calling retrieve_by_item: {e}"
            )
        
        # Secondary: Use CF-based retrieval for the target user
        try:
            query_results = self.call_tool(
                'retrieve_by_query', 
                query=self.query, 
                k=10,
                user_id=target_user_id  # Enable CF-based retrieval
            )
            
            tool_calls.append(ToolCall(
                tool_name='retrieve_by_query',
                arguments={'query': self.query, 'k': 10, 'user_id': target_user_id},
                result=query_results
            ))
            
            # Add query results that aren't already in the list
            for item_id in query_results:
                if item_id not in retrieved_items:
                    retrieved_items.append(item_id)
            
        except Exception as e:
            logger.warning(
                f"ItemAgent {self.agent_id}: Error calling retrieve_by_query: {e}"
            )
        
        return retrieved_items
    
    def _build_instruction_message(
        self, 
        instruction: str, 
        retrieved_items: List[int]
    ) -> str:
        """
        Build the instruction message for the LLM.
        
        Args:
            instruction: Instruction from the orchestrator
            retrieved_items: Items retrieved via tools
            
        Returns:
            str: Complete instruction message
        """
        message_parts = [f"## Orchestrator Instruction\n{instruction}"]
        
        if retrieved_items:
            # Get item descriptions for retrieved items
            items_with_desc = []
            for item_id in retrieved_items[:15]:  # Limit to 15 items
                desc = ""
                if hasattr(self.toolkit, 'index_manager'):
                    desc = self.toolkit.index_manager.get_item_description(item_id)
                if desc and desc.strip():
                    # Truncate long descriptions
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    items_with_desc.append(f"  - Item #{item_id}: {desc}")
                else:
                    items_with_desc.append(f"  - Item #{item_id}")
            
            message_parts.append(
                f"\n## Retrieved Candidate Items\n"
                f"Items similar to you (Item {self.item_id}: {self.item_description[:100]}...) and matching the query:\n" +
                "\n".join(items_with_desc)
            )
        
        message_parts.append(
            "\n## Your Response\n"
            "Based on your perspective as a history item the user previously engaged with, "
            "provide your item suggestions. For each suggestion:\n"
            "1. Quote YOUR description and explain why the user liked you\n"
            "2. Quote the CANDIDATE item's description\n"
            "3. Trace the relevance path with specific content attributes\n\n"
            "Format each suggestion as:\n"
            "ITEM: <item_id>\n"
            "SCORE: <0.0-1.0>\n"
            "REASON: <explanation with specific item content from both you and the candidate>\n"
        )
        
        return "\n".join(message_parts)
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM to generate a response.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            str: The LLM's response text
        """
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nAssistant:"
        
        # Call LLM
        response = self.llm.generate_response(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _parse_response(
        self, 
        raw_response: str, 
        retrieved_items: List[int]
    ) -> tuple:
        """
        Parse LLM response to extract suggestions and rationale.
        
        Args:
            raw_response: Raw text response from LLM
            retrieved_items: Items that were retrieved via tools
            
        Returns:
            tuple: (List[ItemSuggestion], str rationale)
        
        Validates: Requirements 4.6
        """
        from connacf.macf.data_models import ItemSuggestion
        import re
        
        suggestions = []
        
        # Try to parse structured format: ITEM: X, SCORE: Y, REASON: Z
        item_pattern = r'ITEM:\s*(\d+)'
        score_pattern = r'SCORE:\s*([\d.]+)'
        reason_pattern = r'REASON:\s*(.+?)(?=ITEM:|$)'
        
        # Find all item mentions
        item_matches = re.finditer(item_pattern, raw_response, re.IGNORECASE)
        
        for item_match in item_matches:
            try:
                item_id = int(item_match.group(1))
                
                # Find score after this item
                remaining_text = raw_response[item_match.end():]
                score_match = re.search(score_pattern, remaining_text, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0.5
                
                # Clamp score to valid range
                score = max(0.0, min(1.0, score))
                
                # Find reason after this item
                reason_match = re.search(reason_pattern, remaining_text, re.IGNORECASE | re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else f"Similar to Item {self.item_id} based on shared attributes."
                
                # Truncate long reasons
                if len(reason) > 500:
                    reason = reason[:500] + "..."
                
                suggestions.append(ItemSuggestion(
                    item_id=item_id,
                    score=score,
                    reason=reason
                ))
                
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse item suggestion: {e}")
                continue
        
        # If no structured suggestions found, try to extract from retrieved items
        if not suggestions and retrieved_items:
            logger.debug(
                f"ItemAgent {self.agent_id}: No structured suggestions found, "
                f"using retrieved items as fallback"
            )
            for i, item_id in enumerate(retrieved_items[:5]):
                suggestions.append(ItemSuggestion(
                    item_id=item_id,
                    score=0.5 - (i * 0.05),  # Decreasing scores
                    reason=f"Similar to Item {self.item_id} based on embedding similarity"
                ))
        
        # Extract overall rationale from response
        rationale = self._extract_rationale(raw_response)
        
        return suggestions, rationale
    
    def _extract_rationale(self, raw_response: str) -> str:
        """
        Extract overall rationale from the response.
        
        Args:
            raw_response: Raw text response from LLM
            
        Returns:
            str: Extracted rationale (FULL, not truncated)
        """
        import re
        
        rationale_patterns = [
            r'(?:Overall|Summary|Rationale):\s*(.+?)(?=ITEM:|$)',
            r'(?:Based on|Considering|Given).+?(?:I suggest|I recommend|my suggestions)',
        ]
        
        for pattern in rationale_patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            if match:
                rationale = match.group(1) if match.lastindex else match.group(0)
                rationale = rationale.strip()
                # Return FULL rationale - no truncation
                return rationale
        
        # Default rationale based on agent perspective - include full query
        return (
            f"As Item {self.item_id} from the user's history, "
            f"I've traced relevance paths to find similar items that match the query: '{self.query}'"
        )
