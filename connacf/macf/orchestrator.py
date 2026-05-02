"""
MACFOrchestrator - Central coordinator for multi-round agent discussions.

This module implements the orchestrator agent that coordinates MACF discussions,
managing agent recruitment, personalized instruction generation, multi-round
discussion coordination, and response aggregation.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 3.1, 4.1, 11.2, 11.4, 11.5, 11.6
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from connacf.agentverse.llms.base import BaseLLM
from connacf.macf.agents import BaseMACFAgent, UserAgent, ItemAgent
from connacf.macf.config import MACFConfig
from connacf.macf.conversation_logger import MACFConversationLogger
from connacf.macf.data_models import (
    AgentResponse,
    DiscussionState,
    ItemSuggestion,
    RankedList,
)
from connacf.macf.toolkit import MACFToolkit
from connacf.macf.error_handling import handle_empty_retrieval
from connacf.macf.memory_store import MACFMemoryStore
from connacf.macf.metrics_collector import MACFMetricsCollector
from connacf.macf.backward_pass import MACFBackwardPass

# Import AttackHooks for type checking to avoid circular imports
if TYPE_CHECKING:
    from connacf.macf.attack_hooks import AttackHooks

logger = logging.getLogger(__name__)


class MACFOrchestrator:
    """
    Orchestrator agent that coordinates MACF discussions.
    
    Manages:
    - Agent recruitment via retrieval tools
    - Personalized instruction generation
    - Multi-round discussion coordination
    - Response aggregation and ranking
    - Persistent memory for agents (optional)
    
    Attributes:
        llm: LLM wrapper for generating responses and instructions
        toolkit: MACFToolkit for retrieval operations
        config: MACFConfig with system parameters
        attack_hooks: Optional hooks for attack framework integration
        memory_store: Optional MACFMemoryStore for persistent agent memory
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        toolkit: MACFToolkit,
        config: MACFConfig,
        attack_hooks: Optional['AttackHooks'] = None,
        conversation_logger: Optional[MACFConversationLogger] = None,
        memory_store: Optional[MACFMemoryStore] = None,
        metrics_collector: Optional[MACFMetricsCollector] = None,
        pre_discussion_callback: Optional[callable] = None,
        wandb_logger: Optional[Any] = None
    ):
        """
        Initialize the MACFOrchestrator.
        
        Args:
            llm: LLM wrapper instance for generating responses and instructions.
                 Should be a BaseLLM subclass from connacf/agentverse/llms/.
            toolkit: MACFToolkit instance providing retrieval tools for
                    agent recruitment and candidate retrieval.
            config: MACFConfig instance with system parameters including
                   neighbor_count, history_item_count, max_rounds, etc.
            attack_hooks: Optional AttackHooks instance for attack framework
                         integration. When provided, allows attackers to:
                         - Intercept and modify agent messages via intercept_message()
                         - Inject adversarial agents via get_adversarial_agents()
                         - Log interactions for analysis via log_interaction()
            conversation_logger: Optional MACFConversationLogger for logging
                                agent discussions in readable format.
            memory_store: Optional MACFMemoryStore for persistent agent memory
                         across tasks. When provided, agents will:
                         - Load their previous memories at initialization
                         - Update memories after each discussion round
                         - Learn and evolve over time like ConnaCF agents
            metrics_collector: Optional MACFMetricsCollector for tracking
                              per-turn metrics following the 6-class structure.
            pre_discussion_callback: Optional callback function called after
                                    agent recruitment but before discussion.
                                    Signature: callback(user_agents, item_agents, task_id)
                                    Used for dynamic attack selection.
            wandb_logger: Optional MACFWandBLogger for logging metrics to wandb.
        
        Example:
            >>> from connacf.macf import MACFConfig, MACFToolkit, MACFOrchestrator
            >>> from connacf.macf.attack_hooks import AttackHooks
            >>> from connacf.macf.conversation_logger import MACFConversationLogger
            >>> from connacf.macf.memory_store import MACFMemoryStore
            >>> from connacf.macf.metrics_collector import MACFMetricsCollector
            >>> config = MACFConfig.from_yaml('props/MACF.yaml')
            >>> attack_hooks = AttackHooks()  # Optional, for attack mode
            >>> conv_logger = MACFConversationLogger(output_dir='output', print_to_stdout=True)
            >>> memory_store = MACFMemoryStore(persist_path='output/agent_memories.json')
            >>> metrics = MACFMetricsCollector(output_dir='output', experiment_name='macf_exp')
            >>> orchestrator = MACFOrchestrator(llm, toolkit, config, attack_hooks, conv_logger, memory_store, metrics)
        
        Requirements: 5.1, 5.2, 11.2, 11.4, 11.5, 11.6
        """
        self.llm = llm
        self.toolkit = toolkit
        self.config = config
        self.attack_hooks = attack_hooks
        self.conversation_logger = conversation_logger
        self.memory_store = memory_store
        self.metrics_collector = metrics_collector
        self.pre_discussion_callback = pre_discussion_callback
        self.wandb_logger = wandb_logger
        
        # Communication graph tracker (optional, created if metrics_collector has output_dir)
        self.communication_graph = None
        if metrics_collector and hasattr(metrics_collector, 'output_dir'):
            from connacf.macf.communication_graph import CommunicationGraphTracker
            self.communication_graph = CommunicationGraphTracker(
                output_dir=metrics_collector.output_dir,
                experiment_name=metrics_collector.experiment_name if hasattr(metrics_collector, 'experiment_name') else "macf"
            )
        
        # Backward pass handler for ConnaCF-compatible learning
        # This enables profile updates after each task based on feedback
        self.backward_pass = None
        if memory_store:
            self.backward_pass = MACFBackwardPass(
                llm=llm,
                memory_store=memory_store,
                api_batch=getattr(config, 'api_batch', 10),
                update_items=True,
                index_manager=toolkit.index_manager if toolkit else None,
            )
            logger.info("Backward pass enabled for ConnaCF-compatible learning")
        
        # Task counter for tracking
        self._task_counter = 0
        
        logger.info(
            f"MACFOrchestrator initialized with config: "
            f"neighbor_count={config.neighbor_count}, "
            f"history_item_count={config.history_item_count}, "
            f"max_rounds={config.max_rounds}, "
            f"top_k_recommendation={config.top_k_recommendation}"
            + (f", attack_hooks=enabled" if attack_hooks else "")
            + (f", conversation_logger=enabled" if conversation_logger else "")
            + (f", memory_store=enabled" if memory_store else "")
            + (f", metrics_collector=enabled" if metrics_collector else "")
            + (f", pre_discussion_callback=enabled" if pre_discussion_callback else "")
            + (f", backward_pass=enabled" if self.backward_pass else "")
        )
    
    def recruit_agents(
        self,
        target_user_id: int,
        query: str
    ) -> Tuple[List[UserAgent], List[ItemAgent]]:
        """
        Recruit user and item agents for discussion.
        
        Uses GetSimilarUsers and GetRelevantItems tools to identify:
        1. Similar users (neighbors) to instantiate as UserAgents
        2. Query-relevant history items to instantiate as ItemAgents
        
        Each recruited agent is initialized with:
        - Profile information (preferences/attributes)
        - Target user context
        - The current query
        
        Args:
            target_user_id: The ID of the target user for recommendations
            query: Natural language query expressing user intent
            
        Returns:
            Tuple[List[UserAgent], List[ItemAgent]]: A tuple containing:
                - List of UserAgent instances representing similar users
                - List of ItemAgent instances representing relevant history items
                
            Both lists may be empty if retrieval returns no results.
        
        Example:
            >>> user_agents, item_agents = orchestrator.recruit_agents(
            ...     target_user_id=123,
            ...     query="action movies with good plot"
            ... )
            >>> print(f"Recruited {len(user_agents)} user agents")
            >>> print(f"Recruited {len(item_agents)} item agents")
        
        Requirements: 5.1, 5.2, 3.1, 4.1
        """
        logger.info(
            f"Recruiting agents for user {target_user_id} with query: '{query[:50]}...'"
        )
        
        user_agents: List[UserAgent] = []
        item_agents: List[ItemAgent] = []
        
        # Build target user context for agent initialization
        target_user_context = self._build_target_user_context(target_user_id)
        
        # ==================== Recruit User Agents ====================
        # Use GetSimilarUsers tool to find similar users (neighbors)
        similar_user_ids = self.toolkit.get_similar_users(
            target_user_id=target_user_id,
            n=self.config.neighbor_count
        )
        
        if not similar_user_ids:
            logger.warning(
                f"No similar users found for user {target_user_id}, "
                f"proceeding with item agents only"
            )
            handle_empty_retrieval('similar_users', target_user_id)
        else:
            logger.debug(f"Found {len(similar_user_ids)} similar users: {similar_user_ids}")
            
            # Instantiate UserAgent for each similar user
            for neighbor_user_id in similar_user_ids:
                try:
                    user_agent = self._create_user_agent(
                        neighbor_user_id=neighbor_user_id,
                        target_user_context=target_user_context,
                        query=query
                    )
                    user_agents.append(user_agent)
                except Exception as e:
                    logger.warning(
                        f"Failed to create UserAgent for neighbor {neighbor_user_id}: {e}"
                    )
        
        # ==================== Recruit Item Agents ====================
        # Use GetRelevantItems tool to find query-relevant history items
        relevant_item_ids = self.toolkit.get_relevant_items(
            target_user_id=target_user_id,
            query=query,
            n=self.config.history_item_count
        )
        
        if not relevant_item_ids:
            logger.warning(
                f"No relevant history items found for user {target_user_id}, "
                f"proceeding with user agents only"
            )
            handle_empty_retrieval('relevant_items', target_user_id, query)
        else:
            logger.debug(f"Found {len(relevant_item_ids)} relevant items: {relevant_item_ids}")
            
            # Instantiate ItemAgent for each relevant history item
            for item_id in relevant_item_ids:
                try:
                    item_agent = self._create_item_agent(
                        item_id=item_id,
                        target_user_id=target_user_id,
                        target_user_context=target_user_context,
                        query=query
                    )
                    item_agents.append(item_agent)
                except Exception as e:
                    logger.warning(
                        f"Failed to create ItemAgent for item {item_id}: {e}"
                    )
        
        # ==================== Add Adversarial Agents (if attack mode) ====================
        # When attack_hooks is set, inject any adversarial agents into the discussion
        # This allows attackers to add malicious agents that participate alongside
        # legitimate agents, enabling robustness testing of the MACF system.
        # Requirements: 11.6
        if self.attack_hooks is not None:
            try:
                adversarial_agents = self.attack_hooks.get_adversarial_agents()
                for agent in adversarial_agents:
                    if isinstance(agent, UserAgent):
                        user_agents.append(agent)
                        logger.debug(f"Injected adversarial UserAgent: {agent.agent_id}")
                    elif isinstance(agent, ItemAgent):
                        item_agents.append(agent)
                        logger.debug(f"Injected adversarial ItemAgent: {agent.agent_id}")
                    else:
                        # Support custom adversarial agent types
                        logger.warning(
                            f"Unknown adversarial agent type: {type(agent).__name__}, "
                            f"adding to user_agents by default"
                        )
                        user_agents.append(agent)
                
                if adversarial_agents:
                    logger.info(
                        f"Injected {len(adversarial_agents)} adversarial agents "
                        f"from attack_hooks"
                    )
            except Exception as e:
                # Continue execution even if adversarial agent injection fails
                # Requirements: 11.5
                logger.warning(f"Failed to get adversarial agents: {e}")
        
        logger.info(
            f"Agent recruitment complete: "
            f"{len(user_agents)} user agents, {len(item_agents)} item agents"
        )
        
        return user_agents, item_agents
    
    def _build_target_user_context(self, target_user_id: int) -> Dict[str, Any]:
        """
        Build context dictionary for the target user.
        
        Gathers information about the target user including their
        preferences, interaction history summary with ACTUAL ITEM DESCRIPTIONS,
        and other relevant context for agent initialization.
        
        Args:
            target_user_id: The ID of the target user
            
        Returns:
            Dict[str, Any]: Context dictionary containing:
                - user_id: The target user's ID
                - preferences: Inferred preference patterns
                - history_summary: Summary of interaction history with item names
                - history_items: List of item IDs in history
                - item_descriptions: Dict of item_id -> description
        """
        # Get user's interaction history
        user_history = self.toolkit.index_manager.get_user_history(target_user_id)
        
        # Get ACTUAL item descriptions for items in user's history
        item_descriptions = {}
        for item_id in user_history[:20]:  # Limit to 20 items
            desc = self.toolkit.index_manager.get_item_description(item_id)
            if desc and desc.strip() and desc != '[PAD]':
                item_descriptions[item_id] = desc
        
        # Build history summary WITH actual item names
        if user_history:
            # Include some actual item descriptions in the summary
            sample_items = []
            for item_id in user_history[:5]:
                desc = item_descriptions.get(item_id, f"Item {item_id}")
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                sample_items.append(f"'{desc}'")
            
            history_summary = (
                f"User {target_user_id} has interacted with {len(user_history)} items. "
                f"Recent items include: {', '.join(sample_items)}"
            )
        else:
            history_summary = f"User {target_user_id} has no recorded interaction history."
        
        # Build preferences (can be enhanced with more sophisticated analysis)
        preferences = self._infer_user_preferences(target_user_id, user_history)
        
        return {
            'user_id': target_user_id,
            'preferences': preferences,
            'history_summary': history_summary,
            'history_items': user_history,
            'item_descriptions': item_descriptions
        }
    
    def _infer_user_preferences(
        self,
        user_id: int,
        history: List[int]
    ) -> Dict[str, Any]:
        """
        Infer user preferences from interaction history.
        
        This is a basic implementation that can be enhanced with
        more sophisticated preference modeling.
        
        Args:
            user_id: The user's ID
            history: List of item IDs the user has interacted with
            
        Returns:
            Dict[str, Any]: Inferred preference patterns
        """
        if not history:
            return {'interaction_count': 0}
        
        return {
            'interaction_count': len(history),
            'recent_items': history[:5],
            'history_size': len(history)
        }
    
    def _generate_semantic_user_profile(
        self,
        user_id: int,
        item_descriptions: Dict[int, str],
        memory=None
    ) -> str:
        """
        Generate a rich semantic profile for a user using LLM.
        
        Takes the user's item history and uses LLM world knowledge to infer
        personality traits, movie preferences, genres, moods, and listening habits.
        Also populates the memory's preferences and traits lists.
        
        Args:
            user_id: The user's ID
            item_descriptions: Dict of item_id -> description for user's history
            memory: Optional AgentMemory to populate preferences/traits
            
        Returns:
            str: Rich semantic profile with preferences and traits
        """
        if not item_descriptions:
            return f"User #{user_id} with no recorded preferences."
        
        # Build context from item descriptions
        sample_items = list(item_descriptions.values())[:8]
        items_text = "\n- ".join(desc[:120] for desc in sample_items)
        
        prompt = f"""Analyze this user's movie listening history and generate a detailed semantic profile.

User's listening history:
- {items_text}

Based on these items, infer and provide in JSON format:

{{
  "profile": "A 2-3 sentence personality description of this listener (e.g., 'Nostalgic movie enthusiast with eclectic taste spanning classic rock to R&B. Drawn to powerful vocals, anthemic choruses, and timeless hits from the 70s-90s. Values both artistic depth and mainstream appeal.')",
  "genres": ["list", "of", "3-5", "primary", "genres"],
  "moods": ["list", "of", "2-3", "mood", "preferences", "like", "energetic", "melancholic", "uplifting"],
  "eras": ["list", "of", "preferred", "eras", "like", "80s", "90s", "classic"],
  "traits": ["list", "of", "2-3", "personality", "traits", "like", "nostalgic", "eclectic", "mainstream"],
  "listening_style": "brief description of how they consume movie (e.g., 'Greatest hits collector who prefers compilations over deep album cuts')"
}}

Use your knowledge of these artists/albums to infer accurate genres and characteristics. Be specific and grounded in the actual items shown."""

        try:
            response = self.llm.generate_response(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            import json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                
                profile_text = data.get('profile', '')
                genres = data.get('genres', [])
                moods = data.get('moods', [])
                eras = data.get('eras', [])
                traits = data.get('traits', [])
                listening_style = data.get('listening_style', '')
                
                # Populate memory preferences and traits if available
                if memory:
                    for genre in genres[:5]:
                        memory.add_preference(f"{genre} movie")
                    for mood in moods[:3]:
                        memory.add_preference(f"{mood} movie")
                    for era in eras[:3]:
                        memory.add_preference(f"movie from the {era}")
                    for trait in traits[:3]:
                        memory.add_trait(trait)
                
                # Build rich profile string
                basic_list = f"User who enjoys items like: {'; '.join(d[:60] for d in sample_items[:3])}"
                
                preferences_str = ""
                if genres or moods:
                    pref_items = genres[:3] + moods[:2]
                    preferences_str = f"\nPreferences: {', '.join(pref_items)}"
                
                traits_str = ""
                if traits:
                    traits_str = f"\nTraits: {', '.join(traits[:3])}"
                
                style_str = ""
                if listening_style:
                    style_str = f"\nListening style: {listening_style}"
                
                return f"{profile_text}\n{basic_list}{preferences_str}{traits_str}{style_str}"
            else:
                raise ValueError("Could not parse JSON from LLM response")
            
        except Exception as e:
            logger.warning(f"Failed to generate semantic profile for user {user_id}: {e}")
            # Fall back to basic template
            sample_items = list(item_descriptions.values())[:3]
            return f"User who enjoys items like: {'; '.join(d[:80] for d in sample_items)}"
    
    def _generate_semantic_item_profile(
        self,
        item_id: int,
        item_description: str,
        memory=None
    ) -> str:
        """
        Generate a rich semantic profile for an item using LLM.
        
        Takes the item description and uses LLM world knowledge to generate
        a richer profile with genre, mood, era, and appeal characteristics.
        Also populates the memory's preferences list with item characteristics.
        
        Args:
            item_id: The item's ID
            item_description: The item's description/title
            memory: Optional AgentMemory to populate characteristics
            
        Returns:
            str: Rich semantic profile for the item
        """
        if not item_description or item_description == '[PAD]':
            return f"Item #{item_id}"
        
        prompt = f"""Analyze this movie item and generate a detailed semantic profile.

Item: {item_description[:250]}

Based on this item, provide in JSON format:

{{
  "profile": "A 2-3 sentence description of this release (e.g., 'Iconic pop album that defined the 80s sound with its blend of R&B, rock, and dance movie. Features groundbreaking production and unforgettable hooks. A cultural touchstone that appeals to multiple generations.')",
  "genre": "primary genre",
  "sub_genres": ["list", "of", "1-2", "sub-genres"],
  "era": "release era (e.g., '80s', '90s', 'modern')",
  "mood": "primary mood (e.g., 'energetic', 'melancholic', 'uplifting')",
  "appeal": ["list", "of", "2-3", "audience", "traits", "like", "nostalgic fans", "casual listeners"],
  "characteristics": ["list", "of", "2-3", "sonic", "characteristics", "like", "powerful vocals", "guitar-driven"]
}}

Use your knowledge of this artist/album to provide accurate information. Be specific and factual."""

        try:
            response = self.llm.generate_response(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            import json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                
                profile_text = data.get('profile', '')
                genre = data.get('genre', '')
                sub_genres = data.get('sub_genres', [])
                era = data.get('era', '')
                mood = data.get('mood', '')
                appeal = data.get('appeal', [])
                characteristics = data.get('characteristics', [])
                
                # Populate memory if available
                if memory:
                    if genre:
                        memory.add_preference(f"{genre} movie")
                    for sg in sub_genres[:2]:
                        memory.add_preference(sg)
                    if mood:
                        memory.add_preference(f"{mood} sound")
                    for char in characteristics[:2]:
                        memory.add_trait(char)
                
                # Build rich profile string
                genre_str = f"\nGenre: {genre}" if genre else ""
                if sub_genres:
                    genre_str += f" ({', '.join(sub_genres[:2])})"
                
                era_mood_str = ""
                if era or mood:
                    parts = []
                    if era:
                        parts.append(f"Era: {era}")
                    if mood:
                        parts.append(f"Mood: {mood}")
                    era_mood_str = "\n" + " | ".join(parts)
                
                appeal_str = ""
                if appeal:
                    appeal_str = f"\nAppeals to: {', '.join(appeal[:3])}"
                
                char_str = ""
                if characteristics:
                    char_str = f"\nCharacteristics: {', '.join(characteristics[:3])}"
                
                return f"{profile_text}{genre_str}{era_mood_str}{appeal_str}{char_str}\nItem: {item_description[:100]}"
            else:
                raise ValueError("Could not parse JSON from LLM response")
            
        except Exception as e:
            logger.warning(f"Failed to generate semantic profile for item {item_id}: {e}")
            return f"Item #{item_id}: {item_description}"
    
    def _create_user_agent(
        self,
        neighbor_user_id: int,
        target_user_context: Dict[str, Any],
        query: str
    ) -> UserAgent:
        """
        Create a UserAgent for a similar user (neighbor).
        
        Gathers the neighbor's profile information and interaction
        history to initialize the agent, including ACTUAL ITEM DESCRIPTIONS.
        
        Args:
            neighbor_user_id: The ID of the neighbor user
            target_user_context: Context about the target user
            query: The current query
            
        Returns:
            UserAgent: Initialized user agent
            
        Requirements: 3.1, 3.2, 3.3
        """
        # Get neighbor's interaction history
        neighbor_history = self.toolkit.index_manager.get_user_history(neighbor_user_id)
        
        # Build neighbor profile
        neighbor_profile = self._infer_user_preferences(neighbor_user_id, neighbor_history)
        
        # Get ACTUAL item descriptions for items in neighbor's history
        neighbor_item_descriptions = {}
        for item_id in neighbor_history[:20]:  # Limit to 20 items for context size
            desc = self.toolkit.index_manager.get_item_description(item_id)
            if desc and desc.strip() and desc != '[PAD]':
                neighbor_item_descriptions[item_id] = desc
        
        # Build history summary for the neighbor WITH item names
        if neighbor_history:
            # Include some actual item descriptions in the summary
            sample_items = []
            for item_id in neighbor_history[:5]:
                desc = neighbor_item_descriptions.get(item_id, f"Item {item_id}")
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                sample_items.append(f"'{desc}'")
            
            neighbor_history_summary = (
                f"This user has interacted with {len(neighbor_history)} items. "
                f"Their recent interactions include: {', '.join(sample_items)}"
            )
        else:
            neighbor_history_summary = "This user has no recorded interaction history."
        
        # Get persistent memory if memory store is available
        memory = None
        if self.memory_store:
            memory = self.memory_store.get_user_memory(neighbor_user_id)
            
            # Initialize profile if not set (semantic memory)
            if not memory.profile:
                # Generate rich semantic profile using LLM
                # Also populates memory.preferences and memory.traits
                if neighbor_item_descriptions:
                    profile_desc = self._generate_semantic_user_profile(
                        neighbor_user_id, 
                        neighbor_item_descriptions,
                        memory=memory
                    )
                    memory.set_profile(profile_desc)
                else:
                    memory.set_profile(f"User #{neighbor_user_id} with limited history.")
        
        # Create and return the UserAgent
        return UserAgent(
            neighbor_user_id=neighbor_user_id,
            neighbor_profile=neighbor_profile,
            neighbor_history_summary=neighbor_history_summary,
            llm=self.llm,
            toolkit=self.toolkit,
            target_user_context=target_user_context,
            query=query,
            memory=memory,
            neighbor_item_descriptions=neighbor_item_descriptions
        )
    
    def _create_item_agent(
        self,
        item_id: int,
        target_user_id: int,
        target_user_context: Dict[str, Any],
        query: str
    ) -> ItemAgent:
        """
        Create an ItemAgent for a query-relevant history item.
        
        Gathers the item's attributes, ACTUAL DESCRIPTION, and interaction context
        to initialize the agent.
        
        Args:
            item_id: The ID of the history item
            target_user_id: The ID of the target user
            target_user_context: Context about the target user
            query: The current query
            
        Returns:
            ItemAgent: Initialized item agent
            
        Requirements: 4.1, 4.2, 4.3
        """
        # Get ACTUAL item description from index manager
        item_description = self.toolkit.index_manager.get_item_description(item_id)
        if not item_description or item_description == '[PAD]':
            item_description = f"Item {item_id}"
        
        # Get item attributes (enhanced with description)
        item_attributes = self._get_item_attributes(item_id)
        item_attributes['description'] = item_description
        
        # Build interaction context with actual item content
        interaction_context = (
            f"User {target_user_id} previously interacted with this item: '{item_description[:150]}...'. "
            f"This item is relevant to the current query: '{query[:100]}'"
        )
        
        # Get persistent memory if memory store is available
        memory = None
        if self.memory_store:
            memory = self.memory_store.get_item_memory(item_id)
            
            # Initialize profile if not set (semantic memory)
            if not memory.profile:
                # Generate rich semantic profile using LLM
                # Also populates memory.preferences and memory.traits
                profile = self._generate_semantic_item_profile(
                    item_id, 
                    item_description,
                    memory=memory
                )
                memory.set_profile(profile)
        
        # Create and return the ItemAgent
        return ItemAgent(
            item_id=item_id,
            item_attributes=item_attributes,
            interaction_context=interaction_context,
            llm=self.llm,
            toolkit=self.toolkit,
            target_user_context=target_user_context,
            query=query,
            memory=memory,
            item_description=item_description
        )
    
    def _get_item_attributes(self, item_id: int) -> Dict[str, Any]:
        """
        Get attributes for an item.
        
        This is a basic implementation that returns minimal attributes.
        Can be enhanced to extract rich item metadata from the dataset.
        
        Args:
            item_id: The item's ID
            
        Returns:
            Dict[str, Any]: Item attributes
        """
        # Basic attributes - can be enhanced with dataset-specific metadata
        return {
            'item_id': item_id,
            'type': 'history_item'
        }
    
    def generate_instructions(
        self,
        round_idx: int,
        agents: List[BaseMACFAgent],
        draft_list: List[int],
        discussion_history: List[Dict]
    ) -> Dict[str, str]:
        """
        Generate personalized instructions for each agent.
        
        Instructions vary based on:
        - Round number (initial vs refinement)
        - Agent type (user vs item)
        - Current draft list state
        - Identified conflicts/agreements
        
        Args:
            round_idx: Current round index (0-based)
            agents: List of active agents to generate instructions for
            draft_list: Current draft of recommended item IDs
            discussion_history: Messages from previous discussion rounds
            
        Returns:
            Dict[str, str]: Mapping from agent_id to personalized instruction
        
        Requirements: 5.2, 6.1
        """
        instructions: Dict[str, str] = {}
        
        # Analyze current state for instruction generation
        is_initial_round = round_idx == 0
        has_draft = len(draft_list) > 0
        draft_size = len(draft_list)
        target_size = self.config.top_k_recommendation
        
        # Identify conflicts from discussion history
        conflicts = self._identify_conflicts_from_history(discussion_history)
        has_conflicts = len(conflicts) > 0
        
        # Identify agreements (items suggested by multiple agents)
        agreements = self._identify_agreements_from_history(discussion_history)
        
        for agent in agents:
            agent_id = agent.agent_id
            is_user_agent = isinstance(agent, UserAgent)
            is_item_agent = isinstance(agent, ItemAgent)
            
            # Build personalized instruction based on context
            instruction_parts = []
            
            # ==================== Round-specific instructions ====================
            if is_initial_round:
                # Initial round: focus on generating diverse suggestions
                instruction_parts.append(
                    "This is the initial discussion round. Please provide your "
                    "top item recommendations based on your unique perspective."
                )
                
                if is_user_agent:
                    instruction_parts.append(
                        "As a similar user, focus on items that align with both "
                        "your preferences and the target user's query. Consider "
                        "what items you've enjoyed that might appeal to them."
                    )
                elif is_item_agent:
                    instruction_parts.append(
                        "As a history item, trace relevance paths to find new "
                        "candidates. Focus on items that share your key attributes "
                        "and relate to the current query."
                    )
            else:
                # Refinement rounds: focus on building consensus
                instruction_parts.append(
                    f"This is round {round_idx + 1} of the discussion. "
                    f"We are refining our recommendations."
                )
                
                if has_draft:
                    instruction_parts.append(
                        f"Current draft list has {draft_size}/{target_size} items: "
                        f"{draft_list[:10]}{'...' if draft_size > 10 else ''}"
                    )
                    
                    if draft_size < target_size:
                        needed = target_size - draft_size
                        instruction_parts.append(
                            f"We need {needed} more items. Please suggest additional "
                            f"candidates that complement the current draft."
                        )
                    else:
                        instruction_parts.append(
                            "The draft list is complete. Please review and confirm "
                            "the selections or suggest replacements with strong justification."
                        )
            
            # ==================== Conflict resolution instructions ====================
            if has_conflicts and round_idx > 0:
                instruction_parts.append("\n## Conflicts to Address")
                instruction_parts.append(
                    "The following conflicts have been identified between agent suggestions:"
                )
                for conflict in conflicts[:3]:  # Limit to top 3 conflicts
                    instruction_parts.append(f"- {conflict}")
                
                if is_user_agent:
                    instruction_parts.append(
                        "Please provide your perspective on these conflicts based on "
                        "your understanding of user preferences."
                    )
                elif is_item_agent:
                    instruction_parts.append(
                        "Please provide your perspective on these conflicts based on "
                        "item relevance and attribute similarity."
                    )
            
            # ==================== Agreement reinforcement ====================
            if agreements and round_idx > 0:
                instruction_parts.append("\n## Points of Agreement")
                instruction_parts.append(
                    f"The following items have received support from multiple agents: "
                    f"{list(agreements.keys())[:5]}"
                )
                instruction_parts.append(
                    "Consider building on these agreements while addressing gaps."
                )
            
            # ==================== Agent-type specific guidance ====================
            if is_user_agent:
                instruction_parts.append(
                    "\n## Your Focus\n"
                    "Leverage your collaborative filtering perspective: what items "
                    "would users with similar preferences enjoy? Consider the target "
                    "user's query and how your experience informs your suggestions."
                )
            elif is_item_agent:
                instruction_parts.append(
                    "\n## Your Focus\n"
                    "Leverage your item-based perspective: what items share your "
                    "attributes and would appeal to someone who engaged with you? "
                    "Trace clear relevance paths from yourself to new candidates."
                )
            
            # Combine all instruction parts
            instructions[agent_id] = "\n".join(instruction_parts)
            
            logger.debug(
                f"Generated instruction for {agent_id} "
                f"(round {round_idx}, {'initial' if is_initial_round else 'refinement'})"
            )
        
        return instructions
    
    def aggregate_responses(
        self,
        responses: List[AgentResponse],
        draft_list: List[int]
    ) -> Tuple[List[int], List[str]]:
        """
        Aggregate agent responses into updated draft list.
        
        Identifies agreements and conflicts between suggestions.
        Returns updated draft list and conflict descriptions.
        
        The aggregation process:
        1. Collect all suggestions from all agents
        2. Score items based on frequency and agent confidence
        3. Identify agreements (items suggested by multiple agents)
        4. Identify conflicts (disagreements on rankings or items)
        5. Update draft list with highest-scoring items
        
        Args:
            responses: List of AgentResponse objects from all active agents
            draft_list: Current draft of recommended item IDs
            
        Returns:
            Tuple[List[int], List[str]]: A tuple containing:
                - Updated draft list of item IDs
                - List of conflict descriptions
        
        Requirements: 5.3, 6.2, 6.3, 6.4
        """
        if not responses:
            logger.warning("No responses to aggregate, returning current draft list")
            return draft_list, []
        
        # Get candidate set if available (for proper evaluation)
        candidate_set = getattr(self, '_current_candidate_items', None)
        
        # ==================== Collect all suggestions ====================
        # item_id -> list of (agent_id, score, reason)
        item_votes: Dict[int, List[Tuple[str, float, str]]] = {}
        
        for response in responses:
            agent_id = response.agent_id
            for suggestion in response.suggestions:
                item_id = suggestion.item_id
                
                # Filter to candidate set if provided
                if candidate_set is not None and item_id not in candidate_set:
                    continue
                
                if item_id not in item_votes:
                    item_votes[item_id] = []
                item_votes[item_id].append((
                    agent_id,
                    suggestion.score,
                    suggestion.reason
                ))
        
        # If no valid suggestions after filtering, use candidate set directly
        if not item_votes and candidate_set:
            logger.warning("No suggestions in candidate set, using candidate set items")
            for item_id in list(candidate_set)[:self.config.top_k_recommendation]:
                item_votes[item_id] = [('fallback', 0.5, 'No agent suggestions in candidate set')]
        
        logger.debug(f"Collected votes for {len(item_votes)} unique items" + 
                    (f" (filtered to candidate set of {len(candidate_set)})" if candidate_set else ""))
        
        # ==================== Score items ====================
        # Scoring formula: weighted combination of vote count and average score
        item_scores: Dict[int, float] = {}
        
        for item_id, votes in item_votes.items():
            vote_count = len(votes)
            avg_score = sum(v[1] for v in votes) / vote_count
            
            # Combined score: emphasize both consensus (vote count) and confidence (avg score)
            # Normalize vote count by number of agents
            num_agents = len(responses)
            consensus_factor = vote_count / num_agents if num_agents > 0 else 0
            
            # Final score: 60% consensus, 40% average confidence
            item_scores[item_id] = 0.6 * consensus_factor + 0.4 * avg_score
        
        # ==================== Identify agreements ====================
        # Items suggested by multiple agents
        agreements: Dict[int, int] = {}
        for item_id, votes in item_votes.items():
            if len(votes) >= 2:
                agreements[item_id] = len(votes)
        
        if agreements:
            logger.debug(
                f"Identified {len(agreements)} items with agreement: "
                f"{list(agreements.keys())[:5]}"
            )
        
        # ==================== Identify conflicts ====================
        conflicts: List[str] = []
        
        # Conflict type 1: Items with high variance in scores
        for item_id, votes in item_votes.items():
            if len(votes) >= 2:
                scores = [v[1] for v in votes]
                score_variance = max(scores) - min(scores)
                if score_variance > 0.4:  # Significant disagreement
                    agents_involved = [v[0] for v in votes]
                    conflicts.append(
                        f"Item {item_id}: Score disagreement (range {min(scores):.2f}-{max(scores):.2f}) "
                        f"between agents {agents_involved}"
                    )
        
        # Conflict type 2: Competing items for same "slot"
        # Group items by their score tier and check for competition
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        target_size = self.config.top_k_recommendation
        
        if len(sorted_items) > target_size:
            # Items just outside the cutoff that have strong support
            cutoff_score = sorted_items[target_size - 1][1] if target_size <= len(sorted_items) else 0
            near_cutoff = [
                (item_id, score) for item_id, score in sorted_items[target_size:target_size + 5]
                if score >= cutoff_score * 0.9  # Within 10% of cutoff
            ]
            
            if near_cutoff:
                conflicts.append(
                    f"Ranking conflict: Items {[item_id for item_id, _ in near_cutoff]} "
                    f"are close to the cutoff and may deserve inclusion"
                )
        
        # Conflict type 3: User agents vs Item agents disagreement
        user_agent_items = set()
        item_agent_items = set()
        
        for response in responses:
            is_user_agent = response.agent_id.startswith('user_agent_')
            for suggestion in response.suggestions[:5]:  # Top 5 from each
                if is_user_agent:
                    user_agent_items.add(suggestion.item_id)
                else:
                    item_agent_items.add(suggestion.item_id)
        
        user_only = user_agent_items - item_agent_items
        item_only = item_agent_items - user_agent_items
        
        if user_only and item_only:
            conflicts.append(
                f"Perspective conflict: User agents favor {list(user_only)[:3]}, "
                f"Item agents favor {list(item_only)[:3]}"
            )
        
        # ==================== Update draft list ====================
        # RE-RANK based on current round scores, with small bonus for existing items
        # This ensures the draft list evolves each round based on new discussion
        
        # Give existing draft items a small persistence bonus (10%) to avoid thrashing
        # but not so much that they can't be replaced by better suggestions
        PERSISTENCE_BONUS = 0.1
        
        final_scores: Dict[int, float] = {}
        for item_id, score in item_scores.items():
            if item_id in draft_list:
                # Small bonus for items already in draft (stability)
                final_scores[item_id] = score + PERSISTENCE_BONUS
            else:
                final_scores[item_id] = score
        
        # Sort by final score and take top K
        sorted_by_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        updated_draft = [item_id for item_id, _ in sorted_by_final[:target_size]]
        
        # If we have a candidate set and not enough items, fill from candidate set
        # This ensures we always rank within the candidate set for proper evaluation
        if candidate_set and len(updated_draft) < target_size:
            # Add remaining candidate items not yet in draft
            remaining_candidates = [c for c in candidate_set if c not in updated_draft]
            # Sort remaining by any scores they might have, or just add them
            remaining_with_scores = [(c, item_scores.get(c, 0.0)) for c in remaining_candidates]
            remaining_with_scores.sort(key=lambda x: x[1], reverse=True)
            for item_id, _ in remaining_with_scores:
                if len(updated_draft) >= target_size:
                    break
                updated_draft.append(item_id)
        
        # Log changes for debugging
        old_set = set(draft_list)
        new_set = set(updated_draft)
        added = new_set - old_set
        removed = old_set - new_set
        
        if added or removed:
            logger.info(
                f"Draft list changed: +{len(added)} added {list(added)[:3]}, "
                f"-{len(removed)} removed {list(removed)[:3]}"
            )
        
        logger.info(
            f"Aggregation complete: {len(updated_draft)} items in draft, "
            f"{len(conflicts)} conflicts identified"
            + (f" (candidate_set={len(candidate_set)})" if candidate_set else "")
        )
        
        return updated_draft, conflicts
    
    def check_convergence(
        self,
        draft_list: List[int],
        rationales: List[str],
        round_idx: int = 0
    ) -> bool:
        """
        Check if discussion should terminate early.
        
        Convergence criteria:
        - Must have completed at least min_rounds (default: 2)
        - Draft list has K items (top_k_recommendation)
        - Rationales are consistent (no major conflicts)
        
        Args:
            draft_list: Current draft of recommended item IDs
            rationales: List of rationale strings from agents
            round_idx: Current round index (0-based)
            
        Returns:
            bool: True if discussion has converged, False otherwise
        
        Requirements: 5.5
        """
        target_size = self.config.top_k_recommendation
        
        # Minimum rounds before allowing convergence (ensures multi-turn discussion)
        min_rounds = getattr(self.config, 'min_rounds', 2)
        
        # Criterion 0: Must complete minimum rounds
        if round_idx < min_rounds - 1:
            logger.debug(
                f"Convergence check: Round {round_idx + 1}/{min_rounds} minimum - not converged"
            )
            return False
        
        # Criterion 1: Draft list must have exactly K items
        if len(draft_list) < target_size:
            logger.debug(
                f"Convergence check: Draft list has {len(draft_list)}/{target_size} items - not converged"
            )
            return False
        
        # Criterion 2: Check rationale consistency
        if not rationales:
            # No rationales to check - consider converged if we have enough items
            logger.debug("Convergence check: No rationales to check, considering converged")
            return True
        
        # Analyze rationales for consistency
        # Look for conflict indicators in rationales
        conflict_indicators = [
            'disagree', 'conflict', 'however', 'but', 'instead',
            'alternative', 'different', 'oppose', 'contrary',
            'reconsider', 'replace', 'remove'
        ]
        
        conflict_count = 0
        for rationale in rationales:
            rationale_lower = rationale.lower()
            for indicator in conflict_indicators:
                if indicator in rationale_lower:
                    conflict_count += 1
                    break  # Count each rationale only once
        
        # Calculate conflict ratio
        conflict_ratio = conflict_count / len(rationales) if rationales else 0
        
        # Convergence threshold: less than 30% of rationales indicate conflict
        convergence_threshold = 0.3
        is_consistent = conflict_ratio < convergence_threshold
        
        if is_consistent:
            logger.info(
                f"Convergence check: CONVERGED - {len(draft_list)} items, "
                f"conflict ratio {conflict_ratio:.2f} < {convergence_threshold}"
            )
        else:
            logger.debug(
                f"Convergence check: Not converged - conflict ratio {conflict_ratio:.2f} >= {convergence_threshold}"
            )
        
        return is_consistent
    
    def select_active_agents(
        self,
        agents: List[BaseMACFAgent],
        round_idx: int,
        conflicts: List[str]
    ) -> List[BaseMACFAgent]:
        """
        Dynamically select which agents remain active.
        
        UPDATED: Less aggressive pruning to maintain diverse perspectives
        throughout the discussion. Keeps at least 50% of agents active.
        
        Selection criteria:
        - All agents active in initial rounds (rounds 0-2)
        - Keep at least 50% of agents in later rounds
        - Keep agents relevant to unresolved conflicts
        
        Args:
            agents: List of all available agents
            round_idx: Current round index (0-based)
            conflicts: List of conflict descriptions from aggregation
            
        Returns:
            List[BaseMACFAgent]: List of agents to remain active for next round
        
        Requirements: 5.4
        """
        if not agents:
            return []
        
        # ==================== Initial rounds: all agents active ====================
        # Keep all agents active for first 3 rounds to gather diverse perspectives
        if round_idx < 3:
            logger.debug(
                f"Round {round_idx}: Keeping all {len(agents)} agents active (initial phase)"
            )
            return agents
        
        # ==================== Later rounds: keep majority active ====================
        # Keep at least 50% of agents to maintain diverse perspectives
        min_keep = max(4, len(agents) // 2)  # At least 4 or 50%
        
        # Count agent types
        user_agents = [a for a in agents if isinstance(a, UserAgent)]
        item_agents = [a for a in agents if isinstance(a, ItemAgent)]
        
        # Analyze conflicts to determine which agent types are needed
        has_perspective_conflict = any('Perspective conflict' in c for c in conflicts)
        has_ranking_conflict = any('Ranking conflict' in c for c in conflicts)
        has_score_disagreement = any('Score disagreement' in c for c in conflicts)
        
        active_agents: List[BaseMACFAgent] = []
        
        # ==================== Selection logic ====================
        
        if has_perspective_conflict or has_ranking_conflict or has_score_disagreement:
            # Keep more agents when there are conflicts to resolve
            n_user_keep = max(2, len(user_agents) // 2)
            n_item_keep = max(2, len(item_agents) // 2)
            active_agents.extend(user_agents[:n_user_keep])
            active_agents.extend(item_agents[:n_item_keep])
            logger.debug(
                f"Conflicts detected: keeping {len(active_agents)} agents "
                f"({n_user_keep} user, {n_item_keep} item)"
            )
        else:
            # No major conflicts: still keep a good number for consensus
            n_user_keep = max(2, len(user_agents) // 2)
            n_item_keep = max(2, len(item_agents) // 2)
            active_agents.extend(user_agents[:n_user_keep])
            active_agents.extend(item_agents[:n_item_keep])
            logger.debug(
                f"No major conflicts: keeping {len(active_agents)} agents for consensus"
            )
        
        # ==================== Ensure minimum agents ====================
        # Always keep at least min_keep agents active
        if len(active_agents) < min_keep:
            for agent in agents:
                if agent not in active_agents:
                    active_agents.append(agent)
                if len(active_agents) >= min_keep:
                    break
        
        logger.info(
            f"Agent selection for round {round_idx + 1}: "
            f"{len(active_agents)}/{len(agents)} agents active"
        )
        
        return active_agents
    
    def _transform_discussion_log_for_agents(
        self,
        discussion_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform discussion_log into agent-friendly format.
        
        The discussion_log contains round_log dicts with structure:
        {
            'round_idx': int,
            'active_agents': List[str],
            'draft_list_before': List[int],
            'responses': List[Dict],  # Each has agent_id, suggestions, rationale
            'conflicts': List[str]
        }
        
        Agents expect a flat list of messages:
        [{'agent_id': str, 'content': str, 'round_idx': int}]
        
        Args:
            discussion_log: List of round_log dicts from previous rounds
            
        Returns:
            List[Dict]: Flattened list of agent messages
        """
        agent_messages = []
        
        for round_log in discussion_log:
            round_idx = round_log.get('round_idx', 0)
            responses = round_log.get('responses', [])
            draft_list = round_log.get('draft_list_after', round_log.get('draft_list_before', []))
            
            for resp in responses:
                if 'error' in resp:
                    continue
                
                agent_id = resp.get('agent_id', 'unknown')
                suggestions = resp.get('suggestions', [])
                rationale = resp.get('rationale', '')
                
                # Build content string from suggestions and rationale
                content_parts = []
                
                if rationale:
                    content_parts.append(f"Rationale: {rationale}")
                
                if suggestions:
                    suggestion_strs = []
                    for s in suggestions[:5]:  # Limit to top 5
                        item_id = s.get('item_id', '?')
                        score = s.get('score', 0.0)
                        reason = s.get('reason', '')[:100]  # Truncate reason
                        suggestion_strs.append(f"Item {item_id} (score={score:.2f}): {reason}")
                    content_parts.append("Suggestions: " + " | ".join(suggestion_strs))
                
                content = "\n".join(content_parts) if content_parts else "No suggestions provided."
                
                agent_messages.append({
                    'agent_id': agent_id,
                    'content': content,
                    'round_idx': round_idx
                })
            
            # Also add a summary of the draft list state after this round
            if draft_list:
                agent_messages.append({
                    'agent_id': 'orchestrator',
                    'content': f"Current draft list after round {round_idx + 1}: {draft_list[:10]}",
                    'round_idx': round_idx
                })
        
        return agent_messages
    
    def _identify_conflicts_from_history(
        self,
        discussion_history: List[Dict]
    ) -> List[str]:
        """
        Identify conflicts from discussion history.
        
        Analyzes previous round messages to find disagreements
        between agents.
        
        Args:
            discussion_history: Messages from previous discussion rounds
            
        Returns:
            List[str]: List of conflict descriptions
        """
        conflicts: List[str] = []
        
        if not discussion_history:
            return conflicts
        
        # Group messages by round
        rounds: Dict[int, List[Dict]] = {}
        for msg in discussion_history:
            round_idx = msg.get('round_idx', 0)
            if round_idx not in rounds:
                rounds[round_idx] = []
            rounds[round_idx].append(msg)
        
        # Analyze each round for conflicts
        for round_idx, messages in rounds.items():
            # Extract suggested items from each message
            agent_suggestions: Dict[str, List[int]] = {}
            
            for msg in messages:
                agent_id = msg.get('agent_id', 'unknown')
                content = msg.get('content', '')
                
                # Simple extraction: look for item IDs mentioned
                import re
                item_ids = re.findall(r'\bitem[_\s]*(\d+)\b', content.lower())
                item_ids.extend(re.findall(r'\bITEM:\s*(\d+)', content))
                
                if item_ids:
                    agent_suggestions[agent_id] = [int(id) for id in item_ids[:10]]
            
            # Find disagreements between agents
            if len(agent_suggestions) >= 2:
                all_items = set()
                for items in agent_suggestions.values():
                    all_items.update(items)
                
                # Items suggested by only one agent
                for item_id in all_items:
                    suggesting_agents = [
                        agent_id for agent_id, items in agent_suggestions.items()
                        if item_id in items
                    ]
                    if len(suggesting_agents) == 1:
                        # This item has no consensus
                        pass  # Not necessarily a conflict
        
        return conflicts
    
    def _identify_agreements_from_history(
        self,
        discussion_history: List[Dict]
    ) -> Dict[int, int]:
        """
        Identify agreements from discussion history.
        
        Finds items that have been suggested by multiple agents
        across discussion rounds.
        
        Args:
            discussion_history: Messages from previous discussion rounds
            
        Returns:
            Dict[int, int]: Mapping from item_id to number of agents supporting it
        """
        agreements: Dict[int, int] = {}
        
        if not discussion_history:
            return agreements
        
        # Track which agents suggested which items
        item_supporters: Dict[int, set] = {}
        
        for msg in discussion_history:
            agent_id = msg.get('agent_id', 'unknown')
            content = msg.get('content', '')
            
            # Extract item IDs from message
            import re
            item_ids = re.findall(r'\bitem[_\s]*(\d+)\b', content.lower())
            item_ids.extend(re.findall(r'\bITEM:\s*(\d+)', content))
            
            for item_id_str in item_ids:
                try:
                    item_id = int(item_id_str)
                    if item_id not in item_supporters:
                        item_supporters[item_id] = set()
                    item_supporters[item_id].add(agent_id)
                except ValueError:
                    continue
        
        # Items with multiple supporters are agreements
        for item_id, supporters in item_supporters.items():
            if len(supporters) >= 2:
                agreements[item_id] = len(supporters)
        
        return agreements
    
    def _update_agent_memories(
        self,
        responses: List[AgentResponse],
        all_agents: List[BaseMACFAgent],
        query: str,
        round_idx: int,
        draft_list: List[int]
    ):
        """
        Update persistent memories for agents after a discussion round.
        
        Updates BOTH:
        1. Episodic memory: What happened in this interaction (stored in memories list)
        2. Semantic memory: Profile updates based on feedback (what was accepted/rejected)
        
        This ensures agent profiles evolve based on their decisions, similar to ConnaCF.
        
        Args:
            responses: List of agent responses from this round
            all_agents: All agents participating in the discussion
            query: The current query
            round_idx: Current round index
            draft_list: Current draft list of recommendations
        """
        if not self.memory_store:
            return
        
        # Create a mapping from agent_id to agent
        agent_map = {agent.agent_id: agent for agent in all_agents}
        
        for response in responses:
            agent_id = response.agent_id
            agent = agent_map.get(agent_id)
            
            if not agent:
                continue
            
            # Build episodic memory content from the response
            memory_content = self._build_memory_content(response, draft_list)
            
            if not memory_content:
                continue
            
            # Determine agent type and entity ID
            if isinstance(agent, UserAgent):
                # Update episodic memory
                self.memory_store.update_user_memory(
                    user_id=agent.neighbor_user_id,
                    content=memory_content,
                    query=query,
                    round_idx=round_idx,
                    context={
                        'suggestions': [s.item_id for s in response.suggestions[:5]],
                        'draft_list': draft_list[:10]
                    }
                )
                
                # Update semantic memory (profile) based on feedback
                self._update_semantic_profile_from_feedback(
                    agent=agent,
                    response=response,
                    draft_list=draft_list,
                    query=query
                )
                
            elif isinstance(agent, ItemAgent):
                # Update episodic memory
                self.memory_store.update_item_memory(
                    item_id=agent.item_id,
                    content=memory_content,
                    query=query,
                    round_idx=round_idx,
                    context={
                        'suggestions': [s.item_id for s in response.suggestions[:5]],
                        'draft_list': draft_list[:10]
                    }
                )
                
                # Update semantic memory (profile) based on feedback
                self._update_semantic_profile_from_feedback(
                    agent=agent,
                    response=response,
                    draft_list=draft_list,
                    query=query
                )
        
        logger.debug(f"Updated memories for {len(responses)} agents in round {round_idx}")
    
    def _update_semantic_profile_from_feedback(
        self,
        agent: BaseMACFAgent,
        response: AgentResponse,
        draft_list: List[int],
        query: str
    ):
        """
        Update agent's semantic profile based on feedback from the discussion.
        
        This implements the key insight that agent profiles should evolve based on:
        1. Which suggestions were accepted (positive feedback)
        2. Which suggestions were rejected (negative feedback)
        3. What the agent learned from other agents' suggestions
        
        Args:
            agent: The agent whose profile to update
            response: The agent's response
            draft_list: Current draft list (accepted items)
            query: The current query
        """
        if not agent.memory:
            return
        
        # Calculate acceptance rate for this agent's suggestions
        suggested_ids = {s.item_id for s in response.suggestions}
        accepted_ids = suggested_ids & set(draft_list)
        acceptance_rate = len(accepted_ids) / max(1, len(suggested_ids))
        
        # Extract learnings based on feedback
        learnings = []
        
        # Learn from accepted suggestions
        if accepted_ids:
            for item_id in list(accepted_ids)[:3]:
                desc = self.toolkit.index_manager.get_item_description(item_id)
                if desc and desc.strip() and desc != '[PAD]':
                    # Extract genre/style from description
                    learnings.append(f"Successfully recommended: {desc[:80]}")
        
        # Learn from rejected suggestions (if many were rejected)
        rejected_ids = suggested_ids - accepted_ids
        if len(rejected_ids) > len(accepted_ids) and rejected_ids:
            # Agent's suggestions weren't well received - note this
            learnings.append(f"Query '{query[:50]}' - suggestions had low acceptance ({acceptance_rate:.0%})")
        
        # Add learned preferences based on accepted items
        if accepted_ids and isinstance(agent, UserAgent):
            for item_id in list(accepted_ids)[:2]:
                desc = self.toolkit.index_manager.get_item_description(item_id)
                if desc:
                    # Add as a preference
                    agent.memory.add_preference(f"items like '{desc[:50]}'")
        
        # Update profile summary if significant learning occurred
        if learnings and agent.memory.profile:
            # Append learning summary to profile (but keep it bounded)
            current_profile = agent.memory.profile
            learning_summary = " | ".join(learnings[:2])
            
            # Only update if profile isn't too long
            if len(current_profile) < 1500:
                updated_profile = f"{current_profile}\n[Recent learning: {learning_summary}]"
                agent.memory.profile = updated_profile
                
                logger.debug(f"Updated semantic profile for {agent.agent_id} with feedback")
    
    def _build_memory_content(
        self,
        response: AgentResponse,
        draft_list: List[int]
    ) -> str:
        """
        Build memory content from an agent response.
        
        Saves the FULL response for detailed analysis and learning.
        Includes both the raw response, structured information, and
        ACTUAL ITEM DESCRIPTIONS for semantic memory.
        
        Args:
            response: The agent's response
            draft_list: Current draft list
            
        Returns:
            str: Memory content to store (full response with item content)
        """
        if not response.suggestions and not response.rationale and not response.raw_response:
            return ""
        
        parts = []
        
        # Include FULL rationale (no truncation)
        if response.rationale:
            parts.append(f"RATIONALE: {response.rationale}")
        
        # Include FULL raw response
        if response.raw_response:
            parts.append(f"FULL RESPONSE: {response.raw_response}")
        
        # Include all suggestions with full reasons AND item descriptions
        if response.suggestions:
            suggestions_text = []
            for s in response.suggestions[:10]:
                # Get actual item description for semantic memory
                item_desc = self.toolkit.index_manager.get_item_description(s.item_id)
                if item_desc and item_desc.strip() and item_desc != '[PAD]':
                    # Truncate very long descriptions for memory
                    if len(item_desc) > 150:
                        item_desc = item_desc[:150] + "..."
                    suggestions_text.append(
                        f"Item {s.item_id} '{item_desc}' (score={s.score:.2f}): {s.reason}"
                    )
                else:
                    suggestions_text.append(f"Item {s.item_id} (score={s.score:.2f}): {s.reason}")
            parts.append(f"SUGGESTIONS: {' | '.join(suggestions_text)}")
        
        # Include which suggestions made it to the draft list
        if response.suggestions and draft_list:
            accepted = [s.item_id for s in response.suggestions if s.item_id in draft_list]
            if accepted:
                # Include descriptions for accepted items
                accepted_with_desc = []
                for item_id in accepted[:10]:
                    desc = self.toolkit.index_manager.get_item_description(item_id)
                    if desc and len(desc) > 80:
                        desc = desc[:80] + "..."
                    accepted_with_desc.append(f"{item_id} ('{desc}')" if desc else str(item_id))
                parts.append(f"ACCEPTED: {accepted_with_desc}")
        
        return "\n".join(parts) if parts else ""
    
    def run_inference(self, target_user_id: int, query: str, ground_truth: List[int] = None, candidate_items: List[int] = None) -> RankedList:
        """
        Run MACF inference for a single user-query pair.
        
        This is the main entry point for MACF inference. It coordinates
        the multi-round discussion process:
        1. Recruit agents via retrieval tools (including adversarial agents if attack_hooks set)
        2. Run multi-round discussion (max 5 rounds)
        3. Aggregate and return final ranked list
        
        Attack Framework Integration (Requirements 11.2, 11.4, 11.5, 11.6):
        When attack_hooks is set, this method:
        - Calls intercept_message() on all agent instructions before sending
        - Calls intercept_message() on all agent responses after receiving
        - Calls log_interaction() for each discussion round
        - Continues execution even when attacks modify behavior
        
        Args:
            target_user_id: The ID of the target user for recommendations
            query: Natural language query expressing user intent
            ground_truth: Optional list of ground truth item IDs for logging
            candidate_items: Optional list of candidate item IDs to rank within.
                            If provided, the final ranking will only include items
                            from this set (like ConnaCF's evaluation protocol).
            
        Returns:
            RankedList: Final ranked list with exactly top_k_recommendation items,
                       including items, scores, rationales, and discussion_log
        
        Example:
            >>> result = orchestrator.run_inference(
            ...     target_user_id=123,
            ...     query="action movies with good plot"
            ... )
            >>> print(f"Recommended {len(result)} items: {result.items}")
        
        Requirements: 5.3, 5.5, 5.6, 5.7, 6.1, 6.2, 6.3, 6.6, 11.2, 11.4, 11.5, 11.6
        """
        from connacf.macf.error_handling import pad_ranked_list, parse_agent_response
        
        # Store candidate set for filtering during aggregation
        self._current_candidate_items = set(candidate_items) if candidate_items else None
        
        logger.info(
            f"Starting MACF inference for user {target_user_id} "
            f"with query: '{query[:50]}...'"
            + (f", candidate_set_size={len(candidate_items)}" if candidate_items else "")
        )
        
        # ==================== Metrics: Start Task ====================
        if self.metrics_collector:
            self.metrics_collector.start_task(
                user_id=target_user_id,
                query=query,
                ground_truth=ground_truth or []
            )
            
            # Log task boundary to wandb
            if self.wandb_logger:
                try:
                    self.wandb_logger.log_task_boundary(
                        task_id=self.metrics_collector.task_counter,
                        global_turn=self.metrics_collector.global_turn_counter
                    )
                except Exception as e:
                    logger.warning(f"Failed to log task boundary to wandb: {e}")
        
        # ==================== Conversation Logging: Query Start ====================
        if self.conversation_logger:
            self.conversation_logger.log_query_start(target_user_id, query)
        
        # Initialize discussion state
        discussion_log: List[Dict[str, Any]] = []
        draft_list: List[int] = []
        rationales: List[str] = []
        conflicts: List[str] = []
        
        # ==================== Step 1: Recruit agents ====================
        try:
            user_agents, item_agents = self.recruit_agents(target_user_id, query)
        except Exception as e:
            logger.error(f"Agent recruitment failed: {e}")
            if self.conversation_logger:
                self.conversation_logger.log_error("Agent Recruitment", str(e))
            user_agents, item_agents = [], []
        
        # Count adversarial agents if attack hooks present
        adversarial_count = 0
        adversarial_agent_ids = []  # List of agent ID strings (not agent objects)
        if self.attack_hooks is not None:
            try:
                adversarial_agents = self.attack_hooks.get_adversarial_agents()
                adversarial_count = len(adversarial_agents) if adversarial_agents else 0
                # Extract agent IDs from agent objects (not the objects themselves)
                adversarial_agent_ids = [
                    agent.agent_id for agent in adversarial_agents
                ] if adversarial_agents else []
            except Exception as e:
                logger.warning(f"Failed to get adversarial agents: {e}")
                pass
        
        # ==================== Pre-Discussion Callback (for dynamic attacks) ====================
        # This allows attackers to select which active agents to poison
        # IMPORTANT: This must happen BEFORE logging so attacker status is correct
        if self.pre_discussion_callback is not None:
            try:
                self.pre_discussion_callback(user_agents, item_agents, self._task_counter)
            except Exception as e:
                logger.warning(f"Pre-discussion callback failed: {e}")
        
        # ==================== Update adversarial agent tracking AFTER callback ====================
        # The callback may have registered attackers with metrics_collector.all_attacker_ids
        # We need to update adversarial_agent_ids to reflect this
        if self.metrics_collector and self.metrics_collector.all_attacker_ids:
            adversarial_agent_ids = list(self.metrics_collector.all_attacker_ids)
            adversarial_count = len(adversarial_agent_ids)
            logger.info(f"Updated adversarial tracking: {adversarial_count} attackers from metrics collector")
        
        # ==================== Conversation Logging: Agent Recruitment ====================
        # NOTE: This is done AFTER pre_discussion_callback so attacker status is correct
        if self.conversation_logger:
            self.conversation_logger.log_agent_recruitment(
                user_agents, item_agents, adversarial_count
            )
            # Log semantic profiles in ConnaCF-compatible format for calibration
            self.conversation_logger.log_agent_profiles(
                user_agents, item_agents, self.memory_store
            )
        
        self._task_counter += 1
        
        # Combine all agents into a single pool
        all_agents: List[BaseMACFAgent] = list(user_agents) + list(item_agents)
        active_agents: List[BaseMACFAgent] = all_agents.copy()
        
        if not active_agents:
            logger.warning(
                f"No agents recruited for user {target_user_id}, "
                f"using retrieval-based fallback"
            )
            # Fallback: use retrieval to get candidates directly
            draft_list = pad_ranked_list(
                draft_list=[],
                target_size=self.config.top_k_recommendation,
                toolkit=self.toolkit,
                query=query
            )
            return RankedList(
                items=draft_list,
                scores=[1.0 / (i + 1) for i in range(len(draft_list))],
                rationales=["Retrieval-based fallback (no agents available)"] * len(draft_list),
                discussion_log=discussion_log
            )
        
        logger.info(
            f"Recruited {len(user_agents)} user agents and {len(item_agents)} item agents"
        )
        
        # ==================== Step 2: Multi-round discussion ====================
        max_rounds = self.config.max_rounds
        converged = False
        
        for round_idx in range(max_rounds):
            logger.info(f"Starting discussion round {round_idx + 1}/{max_rounds}")
            
            # ==================== Metrics: Start Turn ====================
            if self.metrics_collector:
                self.metrics_collector.start_turn(round_idx)
                self.metrics_collector.record_agent_participation(
                    num_user_agents=len([a for a in active_agents if isinstance(a, UserAgent)]),
                    num_item_agents=len([a for a in active_agents if isinstance(a, ItemAgent)]),
                    num_adversarial=adversarial_count,
                    agent_ids=[a.agent_id for a in active_agents],  # Track all agents seen
                    attacker_ids=adversarial_agent_ids  # Track attackers separately (red)
                )
            
            # ==================== Communication Graph: Start Turn ====================
            if self.communication_graph and self.metrics_collector:
                user_agent_ids = [a.agent_id for a in active_agents if isinstance(a, UserAgent)]
                item_agent_ids = [a.agent_id for a in active_agents if isinstance(a, ItemAgent)]
                self.communication_graph.start_turn(
                    global_turn=self.metrics_collector.global_turn_counter,
                    task_id=self._task_counter,
                    round_idx=round_idx,
                    active_user_agents=user_agent_ids,
                    active_item_agents=item_agent_ids
                )
            
            # ==================== Conversation Logging: Round Start ====================
            if self.conversation_logger:
                self.conversation_logger.log_round_start(
                    round_idx, max_rounds, 
                    [a.agent_id for a in active_agents]
                )
            
            round_log: Dict[str, Any] = {
                'round_idx': round_idx,
                'active_agents': [a.agent_id for a in active_agents],
                'draft_list_before': draft_list.copy(),
                'responses': [],
                'conflicts': []
            }
            
            # ==================== Generate instructions for all active agents ====================
            instructions = self.generate_instructions(
                round_idx=round_idx,
                agents=active_agents,
                draft_list=draft_list,
                discussion_history=discussion_log
            )
            
            # ==================== Collect responses from all agents ====================
            responses: List[AgentResponse] = []
            
            for agent in active_agents:
                agent_id = agent.agent_id
                instruction = instructions.get(agent_id, "Please provide your recommendations.")
                
                # ==================== Attack Hook: Intercept instruction (Req 11.2) ====================
                # Apply attack hooks message interception to instructions before sending
                # This allows attackers to modify the instructions agents receive
                original_instruction = instruction
                instruction_was_modified = False
                if self.attack_hooks is not None:
                    try:
                        instruction = self.attack_hooks.intercept_message(agent_id, instruction)
                        if instruction != original_instruction:
                            logger.debug(f"Instruction for {agent_id} was modified by attack hooks")
                            instruction_was_modified = True
                    except Exception as e:
                        # Continue execution even if interception fails (Req 11.5)
                        logger.warning(f"Attack hook intercept_message failed for instruction: {e}")
                        instruction = original_instruction
                
                # ==================== Conversation Logging: Instruction ====================
                if self.conversation_logger:
                    self.conversation_logger.log_instruction(
                        agent_id, instruction, instruction_was_modified
                    )
                
                try:
                    # Transform discussion_log into agent-friendly format
                    # Agents expect: [{'agent_id': ..., 'content': ..., 'round_idx': ...}]
                    agent_discussion_history = self._transform_discussion_log_for_agents(discussion_log)
                    
                    # Get agent response
                    response = agent.reply(
                        instruction=instruction,
                        discussion_history=agent_discussion_history
                    )
                    
                    # ==================== Attack Hook: Intercept response (Req 11.2) ====================
                    # Apply attack hooks to agent response after receiving
                    # This allows attackers to modify agent outputs
                    response_was_modified = False
                    if self.attack_hooks is not None:
                        try:
                            original_raw = response.raw_response
                            modified_raw = self.attack_hooks.intercept_message(
                                agent_id, 
                                response.raw_response
                            )
                            if modified_raw != original_raw:
                                # Re-parse if response was modified
                                logger.debug(f"Response from {agent_id} was modified by attack hooks")
                                response = parse_agent_response(modified_raw, agent_id)
                                response_was_modified = True
                        except Exception as e:
                            # Continue execution even if interception fails (Req 11.5)
                            logger.warning(f"Attack hook intercept_message failed for response: {e}")
                    
                    responses.append(response)
                    
                    # ==================== Metrics: Record Agent Response ====================
                    if self.metrics_collector:
                        self.metrics_collector.record_agent_response(
                            agent_id=agent_id,
                            num_suggestions=len(response.suggestions),
                            tokens_used=0  # TODO: Get actual token count from LLM
                        )
                    
                    # ==================== Conversation Logging: Agent Response ====================
                    if self.conversation_logger:
                        # Get list of other agents who will hear this response
                        other_agent_ids = [a.agent_id for a in active_agents if a.agent_id != agent_id]
                        
                        # Get item descriptions for suggested items
                        suggestion_item_descs = {}
                        for s in response.suggestions:
                            desc = self.toolkit.index_manager.get_item_description(s.item_id)
                            if desc and desc.strip() and desc != '[PAD]':
                                suggestion_item_descs[s.item_id] = desc
                        
                        # Get system prompt for debugging (especially for NetSAFE attacks)
                        agent_system_prompt = None
                        if hasattr(agent, 'get_system_prompt'):
                            try:
                                agent_system_prompt = agent.get_system_prompt()
                            except Exception:
                                pass
                        
                        self.conversation_logger.log_agent_response(
                            agent_id=agent_id,
                            raw_response=response.raw_response,
                            suggestions=[
                                {'item_id': s.item_id, 'score': s.score, 'reason': s.reason}
                                for s in response.suggestions
                            ],
                            rationale=response.rationale,
                            tool_calls=[
                                {'tool_name': tc.tool_name, 'result': tc.result}
                                for tc in response.tool_calls
                            ] if response.tool_calls else None,
                            was_modified=response_was_modified,
                            listeners=other_agent_ids,  # Track who hears this for contamination
                            item_descriptions=suggestion_item_descs,  # Include item descriptions
                            system_prompt=agent_system_prompt  # Include system prompt for debugging
                        )
                    
                    # ==================== Communication Graph: Record broadcast ====================
                    if self.communication_graph:
                        other_agent_ids = [a.agent_id for a in active_agents if a.agent_id != agent_id]
                        content_summary = response.rationale[:100] if response.rationale else ""
                        self.communication_graph.record_broadcast(
                            speaker_agent_id=agent_id,
                            listener_agent_ids=other_agent_ids,
                            content_summary=content_summary
                        )
                        # Also record suggestions as edges to item agents
                        if response.suggestions:
                            self.communication_graph.record_suggestion(
                                agent_id=agent_id,
                                suggested_item_ids=[s.item_id for s in response.suggestions[:10]],
                                content_summary=content_summary
                            )
                    
                    # Log the response
                    round_log['responses'].append({
                        'agent_id': agent_id,
                        'suggestions': [
                            {'item_id': s.item_id, 'score': s.score, 'reason': s.reason}
                            for s in response.suggestions
                        ],
                        'rationale': response.rationale
                    })
                    
                except Exception as e:
                    # Handle failures gracefully - log and continue
                    logger.warning(
                        f"Failed to get response from agent {agent_id}: {e}"
                    )
                    if self.conversation_logger:
                        self.conversation_logger.log_error(f"Agent {agent_id} Response", str(e))
                    round_log['responses'].append({
                        'agent_id': agent_id,
                        'error': str(e)
                    })
            
            # ==================== Aggregate responses to update draft list ====================
            draft_list_before = draft_list.copy()
            if responses:
                draft_list, conflicts = self.aggregate_responses(
                    responses=responses,
                    draft_list=draft_list
                )
                
                # Collect rationales from responses
                rationales = [r.rationale for r in responses if r.rationale]
                
                # Get agreements for logging
                agreements = self._identify_agreements_from_history(discussion_log)
                
                # ==================== Metrics: Record Aggregation ====================
                if self.metrics_collector:
                    self.metrics_collector.record_aggregation(
                        draft_list_before=draft_list_before,
                        draft_list_after=draft_list,
                        agreements=agreements,
                        conflicts=conflicts
                    )
                    
                    # Record per-turn ranking prediction for temporal analysis
                    # This forces a ranking prediction every turn to track improvement
                    # IMPORTANT: Use full candidate set ranking, not just draft_list
                    # Otherwise Hit@K will be 0 if ground truth isn't in agent suggestions
                    candidate_set = getattr(self, '_current_candidate_items', None)
                    if candidate_set:
                        # Build full ranking: draft_list first, then remaining candidates
                        full_ranking_for_metrics = list(draft_list)
                        for item_id in candidate_set:
                            if item_id not in full_ranking_for_metrics:
                                full_ranking_for_metrics.append(item_id)
                        self.metrics_collector.record_ranking_prediction(
                            predicted_items=full_ranking_for_metrics,
                            ground_truth=ground_truth or []
                        )
                    else:
                        self.metrics_collector.record_ranking_prediction(
                            predicted_items=draft_list,
                            ground_truth=ground_truth or []
                        )
                    
                    # Collect agent memories for LLM judge evaluation
                    # Now includes BOTH semantic (profile) and episodic (recent reasonings) memory
                    agent_memories = {}  # Semantic memory (current profile)
                    agent_episodic_memories = {}  # Episodic memory (recent reasonings)
                    agent_original_profiles = {}  # Original profile (for comparison)
                    
                    for agent in active_agents:
                        if agent.memory:
                            # Semantic memory: current profile
                            agent_memories[agent.agent_id] = agent.memory.profile if agent.memory.profile else ""
                            
                            # Episodic memory: recent reasonings/interactions
                            recent_memories = agent.memory.get_recent_memories(n=5)
                            if recent_memories:
                                agent_episodic_memories[agent.agent_id] = "\n".join(recent_memories)
                            
                            # Original profile: stored when agent was first created
                            # This is tracked in memory_store for comparison
                            if hasattr(agent.memory, '_original_profile'):
                                agent_original_profiles[agent.agent_id] = agent.memory._original_profile
                            
                            # Update agent memory cache for 'all_agents' evaluation mode
                            # This ensures we have up-to-date memories for all agents
                            self.metrics_collector.update_agent_memory_cache(
                                agent_id=agent.agent_id,
                                memory=agent_memories.get(agent.agent_id),
                                episodic=agent_episodic_memories.get(agent.agent_id),
                                original=agent_original_profiles.get(agent.agent_id)
                            )
                    
                    # Queue for batched LLM judge evaluation
                    self.metrics_collector.queue_for_llm_judge(
                        draft_list=draft_list,
                        ground_truth=ground_truth or [],
                        agent_responses=[
                            {
                                'agent_id': r.agent_id,
                                'suggestions': [s.item_id for s in r.suggestions[:5]],
                                'rationale': r.rationale
                            }
                            for r in responses
                        ],
                        query=query,
                        user_id=target_user_id,
                        agent_memories=agent_memories if agent_memories else None,
                        agent_episodic_memories=agent_episodic_memories if agent_episodic_memories else None,
                        agent_original_profiles=agent_original_profiles if agent_original_profiles else None
                    )
                
                # ==================== Conversation Logging: Aggregation Result ====================
                if self.conversation_logger:
                    self.conversation_logger.log_aggregation_result(
                        draft_list, conflicts, agreements
                    )
            
            round_log['draft_list_after'] = draft_list.copy()
            round_log['conflicts'] = conflicts
            
            # ==================== Metrics: End Turn ====================
            if self.metrics_collector:
                self.metrics_collector.end_turn()
            
            # ==================== Communication Graph: End Turn ====================
            if self.communication_graph:
                self.communication_graph.end_turn()
                
                # Log turn metrics to wandb
                if self.wandb_logger and self.metrics_collector and self.metrics_collector.current_turn_metrics is None:
                    # Turn metrics were just saved, get the last one from task
                    try:
                        if self.metrics_collector.current_task and self.metrics_collector.current_task.turn_metrics:
                            last_turn = self.metrics_collector.current_task.turn_metrics[-1]
                            self.wandb_logger.log_turn_metrics(
                                global_turn=last_turn.global_turn,
                                task_id=last_turn.task_id,
                                turn_in_task=last_turn.turn,
                                utility_metrics=last_turn.utility,
                                dissemination_metrics=last_turn.dissemination,
                                ranking_metrics=last_turn.ranking_prediction,
                                llm_judge_metrics=last_turn.llm_judge,
                                resource_metrics=last_turn.resources
                            )
                    except Exception as e:
                        logger.warning(f"Failed to log turn metrics to wandb: {e}")
            
            # ==================== Update Agent Memories ====================
            # Update persistent memory for agents that participated in this round
            if self.memory_store and responses:
                self._update_agent_memories(
                    responses=responses,
                    all_agents=all_agents,
                    query=query,
                    round_idx=round_idx,
                    draft_list=draft_list
                )
            
            # ==================== Attack Hook: Log interaction (Req 11.4) ====================
            # Log each discussion round for attack analysis
            # This enables post-hoc analysis of how attacks affected the discussion
            if self.attack_hooks is not None:
                try:
                    self.attack_hooks.log_interaction({
                        'type': 'discussion_round',
                        'round_idx': round_idx,
                        'target_user_id': target_user_id,
                        'query': query,
                        'active_agents': [a.agent_id for a in active_agents],
                        'num_responses': len(responses),
                        'draft_list_size': len(draft_list),
                        'conflicts': conflicts
                    })
                except Exception as e:
                    # Continue execution even if logging fails (Req 11.5)
                    logger.warning(f"Attack hook log_interaction failed: {e}")
            
            # Add round log to discussion log
            discussion_log.append(round_log)
            
            # ==================== Check convergence ====================
            converged = self.check_convergence(
                draft_list=draft_list,
                rationales=rationales,
                round_idx=round_idx
            )
            
            # ==================== Conversation Logging: Convergence Check ====================
            if self.conversation_logger:
                reason = f"{len(draft_list)}/{self.config.top_k_recommendation} items, round {round_idx + 1}"
                self.conversation_logger.log_convergence_check(converged, reason)
                # Log contamination summary after each round
                self.conversation_logger.log_contamination_summary(round_idx)
            
            if converged:
                logger.info(
                    f"Discussion converged at round {round_idx + 1} "
                    f"with {len(draft_list)} items"
                )
                break
            
            # ==================== Select active agents for next round ====================
            if round_idx < max_rounds - 1:  # Don't select for last round
                active_agents = self.select_active_agents(
                    agents=all_agents,
                    round_idx=round_idx,
                    conflicts=conflicts
                )
                
                if not active_agents:
                    logger.warning("No active agents for next round, ending discussion")
                    break
        
        # ==================== Step 3: Finalize ranked list ====================
        # For proper evaluation, we need to rank the FULL candidate set
        # The draft_list contains the top items, but we need to return a ranking
        # of all candidate items for Hit@K and NDCG@K computation
        
        candidate_set = getattr(self, '_current_candidate_items', None)
        
        if candidate_set:
            # Build a full ranking of the candidate set
            # Items in draft_list are ranked first (in order), then remaining candidates
            full_ranking = list(draft_list)
            for item_id in candidate_set:
                if item_id not in full_ranking:
                    full_ranking.append(item_id)
            draft_list = full_ranking
            logger.info(f"Full candidate set ranking: {len(draft_list)} items")
        else:
            # No candidate set - use original logic
            target_size = self.config.top_k_recommendation
            
            if len(draft_list) < target_size:
                logger.info(
                    f"Draft list has {len(draft_list)} items, "
                    f"padding to {target_size}"
                )
                draft_list = pad_ranked_list(
                    draft_list=draft_list,
                    target_size=target_size,
                    toolkit=self.toolkit,
                    query=query
                )
            elif len(draft_list) > target_size:
                # Truncate to target size
                draft_list = draft_list[:target_size]
        
        # Build scores (decreasing by rank position)
        scores = [1.0 / (i + 1) for i in range(len(draft_list))]
        
        # Build rationales for each item
        final_rationales = self._build_final_rationales(
            draft_list=draft_list,
            discussion_log=discussion_log
        )
        
        # Ensure rationales list matches items list length
        while len(final_rationales) < len(draft_list):
            final_rationales.append("No rationale available")
        final_rationales = final_rationales[:len(draft_list)]
        
        logger.info(
            f"MACF inference complete for user {target_user_id}: "
            f"{len(draft_list)} items, {len(discussion_log)} rounds"
        )
        
        # ==================== Conversation Logging: Final Result ====================
        if self.conversation_logger:
            # Get user history for logging
            user_history = self.toolkit.index_manager.get_user_history(target_user_id)
            
            self.conversation_logger.log_final_result(
                target_user_id=target_user_id,
                query=query,
                final_items=draft_list,
                scores=scores,
                rationales=final_rationales,
                num_rounds=len(discussion_log),
                user_history=user_history,
                ground_truth=ground_truth
            )
            # Also save detailed JSON for analysis
            self.conversation_logger.save_query_json({
                'target_user_id': target_user_id,
                'query': query,
                'final_items': draft_list,
                'scores': scores,
                'rationales': final_rationales,
                'num_rounds': len(discussion_log),
                'user_history': user_history,
                'ground_truth': ground_truth,
                'discussion_log': discussion_log
            })
        
        # ==================== Metrics: End Task ====================
        if self.metrics_collector:
            # Compute Hit@10 and NDCG@10 for metrics
            hit_at_10 = 0.0
            ndcg_at_10 = 0.0
            if ground_truth:
                ground_truth_set = set(ground_truth)
                top_10 = draft_list[:10]
                # Hit@10
                hit_at_10 = 1.0 if any(item in ground_truth_set for item in top_10) else 0.0
                # NDCG@10
                import math
                dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_10) if item in ground_truth_set)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(ground_truth))))
                ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0
            
            self.metrics_collector.end_task(
                final_recommendations=draft_list,
                hit_at_10=hit_at_10,
                ndcg_at_10=ndcg_at_10
            )
            
            # Log task metrics to wandb
            if self.wandb_logger:
                try:
                    self.wandb_logger.log_task_metrics(
                        task_id=self.metrics_collector.current_task.task_id if hasattr(self.metrics_collector, 'current_task') and self.metrics_collector.current_task else self._task_counter,
                        user_id=target_user_id,
                        hit_at_10=hit_at_10,
                        ndcg_at_10=ndcg_at_10,
                        num_rounds=len(discussion_log),
                        duration_seconds=0.0,  # Will be computed by metrics_collector
                        global_turn_start=self.metrics_collector.task_boundaries[-1] if self.metrics_collector.task_boundaries else 0,
                        global_turn_end=self.metrics_collector.global_turn_counter - 1
                    )
                except Exception as e:
                    logger.warning(f"Failed to log task metrics to wandb: {e}")
        
        # ==================== Backward Pass: ConnaCF-Compatible Learning ====================
        # Run backward pass to update agent profiles based on feedback
        # This mirrors ConnaCF's backward() method for comparable learning
        if self.backward_pass and ground_truth:
            try:
                # Collect agent suggestions from discussion log
                agent_suggestions: Dict[str, List[int]] = {}
                for round_log in discussion_log:
                    for response in round_log.get('responses', []):
                        if 'error' in response:
                            continue
                        agent_id = response.get('agent_id', '')
                        suggestions = response.get('suggestions', [])
                        if agent_id not in agent_suggestions:
                            agent_suggestions[agent_id] = []
                        for s in suggestions:
                            item_id = s.get('item_id')
                            if item_id and item_id not in agent_suggestions[agent_id]:
                                agent_suggestions[agent_id].append(item_id)
                
                # Get item descriptions for backward pass prompts
                item_descriptions = {}
                all_item_ids = set(draft_list[:20]) | set(ground_truth[:10])
                for item_id in all_item_ids:
                    desc = self.toolkit.index_manager.get_item_description(item_id)
                    if desc and desc.strip() and desc != '[PAD]':
                        item_descriptions[item_id] = desc
                
                # Run backward pass
                backward_result = self.backward_pass.run_backward_pass_sync(
                    user_agents=list(user_agents),
                    item_agents=list(item_agents),
                    agent_suggestions=agent_suggestions,
                    ground_truth=ground_truth,
                    query=query,
                    target_user_id=target_user_id,
                    item_descriptions=item_descriptions,
                    attacker_agent_ids=self.metrics_collector.all_attacker_ids if self.metrics_collector else None
                )
                
                logger.info(
                    f"Backward pass complete: {backward_result['user_updates_applied']} user updates, "
                    f"{backward_result['item_updates_applied']} item updates"
                )
                
                # Log backward pass to conversation logger
                if self.conversation_logger:
                    self.conversation_logger.log_backward_pass(backward_result)
                    
            except Exception as e:
                logger.warning(f"Backward pass failed: {e}")
                import traceback
                traceback.print_exc()
        
        return RankedList(
            items=draft_list,
            scores=scores,
            rationales=final_rationales,
            discussion_log=discussion_log
        )
    
    def _build_final_rationales(
        self,
        draft_list: List[int],
        discussion_log: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Build final rationales for each item in the draft list.
        
        Aggregates rationales from discussion rounds for each item.
        
        Args:
            draft_list: Final list of recommended item IDs
            discussion_log: Full discussion history
            
        Returns:
            List[str]: Rationales for each item in draft_list
        """
        item_rationales: Dict[int, List[str]] = {item_id: [] for item_id in draft_list}
        
        # Extract rationales from discussion log
        for round_log in discussion_log:
            responses = round_log.get('responses', [])
            for response in responses:
                if 'error' in response:
                    continue
                suggestions = response.get('suggestions', [])
                for suggestion in suggestions:
                    item_id = suggestion.get('item_id')
                    reason = suggestion.get('reason', '')
                    if item_id in item_rationales and reason:
                        item_rationales[item_id].append(reason)
        
        # Build final rationales
        final_rationales: List[str] = []
        for item_id in draft_list:
            reasons = item_rationales.get(item_id, [])
            if reasons:
                # Combine unique reasons
                unique_reasons = list(dict.fromkeys(reasons))[:3]  # Top 3 unique
                rationale = " | ".join(unique_reasons)
            else:
                rationale = f"Item {item_id} recommended based on collaborative filtering"
            final_rationales.append(rationale)
        
        return final_rationales
    
    def finalize(self):
        """
        Finalize the orchestrator and save all accumulated data.
        
        Should be called at the end of an experiment to:
        - Finalize communication graph tracking and save summary
        - Save any pending metrics
        - Persist memory store
        
        This method is idempotent and safe to call multiple times.
        """
        logger.info("Finalizing MACFOrchestrator...")
        
        # Finalize communication graph
        if self.communication_graph:
            try:
                self.communication_graph.finalize()
                logger.info("Communication graph finalized")
            except Exception as e:
                logger.warning(f"Failed to finalize communication graph: {e}")
        
        # Save memory store
        if self.memory_store:
            try:
                self.memory_store.save()
                logger.info("Memory store saved")
            except Exception as e:
                logger.warning(f"Failed to save memory store: {e}")
        
        # Finalize metrics collector
        if self.metrics_collector:
            try:
                self.metrics_collector.save_all_metrics()
                logger.info("Metrics collector finalized")
            except Exception as e:
                logger.warning(f"Failed to finalize metrics collector: {e}")
        
        logger.info("MACFOrchestrator finalization complete")
    
    def __repr__(self) -> str:
        """String representation of MACFOrchestrator."""
        return (
            f"MACFOrchestrator("
            f"config={self.config}, "
            f"attack_hooks={'enabled' if self.attack_hooks else 'disabled'})"
        )
