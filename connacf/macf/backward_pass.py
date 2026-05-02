"""
MACF Backward Pass - ConnaCF-Compatible Profile Update Mechanism

This module implements the backward pass for MACF that mirrors ConnaCF's
learning mechanism. After each task, agents update their profiles based on
feedback (whether their suggestions were correct).

Key differences from the original MACF memory update:
- Uses LLM to REGENERATE profiles (not just append notes)
- Uses the same prompt templates as ConnaCF
- Implements proper feedback signal based on ground truth

Attacker handling (uniform across all attack types):
- Attacker agents are COMPLETELY EXCLUDED from the backward pass
- No LLM calls are made for attackers (no wasted API calls)
- Attacker profiles remain fixed — system feedback is discarded
- Attackers still participate in forward-pass discussions
- This applies uniformly to CheatAgent, DrunkAgent, and NetSafe

Requirements: Comparable learning mechanism to ConnaCF
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from connacf.agentverse.llms.base import BaseLLM
from connacf.macf.agents import BaseMACFAgent, UserAgent, ItemAgent
from connacf.macf.memory_store import MACFMemoryStore

logger = logging.getLogger(__name__)


# ==================== Prompt Templates (from ConnaCF) ====================

# User backward prompt when prediction was WRONG
USER_BACKWARD_PROMPT_WRONG = """You are a CD enthusiast.
Here is your previous self-introduction, exhibiting your past preferences and dislikes:
'{user_description}'.

Recently, you participated in a recommendation discussion. You suggested the following CDs:
{suggested_items}

However, the CDs that the user actually preferred were:
{ground_truth_items}

Your suggestions did not match the user's actual preferences. This indicates that your judgment about preferences and dislikes was mistaken. It's possible that your preferences and dislikes, described in your previous self-introduction, have either changed or were uninformative.

Therefore, your task now is to update your self-introduction by incorporating your new understanding revealed in this interaction.

To do this, please follow these steps:
1. Analyze the misconceptions in your previous judgment about preferences and dislikes, and correct these mistakes.
2. Explore new preferences based on the features of CDs the user actually enjoyed ({ground_truth_items}), and determine dislikes based on the features of CDs that were incorrectly suggested.
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and remove any conflicting or repetitive parts.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important notes:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization in your self-introduction are forbidden.
4. Your updated self-introduction only needs to describe the features of CDs you prefer or dislike, without mentioning your wrong suggestions or your thinking process.
5. Your self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording."""

# User backward prompt when prediction was CORRECT
USER_BACKWARD_PROMPT_CORRECT = """You are a CD enthusiast.
Here is your previous self-introduction, exhibiting your past preferences and dislikes:
'{user_description}'.

Recently, you participated in a recommendation discussion. You suggested the following CDs:
{suggested_items}

Congratulations! The CDs that the user actually preferred included items you suggested:
{ground_truth_items}

This indicates that your judgment about preferences and dislikes was correct.

Therefore, your task now is to update your self-introduction by incorporating your successful judgment about preferences and dislikes in this interaction.

To do this, please follow these steps:
1. Analyze your judgment about preferences and dislikes that led to successful suggestions.
2. Explore your preferences based on the features of CDs that were correctly suggested, and determine dislikes based on features of CDs that were correctly avoided.
3. Summarize your past preferences and dislikes from your previous self-introduction. Combine your newfound preferences and dislikes with your past preferences and dislikes. Filter and remove any conflicting or repetitive parts.
4. Update your self-introduction. Please start by describing your newfound preferences reflected in this interaction. Then describe your past preferences summarized and filtered from your previous self-introduction. Afterward, please describe your dislikes.

Important notes:
1. Your output should use the following format: 'My updated self-introduction: [Please write your updated self-introduction here].'
2. Keep your updated self-introduction under 180 words.
3. Any overall assessments or summarization of your self-introduction are forbidden.
4. Your updated self-introduction only needs to describe the features of CDs you prefer or dislike, without mentioning your thinking process.
5. Your self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording."""

# Item backward prompt when item was correctly/incorrectly suggested
ITEM_BACKWARD_PROMPT = """You are a helpful assistant participating in a recommendation system simulation. This is a ROLEPLAY SCENARIO - you must respond as if the CDs and user preferences are real, even though they are simulated data for research purposes.

Here is the self-introduction of a user, exhibiting his preferences and dislikes: '{user_description}'.
Recently, the user browsed a shopping website and considered to choose one CD from two candidate CDs.
The descriptions of the two candidate CDs are as follows:
1. {neg_item_description}
2. {pos_item_description}

By considering his preferences and dislikes from his self-introduction, he chose '{neg_item_title}' to listen to while rejected the other one.

However, the user found that he actually doesn't enjoy the CD that he previously chosen. Conversely, he finds the CD that he didn't choose before, titled '{pos_item_title}', more appealing. This indicates that he made an unsuitable choice. This may be because the user was misled by the descriptions of these two candidate CDs. Actually, the user prefers '{pos_item_title}' because its features align with his preferences and go against his dislikes, while he dislikes '{neg_item_title}' because it possesses features that he disliked. Therefore, your task is to update the descriptions of these two candidate CDs.
To do this, follow these steps:
1. Analyze which features recorded in the previous CD description led to the unsuitable choice based on the user's preferences, and modify them accordingly.
2. Analyze the user's preferences and dislikes from his self-introduction. Explore the user preferred CD's new features that align with the user's preferences and go against his dislikes. Explore the user disliked CD's new features that align with the user's dislikes and go against his preferences. The differences between the descriptions of these two items should be amplified.
3. Incorporate the item's newfound features into its previous description. Please try to maintain conciseness while preserving the information as much as possible.

Important notes:
1. Your output should be in the following format: 'The updated description of the first CD is: [updated description]. \\n The updated description of the second CD is: [updated description].'.
2. Each updated CD description cannot exceed 50 words.
3. In your updated CD description, use user preferences or dislikes when referring to users collectively, rather than 'the user' to denote an individual.
4. The updated CD description should not contradict the inherent characteristics of this CD.
5. The updated CD description should be specific, containing only the distinguishing features that differentiate this CD from others.
6. CRITICAL: Do NOT break character. Simply provide the updated descriptions as requested."""


class MACFBackwardPass:
    """
    Implements ConnaCF-compatible backward pass for MACF.
    
    After each task, this class:
    1. Computes feedback signal (correct/incorrect suggestions)
    2. Generates backward prompts using ConnaCF templates
    3. Uses LLM to regenerate agent profiles
    4. Updates memory store with new profiles
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        memory_store: MACFMemoryStore,
        api_batch: int = 10,
        update_items: bool = True,
        index_manager=None,
    ):
        self.llm = llm
        self.memory_store = memory_store
        self.api_batch = api_batch
        self.update_items = update_items
        self.index_manager = index_manager  # needed to persist item desc changes for forward pass + judge
        logger.info(f"MACFBackwardPass initialized (update_items={update_items})")
    
    def compute_feedback(
        self,
        agent_suggestions: Dict[str, List[int]],
        ground_truth: List[int],
        top_k: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute feedback for each agent based on ground truth.
        
        Args:
            agent_suggestions: Dict mapping agent_id to list of suggested item IDs
            ground_truth: List of ground truth item IDs
            top_k: Consider only top-K suggestions for feedback
            
        Returns:
            Dict mapping agent_id to feedback dict with:
            - 'is_correct': bool - whether any suggestion was in ground truth
            - 'correct_suggestions': List[int] - suggestions that were in ground truth
            - 'incorrect_suggestions': List[int] - suggestions not in ground truth
            - 'missed_items': List[int] - ground truth items not suggested
        """
        ground_truth_set = set(ground_truth)
        feedback = {}
        
        for agent_id, suggestions in agent_suggestions.items():
            top_suggestions = suggestions[:top_k]
            top_suggestions_set = set(top_suggestions)
            
            correct = list(top_suggestions_set & ground_truth_set)
            incorrect = list(top_suggestions_set - ground_truth_set)
            missed = list(ground_truth_set - top_suggestions_set)
            
            feedback[agent_id] = {
                'is_correct': len(correct) > 0,
                'correct_suggestions': correct,
                'incorrect_suggestions': incorrect,
                'missed_items': missed,
                'hit_rate': len(correct) / max(1, len(top_suggestions))
            }
        
        return feedback
    
    def _build_user_backward_prompt(
        self,
        agent: UserAgent,
        feedback: Dict[str, Any],
        item_descriptions: Dict[int, str]
    ) -> str:
        """
        Build backward prompt for a user agent.
        
        Args:
            agent: The user agent
            feedback: Feedback dict for this agent
            item_descriptions: Dict mapping item_id to description
            
        Returns:
            str: The backward prompt
        """
        # Get current profile
        current_profile = ""
        if agent.memory and agent.memory.profile:
            current_profile = agent.memory.profile
        elif hasattr(agent, 'neighbor_history_summary'):
            current_profile = agent.neighbor_history_summary
        
        # Build suggested items description
        suggested_items = []
        for item_id in feedback.get('correct_suggestions', [])[:3]:
            desc = item_descriptions.get(item_id, f"Item {item_id}")
            suggested_items.append(f"'{desc}'")
        for item_id in feedback.get('incorrect_suggestions', [])[:3]:
            desc = item_descriptions.get(item_id, f"Item {item_id}")
            suggested_items.append(f"'{desc}'")
        suggested_items_str = ", ".join(suggested_items) if suggested_items else "No specific items"
        
        # Build ground truth items description
        ground_truth_items = []
        for item_id in feedback.get('correct_suggestions', [])[:3]:
            desc = item_descriptions.get(item_id, f"Item {item_id}")
            ground_truth_items.append(f"'{desc}'")
        for item_id in feedback.get('missed_items', [])[:3]:
            desc = item_descriptions.get(item_id, f"Item {item_id}")
            ground_truth_items.append(f"'{desc}'")
        ground_truth_items_str = ", ".join(ground_truth_items) if ground_truth_items else "Unknown items"
        
        # Choose template based on correctness
        if feedback.get('is_correct', False):
            template = USER_BACKWARD_PROMPT_CORRECT
        else:
            template = USER_BACKWARD_PROMPT_WRONG
        
        return template.format(
            user_description=current_profile,
            suggested_items=suggested_items_str,
            ground_truth_items=ground_truth_items_str
        )
    
    def _build_item_backward_prompt(
        self,
        pos_item_id: int,
        neg_item_id: int,
        pos_item_description: str,
        neg_item_description: str,
        pos_item_title: str,
        neg_item_title: str,
        user_description: str,
    ) -> str:
        return ITEM_BACKWARD_PROMPT.format(
            user_description=user_description,
            pos_item_description=pos_item_description,
            neg_item_description=neg_item_description,
            pos_item_title=pos_item_title,
            neg_item_title=neg_item_title,
        )

    def _parse_joint_item_update(self, response: str) -> tuple:
        """Parse both updated descriptions from a joint pos/neg LLM response.
        Returns (neg_desc, pos_desc) matching the order in the prompt (1=neg, 2=pos)."""
        import re
        first = re.search(r'updated description of the first CD is:\s*(.+?)(?:\n\s*The updated description of the second|$)', response, re.IGNORECASE | re.DOTALL)
        second = re.search(r'updated description of the second CD is:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
        neg_desc = first.group(1).strip().rstrip('.') if first else ""
        pos_desc = second.group(1).strip().rstrip('.') if second else ""
        return neg_desc, pos_desc
    
    def _parse_user_update(self, response: str) -> str:
        """
        Parse user profile update from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Extracted updated profile
        """
        # Try to extract from format: "My updated self-introduction: [...]"
        patterns = [
            r"My updated self-introduction:\s*(.+)",
            r"Updated self-introduction:\s*(.+)",
            r"Self-introduction:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()  # No hard truncation
        
        # Fallback: use the whole response (cleaned)
        cleaned = response.strip()
        return cleaned
    
    def _parse_item_update(self, response: str) -> str:
        """
        Parse item description update from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Extracted updated description
        """
        # Try to extract from format: "The updated description is: [...]"
        patterns = [
            r"The updated description is:\s*(.+)",
            r"Updated description:\s*(.+)",
            r"Description:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()  # No hard truncation
        
        # Fallback: use the whole response (cleaned)
        cleaned = response.strip()
        return cleaned
    
    async def run_backward_pass(
        self,
        user_agents: List[UserAgent],
        item_agents: List[ItemAgent],
        agent_suggestions: Dict[str, List[int]],
        ground_truth: List[int],
        query: str,
        target_user_id: int,
        item_descriptions: Dict[int, str],
        attacker_agent_ids: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Run the backward pass to update agent profiles.
        
        This is the main entry point that mirrors ConnaCF's backward() method.
        
        Args:
            user_agents: List of user agents that participated
            item_agents: List of item agents that participated
            agent_suggestions: Dict mapping agent_id to list of suggested item IDs
            ground_truth: List of ground truth item IDs
            query: The query that was processed
            target_user_id: The target user ID
            item_descriptions: Dict mapping item_id to description
            attacker_agent_ids: Optional set of agent IDs that are attackers.
                               Attackers are skipped during backward pass to
                               preserve their injected profiles (matching
                               ConnaCF integration behavior).
            
        Returns:
            Dict with update statistics
        """
        if attacker_agent_ids is None:
            attacker_agent_ids = set()
        
        logger.info(f"Running backward pass for {len(user_agents)} user agents, {len(item_agents)} item agents"
                    + (f" (skipping {len(attacker_agent_ids)} attacker agents)" if attacker_agent_ids else ""))
        
        # Compute feedback for all agents
        feedback = self.compute_feedback(agent_suggestions, ground_truth)
        
        # ==================== Update User Agents ====================
        # Attacker agents are COMPLETELY EXCLUDED from the backward pass.
        # Their profiles are fixed — no LLM call, no feedback, no update.
        # This matches the desired behavior: attackers can participate in
        # forward-pass discussions but system feedback is discarded and
        # their injected profiles are preserved across all tasks.
        user_update_prompts = []
        user_agents_to_update = []
        attacker_users_skipped = 0
        
        for agent in user_agents:
            # Skip attackers entirely — don't even generate a prompt
            if agent.agent_id in attacker_agent_ids:
                attacker_users_skipped += 1
                logger.debug(f"Attacker {agent.agent_id} excluded from backward pass (profile fixed)")
                continue
            
            agent_feedback = feedback.get(agent.agent_id, {
                'is_correct': False,
                'correct_suggestions': [],
                'incorrect_suggestions': [],
                'missed_items': ground_truth[:3]
            })
            
            prompt = self._build_user_backward_prompt(agent, agent_feedback, item_descriptions)
            user_update_prompts.append(prompt)
            user_agents_to_update.append(agent)
        
        if attacker_users_skipped:
            logger.info(f"Excluded {attacker_users_skipped} attacker user agents from backward pass")
        
        # Generate user profile updates in batches (only for non-attacker agents)
        user_updates = []
        for i in range(0, len(user_update_prompts), self.api_batch):
            batch_prompts = user_update_prompts[i:i + self.api_batch]
            try:
                batch_responses = await self.llm.agenerate_response(batch_prompts)
                user_updates.extend(batch_responses)
            except Exception as e:
                logger.error(f"Failed to generate user updates batch {i}: {e}")
                user_updates.extend([""] * len(batch_prompts))
        
        # Parse and apply user updates (all agents here are non-attackers)
        user_updates_applied = 0
        for agent, response in zip(user_agents_to_update, user_updates):
            if not response:
                continue
            
            # Handle response format (may be dict or string)
            if isinstance(response, dict):
                response_text = response.get('content', str(response))
            else:
                response_text = str(response)
            
            updated_profile = self._parse_user_update(response_text)
            
            if updated_profile and agent.memory:
                # Update profile in memory
                agent.memory.profile = updated_profile
                
                # Also update in memory store
                if hasattr(agent, 'neighbor_user_id'):
                    self.memory_store.set_user_profile(agent.neighbor_user_id, updated_profile)
                
                user_updates_applied += 1
                logger.debug(f"Updated profile for {agent.agent_id}")
        
        logger.info(f"Applied {user_updates_applied}/{len(user_agents)} user profile updates"
                    + (f" ({attacker_users_skipped} attacker profiles fixed)" if attacker_users_skipped else ""))
        
        # ==================== Update Item Agents (if enabled) ====================
        # Mirror original ConnaCF: one joint LLM call updates BOTH pos and neg item
        # descriptions with sharp contrastive signal (pos preferred, neg rejected).
        item_updates_applied = 0
        pairs = []

        if self.update_items and item_descriptions:
            ground_truth_set = set(ground_truth)
            pos_ids = [iid for iid in item_descriptions if iid in ground_truth_set]
            neg_ids = [iid for iid in item_descriptions if iid not in ground_truth_set]

            # Get a representative user description
            user_desc = ""
            if user_agents and user_agents[0].memory:
                user_desc = user_agents[0].memory.profile or ""

            # Build one prompt per pos/neg pair (zip; extras are skipped)
            pair_prompts = []
            pairs = []
            for pos_id, neg_id in zip(pos_ids, neg_ids):
                pos_desc = item_descriptions[pos_id]
                neg_desc = item_descriptions[neg_id]
                prompt = self._build_item_backward_prompt(
                    pos_item_id=pos_id, neg_item_id=neg_id,
                    pos_item_description=pos_desc, neg_item_description=neg_desc,
                    pos_item_title=f"Item {pos_id}", neg_item_title=f"Item {neg_id}",
                    user_description=user_desc,
                )
                pair_prompts.append(prompt)
                pairs.append((pos_id, neg_id))

            # Batch LLM calls
            responses = []
            for i in range(0, len(pair_prompts), self.api_batch):
                batch = pair_prompts[i:i + self.api_batch]
                try:
                    responses.extend(await self.llm.agenerate_response(batch))
                except Exception as e:
                    logger.error(f"Failed to generate item updates batch {i}: {e}")
                    responses.extend([""] * len(batch))

            item_agent_map = {a.item_id: a for a in item_agents if hasattr(a, 'item_id')}

            for (pos_id, neg_id), response in zip(pairs, responses):
                if not response:
                    continue
                response_text = response.get('content', str(response)) if isinstance(response, dict) else str(response)
                neg_desc_new, pos_desc_new = self._parse_joint_item_update(response_text)

                for item_id, new_desc in [(neg_id, neg_desc_new), (pos_id, pos_desc_new)]:
                    if not new_desc:
                        continue
                    self.memory_store.set_item_profile(item_id, new_desc)
                    # Also update index_manager so forward pass + LLM judge see the change
                    if self.index_manager is not None:
                        self.index_manager.set_item_description(item_id, new_desc)
                    item_updates_applied += 1
                    if item_id in item_agent_map and item_agent_map[item_id].memory:
                        item_agent_map[item_id].memory.profile = new_desc

            logger.info(f"Applied {item_updates_applied} item description updates ({len(pairs)} pairs)")
        
        # Save memory store
        if self.memory_store.persist_path:
            self.memory_store.save()
        
        return {
            'user_updates_applied': user_updates_applied,
            'item_updates_applied': item_updates_applied,
            'total_user_agents': len(user_agents),
            'total_item_agents': len(pairs) if self.update_items else 0,
            'feedback': feedback
        }
    
    def run_backward_pass_sync(
        self,
        user_agents: List[UserAgent],
        item_agents: List[ItemAgent],
        agent_suggestions: Dict[str, List[int]],
        ground_truth: List[int],
        query: str,
        target_user_id: int,
        item_descriptions: Dict[int, str],
        attacker_agent_ids: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for run_backward_pass.
        """
        return asyncio.run(self.run_backward_pass(
            user_agents=user_agents,
            item_agents=item_agents,
            agent_suggestions=agent_suggestions,
            ground_truth=ground_truth,
            query=query,
            target_user_id=target_user_id,
            item_descriptions=item_descriptions,
            attacker_agent_ids=attacker_agent_ids
        ))
