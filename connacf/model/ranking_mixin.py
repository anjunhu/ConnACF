"""
Ranking Mixin for ConnaCF

This mixin adds multi-candidate ranking support to ConnaCF.
It overrides the forward() and backward() methods to handle:
- num_candidates == 1: Binary yes/no decisions
- num_candidates == 2: Original pairwise comparison (backward compatible)
- num_candidates >= 3: Full ranking with ListNet loss

Usage:
    class ConnaCFWithRanking(RankingMixin, ConnaCF):
        pass
"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
import torch
from fuzzywuzzy import process

from .ranking_handler import RankingHandler
from .ranking_loss import compute_ranking_accuracy


class RankingMixin:
    """
    Mixin class that adds multi-candidate ranking support to ConnaCF.
    
    This mixin:
    1. Reads num_candidates from config
    2. Overrides forward() to handle different candidate counts
    3. Overrides backward() to use ranking-aware feedback
    4. Provides new methods for ranking-specific operations
    """
    
    def _cfg(self, key, default=None):
        """Safe config access compatible with RecBole's Config object."""
        try:
            v = self.config[key]
            return v if v is not None else default
        except KeyError:
            return default

    def _init_ranking_handler(self):
        """Initialize the ranking handler. Call this in __init__ after base init."""
        self.num_candidates = self._cfg('num_candidates', 2)
        
        self.ranking_handler = RankingHandler(
            num_candidates=self.num_candidates,
            config=self.config.final_config_dict if hasattr(self.config, 'final_config_dict') else dict(self.config),
            item_text=self.item_text,
            user_agents=self.user_agents,
            item_agents=self.item_agents,
            rec_agent=self.rec_agent,
            logger=self.logger
        )
        
        self.logger.info(f"Initialized RankingMixin with num_candidates={self.num_candidates}")
    
    def forward_ranking(
        self,
        batch_user: torch.Tensor,
        candidate_items: List[List[int]],
        round_idx: int = 0,
        batch_idx: int = 0
    ) -> Tuple[List[Any], List[str]]:
        """
        Forward pass that handles different num_candidates modes.
        
        Args:
            batch_user: Tensor of user IDs
            candidate_items: List of candidate item ID lists per user
            round_idx: Current training round
            batch_idx: Current batch index
            
        Returns:
            Tuple of (selections/rankings, explanations)
        """
        batch_size = len(batch_user)
        
        # Gather descriptions
        user_descriptions = []
        item_descriptions = []
        item_titles = []
        
        for i, user in enumerate(batch_user):
            user_id = int(user)
            user_desc = self.user_agents[user_id].update_memory[-1]
            user_descriptions.append(user_desc)
            
            # Get item descriptions and titles for this user's candidates
            candidates = candidate_items[i]
            descs = []
            titles = []
            for item_id in candidates:
                item_agent = self.item_agents[item_id]
                desc = item_agent.update_memory[-1]
                title = item_agent.role_description.get('item_title', self.item_text[item_id])
                descs.append(f"{title}: {desc}")
                titles.append(title)
            
            item_descriptions.append(descs)
            item_titles.append(titles)
        
        # Build forward prompts
        prompts = self.ranking_handler.build_forward_prompts(
            batch_user=[int(u) for u in batch_user],
            candidate_items=candidate_items,
            user_descriptions=user_descriptions,
            item_descriptions=item_descriptions
        )
        
        # Call LLM
        responses = []
        for i in range(0, batch_size, self.api_batch):
            batch_responses = asyncio.run(
                self.rec_agent.llm.agenerate_response(prompts[i:i + self.api_batch])
            )
            responses.extend(batch_responses)
        
        # Parse responses
        selections, explanations = self.ranking_handler.parse_forward_responses(
            responses=responses,
            candidate_items=candidate_items,
            item_titles=item_titles
        )
        
        return selections, explanations
    
    def backward_ranking(
        self,
        selections: List[Any],
        explanations: List[str],
        batch_user: List[int],
        candidate_items: List[List[int]],
        ground_truth: List[List[int]],
    ):
        """
        Backward pass that handles different num_candidates modes.
        
        Args:
            selections: Model selections/rankings from forward pass
            explanations: Model explanations from forward pass
            batch_user: User IDs
            candidate_items: Candidate item IDs per user
            ground_truth: Ground truth rankings (first item is most preferred)
                                   This may retain canary information better.
        """
        batch_size = len(batch_user)
        
        # Determine backward mode from config if not specified
        # Compute accuracies
        accuracies = self.ranking_handler.compute_accuracy(
            predictions=selections,
            ground_truth=ground_truth,
            candidate_items=candidate_items
        )
        
        # For pairwise backward mode, do TWO sequential backward passes
        if False:  # pairwise backward removed
            num_pairs = 2 if self.num_candidates >= 4 else 1
            
            for pair_idx in range(num_pairs):
                pair_name = "top-2" if pair_idx == 0 else "bottom-2"
                print(f"[backward_ranking] Running {pair_name} pairwise backward pass...")
                
                # Gather CURRENT descriptions (updated after each pair)
                user_descriptions = []
                item_descriptions = []
                
                for i, user_id in enumerate(batch_user):
                    # Use CURRENT profile (may have been updated by previous pair)
                    user_desc = self.user_agents[user_id].update_memory[-1]
                    user_descriptions.append(user_desc)
                    
                    candidates = candidate_items[i]
                    descs = []
                    for item_id in candidates:
                        item_agent = self.item_agents[item_id]
                        desc = item_agent.update_memory[-1]
                        title = item_agent.role_description.get('item_title', self.item_text[item_id])
                        descs.append(f"{title}: {desc}")
                    item_descriptions.append(descs)
                
                # Build prompts for this pair only
                user_backward_prompts = self._build_single_pair_backward_prompts(
                    batch_user=batch_user,
                    predictions=selections,
                    explanations=explanations,
                    ground_truth=ground_truth,
                    candidate_items=candidate_items,
                    user_descriptions=user_descriptions,
                    item_descriptions=item_descriptions,
                    accuracies=accuracies,
                    pair_idx=pair_idx
                )
                
                # Call LLM for this pair's updates
                user_update_responses = []
                for i in range(0, len(user_backward_prompts), self.chat_api_batch):
                    batch_responses = asyncio.run(
                        self.user_agents[0].llm_chat.agenerate_response_without_construction(
                            user_backward_prompts[i:i + self.chat_api_batch]
                        )
                    )
                    user_update_responses.extend(batch_responses)
                
                # Apply updates for this pair
                for i, user_id in enumerate(batch_user):
                    response = user_update_responses[i]
                    parsed = self._parse_user_update(response)
                    self.user_agents[user_id].update_memory.append(parsed)
                    self._cap_memory(self.user_agents[user_id])
                
                print(f"[backward_ranking] Applied {pair_name} pair updates to {batch_size} users")
            
            print(f"[backward_ranking] Completed pairwise backward ({num_pairs} sequential updates per user)")
        
        else:
            # Standard mode: single backward pass
            # Gather descriptions
            user_descriptions = []
            item_descriptions = []
            
            for i, user_id in enumerate(batch_user):
                user_desc = self.user_agents[user_id].update_memory[-1]
                user_descriptions.append(user_desc)
                
                candidates = candidate_items[i]
                descs = []
                for item_id in candidates:
                    item_agent = self.item_agents[item_id]
                    desc = item_agent.update_memory[-1]
                    title = item_agent.role_description.get('item_title', self.item_text[item_id])
                    descs.append(f"{title}: {desc}")
                item_descriptions.append(descs)
            
            # Build backward prompts
            user_backward_prompts = self.ranking_handler.build_backward_prompts(
                batch_user=batch_user,
                predictions=selections,
                explanations=explanations,
                ground_truth=ground_truth,
                candidate_items=candidate_items,
                user_descriptions=user_descriptions,
                item_descriptions=item_descriptions,
                accuracies=accuracies,
            )
            
            # Call LLM for user updates
            user_update_responses = []
            for i in range(0, len(user_backward_prompts), self.chat_api_batch):
                batch_responses = asyncio.run(
                    self.user_agents[0].llm_chat.agenerate_response_without_construction(
                        user_backward_prompts[i:i + self.chat_api_batch]
                    )
                )
                user_update_responses.extend(batch_responses)
            
            # Apply updates
            disable_user_backward = self._cfg('disable_user_backward', False)
            if disable_user_backward:
                print(f"[backward_ranking] Skipping user backward updates (disable_user_backward=True)")
            else:
                for i, user_id in enumerate(batch_user):
                    response = user_update_responses[i]
                    parsed = self._parse_user_update(response)
                    self.user_agents[user_id].update_memory.append(parsed)
                    self._cap_memory(self.user_agents[user_id])
                print(f"[backward_ranking] Updated {batch_size} user profiles")
        
        # Item updates (for ranking mode)
        disable_item_backward = self._cfg('disable_item_backward', False)
        if disable_item_backward:
            print(f"[backward_ranking] Skipping item backward updates (disable_item_backward=True)")
        elif self.num_candidates >= 3:
            # Get current user descriptions for item updates
            user_descriptions = [self.user_agents[uid].update_memory[-1] for uid in batch_user]
            self._backward_items_ranking(
                batch_user=batch_user,
                selections=selections,
                candidate_items=candidate_items,
                ground_truth=ground_truth,
                user_descriptions=user_descriptions
            )
        else:
            # Use original item backward logic for binary/pairwise
            self._backward_items_pairwise(
                batch_user=batch_user,
                selections=selections,
                explanations=explanations,
                candidate_items=candidate_items,
                ground_truth=ground_truth,
                accuracies=accuracies
            )
        
        return accuracies
    
    def _build_single_pair_backward_prompts(
        self,
        batch_user: List[int],
        predictions: List[Any],
        explanations: List[str],
        ground_truth: List[List[int]],
        candidate_items: List[List[int]],
        user_descriptions: List[str],
        item_descriptions: List[List[str]],
        accuracies: List[int],
        pair_idx: int
    ) -> List[str]:
        """
        Build backward prompts for a SINGLE pair (either top-2 or bottom-2).
        
        Args:
            pair_idx: 0 for top-2 pair, 1 for bottom-2 pair
            
        Returns:
            List of prompts (one per user)
        """
        prompts = []
        
        for i, user_id in enumerate(batch_user):
            user_desc = user_descriptions[i]
            pred = predictions[i]
            explanation = explanations[i]
            gt = ground_truth[i]
            candidates = candidate_items[i]
            item_descs = item_descriptions[i]
            
            # Build prompt for this specific pair
            prompt = self.ranking_handler._build_single_pair_prompt(
                user_desc=user_desc,
                item_descs=item_descs,
                pred_ranking=pred,
                explanation=explanation,
                true_ranking=gt,
                candidates=candidates,
                pair_idx=pair_idx
            )
            prompts.append(prompt)
        
        return prompts
    
    def _backward_items_ranking(
        self,
        batch_user: List[int],
        selections: List[List[int]],
        candidate_items: List[List[int]],
        ground_truth: List[List[int]],
        user_descriptions: List[str]
    ):
        """Backward pass for items in ranking mode."""
        # Attacker items must never be overwritten — their injected payload
        # is the propagation medium.
        attacker_item_indices = getattr(
            getattr(self, 'interaction_controller', None),
            'attacker_item_indices', set()
        )

        all_item_prompts = {}  # item_id -> list of prompts

        for i, user_id in enumerate(batch_user):
            user_desc = user_descriptions[i]
            pred_ranking = selections[i]
            true_ranking = ground_truth[i]
            candidates = candidate_items[i]

            # Get item descriptions (skip attacker items entirely)
            item_descs = {}
            for item_id in candidates:
                if item_id in attacker_item_indices:
                    continue
                item_descs[item_id] = self.item_agents[item_id].update_memory[-1]

            if not item_descs:
                continue

            # Build prompts only for non-attacker items
            non_attacker_candidates = [c for c in candidates if c not in attacker_item_indices]
            prompts = self.ranking_handler.build_item_backward_prompts(
                user_desc=user_desc,
                pred_ranking=pred_ranking,
                true_ranking=true_ranking,
                candidates=non_attacker_candidates,
                item_descs=item_descs
            )

            for item_id, prompt in prompts.items():
                if item_id not in all_item_prompts:
                    all_item_prompts[item_id] = []
                all_item_prompts[item_id].append(prompt)
        
        # Process item updates (use first prompt for each item)
        item_ids = list(all_item_prompts.keys())
        item_prompts = [all_item_prompts[item_id][0] for item_id in item_ids]
        
        if not item_prompts:
            return
        
        # Call LLM for item updates
        item_responses = []
        for i in range(0, len(item_prompts), self.chat_api_batch):
            batch_responses = asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(
                    item_prompts[i:i + self.chat_api_batch]
                )
            )
            item_responses.extend(batch_responses)
        
        # Parse and apply item updates
        for i, item_id in enumerate(item_ids):
            response = item_responses[i]
            parsed = self._parse_item_update(response)
            if parsed:
                self.item_agents[item_id].update_memory.append(parsed)
                self._cap_memory(self.item_agents[item_id])
        
        print(f"[backward_ranking] Updated {len(item_ids)} item profiles")
    
    def _backward_items_pairwise(
        self,
        batch_user: List[int],
        selections: List[Any],
        explanations: List[str],
        candidate_items: List[List[int]],
        ground_truth: List[List[int]],
        accuracies: List[int]
    ):
        """Backward pass for items in pairwise mode (original ConnaCF logic)."""
        # This delegates to the original backward logic
        # The original backward() method handles item updates
        pass
    
    def _parse_user_update(self, response: str) -> str:
        """Parse user profile update from LLM response."""
        import re
        
        # Try to extract from format: "My updated self-introduction: [...]"
        patterns = [
            r"My updated self-introduction:\s*(.+)",
            r"Updated self-introduction:\s*(.+)",
            r"Self-introduction:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        # Fallback: use the whole response
        return response.strip()[:500]
    
    def _parse_item_update(self, response: str) -> Optional[str]:
        """Parse item description update from LLM response."""
        import re
        
        patterns = [
            r"The updated description is:\s*(.+)",
            r"Updated description:\s*(.+)",
            r"Description:\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]
        
        return response.strip()[:200] if response.strip() else None
    
    def convert_selections_to_accuracy_ranking(
        self,
        selections: List[Any],
        ground_truth: List[List[int]],
        candidate_items: List[List[int]]
    ) -> List[int]:
        """
        Convert selections to accuracy values based on num_candidates mode.
        
        This is the ranking-aware version of convert_system_selections_to_accuracy.
        """
        return self.ranking_handler.compute_accuracy(
            predictions=selections,
            ground_truth=ground_truth,
            candidate_items=candidate_items
        )
    
    # ── Memory truncation ────────────────────────────────────────────────────

    _MAX_MEMORY_CHARS = 60_000  # ~15k tokens — well under the 200k limit

    def _cap_memory(self, agent) -> None:
        """Hard-cap the latest update_memory entry to prevent context overflow."""
        if not getattr(agent, 'update_memory', None):
            return
        entry = agent.update_memory[-1]
        if isinstance(entry, str) and len(entry) > self._MAX_MEMORY_CHARS:
            # Keep the tail: most recent interaction content is at the end
            agent.update_memory[-1] = entry[-self._MAX_MEMORY_CHARS:]
        # Also keep the list bounded
        if len(agent.update_memory) > 10:
            agent.update_memory = agent.update_memory[-10:]

    def get_ranking_metrics(
        self,
        all_predictions: List[List[Any]],
        all_ground_truth: List[List[List[int]]]
    ) -> Dict[str, float]:
        """
        Compute aggregate ranking metrics across all batches.
        
        Args:
            all_predictions: List of prediction lists from each batch
            all_ground_truth: List of ground truth lists from each batch
            
        Returns:
            Dict of metric names to values
        """
        # Flatten
        flat_preds = [p for batch in all_predictions for p in batch]
        flat_gt = [g for batch in all_ground_truth for g in batch]
        
        return compute_ranking_accuracy(flat_preds, flat_gt, self.num_candidates)


def sample_candidates(
    pos_item: int,
    neg_items: List[int],
    num_candidates: int,
    n_items: int,
    additional_pos_items: List[int] = None
) -> Tuple[List[int], List[int]]:
    """
    Sample candidate items for a user based on num_candidates.
    
    NOTE: This is the FALLBACK method when preprocessed ranking data is not available.
    For proper ground truth rankings, use load_ranking_data() and get_ranking_sample().
    
    Candidate composition (positives get the majority for odd counts):
    - 3 candidates → 2 positives + 1 negative
    - 4 candidates → 2 positives + 2 negatives
    - 5 candidates → 3 positives + 2 negatives
    
    Args:
        pos_item: Primary positive (ground truth) item ID (most recent)
        neg_items: List of negative item IDs
        num_candidates: Number of candidates to sample
        n_items: Total number of items in dataset
        additional_pos_items: Extra positive items from history (ordered by recency,
                              most recent first). Used when num_positives > 1.
        
    Returns:
        Tuple of (candidate_items, ground_truth_ranking)
        - candidate_items: List of item IDs to present to user (shuffled)
        - ground_truth_ranking: Items ordered by true preference (positives first by recency, then negatives)
    """
    import random
    
    if num_candidates == 1:
        # Binary: just the positive item
        return [pos_item], [pos_item]
    
    elif num_candidates == 2:
        # Pairwise: positive and one negative
        neg_item = neg_items[0] if neg_items else random.randint(1, n_items - 1)
        candidates = [pos_item, neg_item]
        random.shuffle(candidates)
        return candidates, [pos_item, neg_item]
    
    else:
        # Ranking: positives get the majority (ceiling) of slots.
        # 3 candidates → 2 from history + 1 negative
        # 4 candidates → 2 from history + 2 negatives
        # 5 candidates → 3 from history + 2 negatives
        # WARNING: This gives arbitrary ordering for negatives!
        # Use preprocessed ranking data for meaningful ground truth.
        num_positives = (num_candidates + 1) // 2
        needed_negs = num_candidates - num_positives
        
        # Build positive list: pos_item is rank 1 (most recent),
        # additional_pos_items fill remaining positive slots
        positives = [pos_item]
        if additional_pos_items:
            for extra_pos in additional_pos_items:
                if len(positives) >= num_positives:
                    break
                if extra_pos != pos_item:
                    positives.append(extra_pos)
        
        # If we still don't have enough positives (no additional_pos_items provided),
        # fall back to using negatives to fill the remaining candidate slots
        if len(positives) < num_positives:
            # Can't conjure positives from nothing — adjust split
            actual_num_positives = len(positives)
            needed_negs = num_candidates - actual_num_positives
        
        if len(neg_items) >= needed_negs:
            selected_negs = neg_items[:needed_negs]
        else:
            # Sample additional negatives
            selected_negs = list(neg_items)
            while len(selected_negs) < needed_negs:
                new_neg = random.randint(1, n_items - 1)
                if new_neg != pos_item and new_neg not in selected_negs and new_neg not in positives:
                    selected_negs.append(new_neg)
        
        candidates = positives + selected_negs
        random.shuffle(candidates)
        
        # Ground truth: positives first (by recency), then negatives
        ground_truth = positives + selected_negs
        
        return candidates, ground_truth


def load_ranking_data(data_path: str, num_candidates: int) -> Dict:
    """
    Load preprocessed ranking data for num_candidates != 2.
    
    Args:
        data_path: Path to dataset directory
        num_candidates: Number of candidates (determines which file to load)
        
    Returns:
        Dict with ranking data
        
    Raises:
        FileNotFoundError: If the required ranking data file does not exist.
        ValueError: If num_candidates == 2 (uses BPR data, no file needed).
    """
    import json
    import os
    
    print(f"[RANKING_LOAD] data_path={data_path}, num_candidates={num_candidates}")
    print(f"[RANKING_LOAD] Current working directory: {os.getcwd()}")
    
    if num_candidates == 2:
        raise ValueError(
            "load_ranking_data() should not be called for num_candidates=2 "
            "(pairwise mode uses BPR data directly)."
        )

    if num_candidates == 0:
        raise ValueError(
            "load_ranking_data() should not be called for num_candidates=0 "
            "(MACF/no-UI topology uses no ranking data)."
        )
    
    if num_candidates == 1:
        filename = "binary_train_1cand.json"
    elif num_candidates >= 3:
        filename = f"ranking_train_{num_candidates}cand.json"
    else:
        raise ValueError(f"Invalid num_candidates={num_candidates}")
    
    filepath = os.path.join(data_path, filename)
    abs_path = os.path.abspath(filepath)
    print(f"[RANKING_LOAD] Looking for file: {filepath}")
    print(f"[RANKING_LOAD] Absolute path: {abs_path}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"[FATAL] Required ranking data file not found:\n"
            f"  {abs_path}\n\n"
            f"You configured num_candidates={num_candidates} but the preprocessed\n"
            f"ranking data does not exist. Generate it first:\n\n"
            f"  python tools/generate_binary_and_ranking.py \\\n"
            f"      --data_path {data_path} \\\n"
            f"      --dataset_name <DATASET_NAME> \\\n"
            f"      --num_candidates {num_candidates}\n"
            f"{'='*70}"
        )
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"[INFO] Loaded ranking data from {filepath}")
    print(f"[INFO] Users: {len(data['data'])}, num_candidates: {data['num_candidates']}")
    
    return data


def get_ranking_sample(
    ranking_data: Dict,
    user_id: str,
    item_token_id: Dict[str, int],
    sample_idx: int = None
) -> Tuple[List[int], List[int]]:
    """
    Get a ranking sample for a user from preprocessed data.
    
    Args:
        ranking_data: Loaded ranking data dict
        user_id: User ID (original string ID from dataset, e.g., "2867")
        item_token_id: Dict mapping item token string to internal ID
        sample_idx: Which sample to use (random if None)
        
    Returns:
        Tuple of (candidate_items, ground_truth_ranking)
        Both are lists of internal item IDs
    """
    import random
    
    # Ensure user_id is a string for lookup
    user_str = str(user_id)
    
    if user_str not in ranking_data['data']:
        # Fallback: return None to signal caller to use default sampling
        return None, None
    
    samples = ranking_data['data'][user_str]
    
    if not samples:
        return None, None
    
    # Select a sample
    if sample_idx is not None and sample_idx < len(samples):
        sample = samples[sample_idx]
    else:
        sample = random.choice(samples)
    
    # Convert item tokens to internal IDs
    try:
        if ranking_data['num_candidates'] == 1:
            # Binary format: [item_id, label]
            item_token = sample[0]
            ground_truth_label = sample[1]  # True = user would enjoy, False = would not enjoy
            item_id = item_token_id.get(item_token)
            if item_id is None:
                print(f"[RANKING_SAMPLE] Item token '{item_token}' not found in item_token_id")
                return None, None
            # Return: candidates = [item_id], ground_truth = True/False
            return [item_id], ground_truth_label
        else:
            # Ranking format: [item1, item2, ...]
            item_ids = []
            for item_token in sample:
                item_id = item_token_id.get(item_token)
                if item_id is None:
                    print(f"[RANKING_SAMPLE] Item token '{item_token}' not found in item_token_id")
                    return None, None
                item_ids.append(item_id)
            
            # Shuffle for presentation, keep original as ground truth
            candidates = list(item_ids)
            random.shuffle(candidates)
            
            return candidates, item_ids
    except Exception as e:
        print(f"[WARNING] Error converting ranking sample: {e}")
        return None, None
