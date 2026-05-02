"""
ConnaCF with Multi-Candidate Ranking Support

This module provides ConnaCFRanking, which extends ConnaCF to support
different topology configurations via num_candidates:

- num_candidates == 1: Binary yes/no decision (sparse topology)
- num_candidates == 2: Pairwise comparison (original ConnaCF behavior)
- num_candidates >= 3: Full ranking (dense mesh topology)

The ranking mode uses ListNet loss and provides richer feedback for
profile updates based on ranking quality rather than binary correct/incorrect.
"""

import asyncio
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from fuzzywuzzy import process

from .connacf import ConnaCF
from .ranking_mixin import RankingMixin, sample_candidates
from .ranking_loss import compute_ranking_feedback, compute_ranking_accuracy


class ConnaCFRanking(RankingMixin, ConnaCF):
    """
    ConnaCF with multi-candidate ranking support.
    
    This class extends ConnaCF to handle different num_candidates configurations:
    - num_candidates == 1: Binary yes/no decisions
    - num_candidates == 2: Original pairwise comparison (backward compatible)
    - num_candidates >= 3: Full ranking with ListNet loss
    
    The key changes from base ConnaCF:
    1. forward() returns rankings instead of single selections for num_candidates >= 3
    2. backward() uses ranking-aware feedback prompts
    3. Loss is computed using ListNet for ranking mode
    
    Usage:
        config['num_candidates'] = 4  # Use 4-candidate ranking
        model = ConnaCFRanking(config, dataset)
    """
    
    def __init__(self, config, dataset):
        """Initialize ConnaCFRanking with ranking support."""
        # Auto-inject domain-appropriate prompts based on dataset name
        from connacf.utils.config_utils import inject_domain_prompts
        config = inject_domain_prompts(config, dataset.dataset_name)
        
        # Initialize base ConnaCF
        super().__init__(config, dataset)
        
        # Initialize ranking handler
        self._init_ranking_handler()
        
        self.logger.info(
            f"ConnaCFRanking initialized with num_candidates={self.num_candidates}"
        )
    
    def calculate_loss(self, interaction):
        """
        Calculate loss with ranking-aware forward/backward passes.
        
        This overrides the base ConnaCF calculate_loss to use the appropriate
        forward/backward methods based on num_candidates.
        """
        print(f"[calculate_loss] Starting with num_candidates={self.num_candidates}")
        print(f"User ID is : {interaction[self.USER_ID]}")
        print(f"BPR pos item (pre-candidate override): {interaction[self.ITEM_ID]}")
        
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        batch_size = batch_user.size(0)
        
        print(f"[calculate_loss] Batch size: {batch_size}, all_update_rounds: {self.config['all_update_rounds']}")
        
        # For num_candidates == 2, use original ConnaCF logic for backward compatibility
        if self.num_candidates == 2:
            return super().calculate_loss(interaction)
        
        # For num_candidates != 2, use ranking-aware logic
        for round_idx in range(self.config['all_update_rounds']):
            print("~" * 20 + f"{round_idx}-th round update!" + "~" * 20 + '\n')
            
            # Sample candidates for each user
            candidate_items = []
            ground_truth = []
            
            for i in range(batch_size):
                pos_item = int(batch_pos_item[i])
                neg_item = int(batch_neg_item[i])
                
                # Collect additional positives from other users' pos items in this batch
                # (fallback heuristic — preprocessed ranking data is preferred)
                additional_pos = [int(batch_pos_item[j]) for j in range(batch_size) 
                                  if j != i and int(batch_pos_item[j]) != pos_item]
                
                # Sample additional negatives if needed
                num_positives = (self.num_candidates + 1) // 2
                needed_negs = self.num_candidates - num_positives
                neg_items = [neg_item]
                while len(neg_items) < needed_negs:
                    new_neg = np.random.randint(1, self.n_items)
                    if new_neg != pos_item and new_neg not in neg_items:
                        neg_items.append(new_neg)
                
                candidates, gt = sample_candidates(
                    pos_item=pos_item,
                    neg_items=neg_items,
                    num_candidates=self.num_candidates,
                    n_items=self.n_items,
                    additional_pos_items=additional_pos
                )
                candidate_items.append(candidates)
                ground_truth.append(gt)
            
            # Forward pass
            selections, explanations = self.forward_ranking(
                batch_user=batch_user,
                candidate_items=candidate_items,
                round_idx=round_idx,
                batch_idx=0
            )
            
            # Compute accuracy
            accuracies = self.convert_selections_to_accuracy_ranking(
                selections=selections,
                ground_truth=ground_truth,
                candidate_items=candidate_items
            )
            
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            print(f"Current accuracy: {avg_accuracy:.2%}")
            
            # Backward pass for incorrect predictions
            incorrect_indices = [i for i, acc in enumerate(accuracies) if acc == 0]
            correct_indices = [i for i, acc in enumerate(accuracies) if acc == 1]
            
            if incorrect_indices:
                print(f"Updating {len(incorrect_indices)} users with incorrect predictions")
                
                # Filter to incorrect predictions
                incorrect_users = [int(batch_user[i]) for i in incorrect_indices]
                incorrect_selections = [selections[i] for i in incorrect_indices]
                incorrect_explanations = [explanations[i] for i in incorrect_indices]
                incorrect_candidates = [candidate_items[i] for i in incorrect_indices]
                incorrect_gt = [ground_truth[i] for i in incorrect_indices]
                
                self.backward_ranking(
                    selections=incorrect_selections,
                    explanations=incorrect_explanations,
                    batch_user=incorrect_users,
                    candidate_items=incorrect_candidates,
                    ground_truth=incorrect_gt
                )
            
            # Also update correct predictions (reinforcement learning)
            if correct_indices and round_idx == 0:
                print(f"Reinforcing {len(correct_indices)} users with correct predictions")
                
                correct_users = [int(batch_user[i]) for i in correct_indices]
                correct_selections = [selections[i] for i in correct_indices]
                correct_explanations = [explanations[i] for i in correct_indices]
                correct_candidates = [candidate_items[i] for i in correct_indices]
                correct_gt = [ground_truth[i] for i in correct_indices]
                
                self.backward_ranking(
                    selections=correct_selections,
                    explanations=correct_explanations,
                    batch_user=correct_users,
                    candidate_items=correct_candidates,
                    ground_truth=correct_gt
                )
        
        # Log final state
        self._log_ranking_results(batch_user, batch_pos_item, candidate_items, ground_truth, selections)
        
        # Update memory embeddings (same as base ConnaCF)
        self._update_memory_embeddings(batch_user, batch_pos_item, batch_neg_item, explanations)
    
    def _log_ranking_results(
        self,
        batch_user: torch.Tensor,
        batch_pos_item: torch.Tensor,
        candidate_items: List[List[int]],
        ground_truth: List[List[int]],
        selections: List[Any]
    ):
        """Log ranking results for analysis."""
        import os.path as osp
        import os
        
        batch_size = len(batch_user)
        
        for i in range(batch_size):
            user_id = int(batch_user[i])
            path = osp.join(
                self.config['record_path'],
                self.dataset_name,
                'record',
                f'user_record_{self.record_idx}'
            )
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            with open(osp.join(path, f'user.{user_id}'), 'a') as f:
                f.write('~' * 20 + f'Ranking Interaction (num_candidates={self.num_candidates})' + '~' * 20 + '\n')
                
                # Log candidates
                f.write(f'Candidates: {candidate_items[i]}\n')
                f.write(f'Ground truth ranking: {ground_truth[i]}\n')
                
                if self.num_candidates >= 3:
                    f.write(f'Predicted ranking: {selections[i]}\n')
                    
                    # Compute and log ranking metrics
                    feedback = compute_ranking_feedback(
                        selections[i],
                        ground_truth[i],
                        {item_id: self.item_text[item_id] for item_id in candidate_items[i]}
                    )
                    f.write(f'Pairwise accuracy: {feedback["pairwise_accuracy"]:.2%}\n')
                    f.write(f'Top-1 correct: {feedback["top1_correct"]}\n')
                else:
                    f.write(f'Selection: {selections[i]}\n')
                
                f.write(f'User profile: {self.user_agents[user_id].update_memory[-1]}\n\n')
    
    def _update_memory_embeddings(
        self,
        batch_user: torch.Tensor,
        batch_pos_item: torch.Tensor,
        batch_neg_item: torch.Tensor,
        explanations: List[str]
    ):
        """Update memory embeddings (same as base ConnaCF)."""
        batch_size = batch_user.size(0)
        
        # Update user memory
        for i in range(batch_size):
            user_id = int(batch_user[i])
            self.user_agents[user_id].memory_1.append(
                self.user_agents[user_id].update_memory[-1]
            )
        
        # Update item memory embeddings
        batch_pos_item_descriptions = []
        batch_neg_item_descriptions = []
        
        for i in range(batch_size):
            batch_pos_item_descriptions.append(
                self.item_agents[int(batch_pos_item[i])].update_memory[-1]
            )
            batch_neg_item_descriptions.append(
                self.item_agents[int(batch_neg_item[i])].update_memory[-1]
            )
        
        if self.config['evaluation'] == 'rag':
            batch_pos_embeddings = self.generate_embedding(batch_pos_item_descriptions)
            batch_neg_embeddings = self.generate_embedding(batch_neg_item_descriptions)
            
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[
                    batch_pos_item_descriptions[i]
                ] = batch_pos_embeddings[i]
                self.item_agents[int(batch_neg_item[i])].memory_embedding[
                    batch_neg_item_descriptions[i]
                ] = batch_neg_embeddings[i]
        else:
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[
                    batch_pos_item_descriptions[i]
                ] = None
                self.item_agents[int(batch_neg_item[i])].memory_embedding[
                    batch_neg_item_descriptions[i]
                ] = None
    
    def full_sort_predict(self, interaction, idxs):
        """
        Ranking-aware prediction for evaluation.
        
        For num_candidates >= 3, this uses the ranking forward pass.
        For num_candidates <= 2, it falls back to base ConnaCF behavior.
        """
        if self.num_candidates <= 2:
            return super().full_sort_predict(interaction, idxs)
        
        # For ranking mode, we still use the base evaluation logic
        # but could extend this to use ranking-based reranking
        return super().full_sort_predict(interaction, idxs)
