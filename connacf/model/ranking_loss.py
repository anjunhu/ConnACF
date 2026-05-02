"""
Ranking Loss Functions for Multi-Candidate ConnaCF

This module implements THREE PARALLEL loss functions for different num_candidates modes:

=============================================================================
| num_candidates | Loss Function | Output Type    | Prompt Template Key     |
|----------------|---------------|----------------|-------------------------|
| == 1           | BCE Loss      | bool (Yes/No)  | system_prompt_template_binary |
| == 2           | BPR Loss      | item (Choice)  | system_prompt_template  |
| >= 3           | ListNet Loss  | List[items]    | system_prompt_template_ranking |
=============================================================================

Each mode has its own:
- Loss function (BCE, BPR, ListNet)
- Output format (boolean, binary choice string, ranked list)
- Forward prompt template
- Backward prompt template(s)

This ensures full backward compatibility with original ConnaCF (num_candidates=2)
while supporting new sparse (1) and dense (3+) topologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import numpy as np


# =============================================================================
# MODE 1: Binary (num_candidates == 1) - BCE Loss
# =============================================================================

def bce_loss_from_decision(predicted: bool, ground_truth: bool) -> torch.Tensor:
    """
    Binary Cross-Entropy loss for single-item yes/no decisions (num_candidates == 1).
    
    Output type: bool (True = Yes, False = No)
    Prompt template: system_prompt_template_binary
    
    Args:
        predicted: Model's prediction (True = would enjoy, False = would not enjoy)
        ground_truth: Actual preference (True = user liked it, False = user didn't)
        
    Returns:
        Scalar loss value (0.0 if correct, 1.0 if wrong)
    """
    # Simple 0/1 loss for discrete decisions
    correct = predicted == ground_truth
    return torch.tensor(0.0 if correct else 1.0)


# =============================================================================
# MODE 2: Pairwise (num_candidates == 2) - BPR Loss (Original ConnaCF)
# =============================================================================

def bpr_loss_from_selection(selected_item: int, pos_item: int) -> torch.Tensor:
    """
    BPR (Bayesian Personalized Ranking) loss for pairwise comparison (num_candidates == 2).
    
    This PRESERVES the original ConnaCF behavior exactly.
    
    Output type: str (title of selected item)
    Prompt template: system_prompt_template (original)
    
    Args:
        selected_item: The item ID that was selected by the model
        pos_item: The ground truth positive item ID
        
    Returns:
        Scalar loss value (0.0 if correct, 1.0 if wrong)
    """
    correct = selected_item == pos_item
    return torch.tensor(0.0 if correct else 1.0)


# =============================================================================
# MODE 3: Ranking (num_candidates >= 3) - ListNet Loss
# =============================================================================

def ranking_to_relevance(ranking: List[int], num_items: int) -> torch.Tensor:
    """
    Convert a ranking (list of item indices in preference order) to relevance scores.
    
    NOTE: This is ONLY used for num_candidates >= 3 (ListNet loss).
    
    Args:
        ranking: List of item indices, where ranking[0] is most preferred
        num_items: Total number of candidate items
        
    Returns:
        Tensor of shape [num_items] with relevance scores (higher = more preferred)
    """
    relevance = torch.zeros(num_items)
    for rank_pos, item_idx in enumerate(ranking):
        if item_idx < num_items:
            # Higher relevance for items ranked earlier
            relevance[item_idx] = num_items - rank_pos
    return relevance


def listnet_loss(pred_ranking: List[int], true_ranking: List[int], num_items: int) -> torch.Tensor:
    """
    ListNet loss: Cross-entropy between predicted and true ranking distributions.
    
    NOTE: This is ONLY used for num_candidates >= 3.
    
    Output type: List[int] (ordered list of item IDs)
    Prompt template: system_prompt_template_ranking
    
    Args:
        pred_ranking: Predicted ranking (list of item indices, best first)
        true_ranking: Ground truth ranking (list of item indices, best first)
        num_items: Total number of candidate items
        
    Returns:
        Scalar loss value
    """
    # Convert rankings to relevance scores
    pred_relevance = ranking_to_relevance(pred_ranking, num_items)
    true_relevance = ranking_to_relevance(true_ranking, num_items)
    
    # Convert to probability distributions via softmax
    pred_probs = F.softmax(pred_relevance, dim=-1)
    true_probs = F.softmax(true_relevance, dim=-1)
    
    # Cross-entropy loss
    loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))
    return loss


def pairwise_accuracy_from_rankings(pred_ranking: List[int], true_ranking: List[int]) -> Tuple[int, int, float]:
    """
    Compute pairwise accuracy: how many pairs are correctly ordered.
    
    Args:
        pred_ranking: Predicted ranking
        true_ranking: Ground truth ranking
        
    Returns:
        Tuple of (correct_pairs, total_pairs, accuracy)
    """
    # Build position maps
    pred_pos = {item: pos for pos, item in enumerate(pred_ranking)}
    true_pos = {item: pos for pos, item in enumerate(true_ranking)}
    
    correct = 0
    total = 0
    
    items = list(set(pred_ranking) & set(true_ranking))
    for i, item_i in enumerate(items):
        for item_j in items[i+1:]:
            # Check if relative order is preserved
            pred_order = pred_pos.get(item_i, 999) < pred_pos.get(item_j, 999)
            true_order = true_pos.get(item_i, 999) < true_pos.get(item_j, 999)
            
            if pred_order == true_order:
                correct += 1
            total += 1
    
    accuracy = correct / max(total, 1)
    return correct, total, accuracy


def compute_ranking_feedback(
    pred_ranking: List[int],
    true_ranking: List[int],
    item_titles: dict
) -> dict:
    """
    Compute detailed feedback for ranking-based backward pass.
    
    Args:
        pred_ranking: Predicted ranking (item IDs in preference order)
        true_ranking: Ground truth ranking (item IDs in preference order)
        item_titles: Dict mapping item_id to title
        
    Returns:
        Dict with feedback information for backward prompts
    """
    correct_pairs, total_pairs, pairwise_acc = pairwise_accuracy_from_rankings(
        pred_ranking, true_ranking
    )
    
    # Find top-k accuracy
    top1_correct = pred_ranking[0] == true_ranking[0] if pred_ranking and true_ranking else False
    top3_overlap = len(set(pred_ranking[:3]) & set(true_ranking[:3])) if len(pred_ranking) >= 3 else 0
    
    # Find specific misrankings
    pred_pos = {item: pos for pos, item in enumerate(pred_ranking)}
    true_pos = {item: pos for pos, item in enumerate(true_ranking)}
    
    overranked = []  # Items ranked higher than they should be
    underranked = []  # Items ranked lower than they should be
    
    for item in set(pred_ranking) & set(true_ranking):
        pred_p = pred_pos.get(item, 999)
        true_p = true_pos.get(item, 999)
        if pred_p < true_p - 1:  # Ranked too high
            overranked.append((item, pred_p + 1, true_p + 1))
        elif pred_p > true_p + 1:  # Ranked too low
            underranked.append((item, pred_p + 1, true_p + 1))
    
    return {
        'pairwise_accuracy': pairwise_acc,
        'correct_pairs': correct_pairs,
        'total_pairs': total_pairs,
        'top1_correct': top1_correct,
        'top3_overlap': top3_overlap,
        'overranked': overranked,  # [(item_id, pred_rank, true_rank), ...]
        'underranked': underranked,
        'pred_ranking': pred_ranking,
        'true_ranking': true_ranking,
        'is_perfect': pairwise_acc == 1.0,
        'is_mostly_correct': pairwise_acc >= 0.7,
    }


class RankingLoss(nn.Module):
    """
    Unified loss function for ConnaCF with configurable num_candidates.
    
    =============================================================================
    | num_candidates | Loss Function | Output Type    | Prompt Template Key     |
    |----------------|---------------|----------------|-------------------------|
    | == 1           | BCE Loss      | bool (Yes/No)  | system_prompt_template_binary |
    | == 2           | BPR Loss      | str (Choice)   | system_prompt_template  |
    | >= 3           | ListNet Loss  | List[int]      | system_prompt_template_ranking |
    =============================================================================
    
    For backward compatibility, num_candidates == 2 uses the EXACT same loss
    semantics as the original ConnaCF implementation (BPR).
    """
    
    def __init__(self, num_candidates: int = 2, temperature: float = 1.0):
        super().__init__()
        self.num_candidates = num_candidates
        self.temperature = temperature
        
        # Set loss type based on num_candidates
        if num_candidates == 1:
            self._loss_type = "bce"      # Binary Cross-Entropy
            self._output_type = "bool"   # True/False
        elif num_candidates == 2:
            self._loss_type = "bpr"      # Bayesian Personalized Ranking (original ConnaCF)
            self._output_type = "str"    # Item title string
        else:
            self._loss_type = "listnet"  # ListNet ranking loss
            self._output_type = "list"   # List[int] of item IDs
    
    @property
    def loss_type(self) -> str:
        """Return the loss type being used (bce, bpr, or listnet)."""
        return self._loss_type
    
    @property
    def output_type(self) -> str:
        """Return the expected output type (bool, str, or list)."""
        return self._output_type
    
    def forward(
        self,
        prediction: Union[bool, int, List[int]],
        ground_truth: Union[bool, int, List[int]],
        num_items: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute loss based on prediction vs ground truth.
        
        Args:
            prediction: Model prediction (type depends on num_candidates mode)
                - num_candidates == 1: bool (True = Yes, False = No)
                - num_candidates == 2: int (selected item ID)
                - num_candidates >= 3: List[int] (ranked item IDs)
            ground_truth: Ground truth (type depends on num_candidates mode)
                - num_candidates == 1: bool (True = user liked, False = didn't)
                - num_candidates == 2: int (positive item ID)
                - num_candidates >= 3: List[int] (true ranking)
            num_items: Number of candidate items (only for ranking mode)
            
        Returns:
            Scalar loss tensor
        """
        if self.num_candidates == 1:
            # MODE 1: Binary - BCE Loss
            # prediction: bool, ground_truth: bool
            return bce_loss_from_decision(prediction, ground_truth)
        
        elif self.num_candidates == 2:
            # MODE 2: Pairwise - BPR Loss (original ConnaCF)
            # prediction: int (selected item), ground_truth: int (positive item)
            return bpr_loss_from_selection(prediction, ground_truth)
        
        else:
            # MODE 3: Ranking - ListNet Loss
            # prediction: List[int], ground_truth: List[int]
            if num_items is None:
                num_items = max(
                    max(prediction) + 1 if prediction else 1,
                    max(ground_truth) + 1 if ground_truth else 1
                )
            return listnet_loss(prediction, ground_truth, num_items)
    
    def compute_batch_loss(
        self,
        pred_rankings: List[List[int]],
        true_rankings: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute average loss over a batch of rankings.
        """
        losses = []
        for pred, true in zip(pred_rankings, true_rankings):
            losses.append(self.forward(pred, true))
        
        if not losses:
            return torch.tensor(0.0)
        
        return torch.stack(losses).mean()


# =============================================================================
# Accuracy Metrics (mode-specific)
# =============================================================================

def compute_ndcg(pred_ranking: List[int], true_ranking: List[int], k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    NDCG measures ranking quality with position-weighted relevance.
    Items ranked higher get more weight via log discount.
    
    Args:
        pred_ranking: Predicted ranking (list of item IDs, best first)
        true_ranking: Ground truth ranking (list of item IDs, best first)
        k: Cutoff position (None = use full list)
        
    Returns:
        NDCG score in [0, 1]
    """
    if not pred_ranking or not true_ranking:
        return 0.0
    
    if k is None:
        k = len(pred_ranking)
    
    # Build relevance scores from true ranking (higher rank = higher relevance)
    n = len(true_ranking)
    true_relevance = {item: n - pos for pos, item in enumerate(true_ranking)}
    
    # Compute DCG for predicted ranking
    dcg = 0.0
    for pos, item in enumerate(pred_ranking[:k]):
        rel = true_relevance.get(item, 0)
        dcg += rel / np.log2(pos + 2)  # +2 because pos is 0-indexed
    
    # Compute ideal DCG (items in true order)
    idcg = 0.0
    for pos, item in enumerate(true_ranking[:k]):
        rel = true_relevance.get(item, 0)
        idcg += rel / np.log2(pos + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_hr_at_k(pred_ranking: List[int], true_ranking: List[int], k: int = 1) -> float:
    """
    Compute Hit Rate at K (HR@K).
    
    HR@K = 1 if the true top-1 item appears in predicted top-K, else 0.
    
    Args:
        pred_ranking: Predicted ranking
        true_ranking: Ground truth ranking
        k: Cutoff position
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not pred_ranking or not true_ranking:
        return 0.0
    
    true_top1 = true_ranking[0]
    return 1.0 if true_top1 in pred_ranking[:k] else 0.0


def compute_ranking_accuracy(
    predictions: List[Union[bool, int, List[int]]],
    ground_truths: List[Union[bool, int, List[int]]],
    num_candidates: int
) -> dict:
    """
    Compute accuracy metrics based on num_candidates mode.
    
    =============================================================================
    | num_candidates | Metrics                                                  |
    |----------------|----------------------------------------------------------|
    | == 1           | binary_accuracy                                          |
    | == 2           | pairwise_accuracy                                        |
    | >= 3           | NDCG, HR@1, pairwise_accuracy, top1_accuracy             |
    =============================================================================
    
    Args:
        predictions: List of predictions (type depends on mode)
        ground_truths: List of ground truths (type depends on mode)
        num_candidates: Number of candidates per task
        
    Returns:
        Dict with accuracy metrics
    """
    n = max(len(predictions), 1)
    
    if num_candidates == 1:
        # MODE 1: Binary accuracy
        # predictions: List[bool], ground_truths: List[bool]
        correct = sum(
            1 for p, t in zip(predictions, ground_truths)
            if p == t
        )
        return {
            'binary_accuracy': correct / n,
            'mode': 'binary',
            'loss_type': 'bce',
            'output_type': 'bool',
            'feedback_items_per_round': 1
        }
    
    elif num_candidates == 2:
        # MODE 2: Pairwise accuracy (original ConnaCF BPR)
        # predictions: List[int], ground_truths: List[int]
        correct = sum(
            1 for p, t in zip(predictions, ground_truths)
            if p == t
        )
        return {
            'pairwise_accuracy': correct / n,
            'mode': 'pairwise',
            'loss_type': 'bpr',
            'output_type': 'str',
            'feedback_items_per_round': 2
        }
    
    else:
        # MODE 3: Full ranking metrics (ListNet)
        # predictions: List[List[int]], ground_truths: List[List[int]]
        total_ndcg = 0.0
        total_hr1 = 0.0
        total_pairwise_acc = 0.0
        top1_correct = 0
        
        for pred, true in zip(predictions, ground_truths):
            if not pred or not true:
                continue
            
            # NDCG (primary metric for ranking)
            total_ndcg += compute_ndcg(pred, true)
            
            # HR@1 (did we get the top item right?)
            total_hr1 += compute_hr_at_k(pred, true, k=1)
            
            # Pairwise accuracy
            _, _, pair_acc = pairwise_accuracy_from_rankings(pred, true)
            total_pairwise_acc += pair_acc
            
            # Top-1 accuracy (same as HR@1 but explicit)
            if pred[0] == true[0]:
                top1_correct += 1
        
        return {
            'ndcg': total_ndcg / n,
            'hr@1': total_hr1 / n,
            'pairwise_accuracy': total_pairwise_acc / n,
            'top1_accuracy': top1_correct / n,
            'mode': 'ranking',
            'loss_type': 'listnet',
            'output_type': 'list',
            'feedback_items_per_round': num_candidates
        }
