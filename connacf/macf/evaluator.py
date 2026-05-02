"""
MACF Evaluator for computing recommendation metrics.

This module provides the MACFEvaluator class for evaluating MACF recommendations
using standard information retrieval metrics like Hit@K and NDCG@K.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import math
import logging
from typing import List, Dict, Any, TYPE_CHECKING

from connacf.macf.data_models import EvaluationResult

if TYPE_CHECKING:
    from connacf.dataset import BPRDataset

logger = logging.getLogger(__name__)


class MACFEvaluator:
    """
    Computes evaluation metrics for MACF recommendations.
    
    Provides methods to evaluate individual recommendations and aggregate
    metrics across multiple evaluations. Supports Hit@K and NDCG@K metrics.
    
    Attributes:
        dataset: The BPRDataset used for evaluation context
        results: List of EvaluationResult objects from all evaluations
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def __init__(self, dataset: 'BPRDataset'):
        """
        Initialize the evaluator with a dataset.
        
        Args:
            dataset: BPRDataset instance for evaluation context
        
        Requirements: 8.1
        """
        self.dataset = dataset
        self.results: List[EvaluationResult] = []
    
    def evaluate_single(
        self,
        user_id: int,
        ranked_list: List[int],
        ground_truth: List[int],
        query: str = "",
        num_rounds: int = 0,
        discussion_log: List[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single recommendation.
        
        Computes Hit@10 and NDCG@10 for a single user-query recommendation
        and stores the result for later aggregation.
        
        Args:
            user_id: The target user ID
            ranked_list: List of recommended item IDs in ranked order
            ground_truth: List of relevant item IDs (ground truth)
            query: The natural language query (optional)
            num_rounds: Number of discussion rounds used (optional)
            discussion_log: Full discussion history (optional)
        
        Returns:
            EvaluationResult containing the computed metrics
        
        Requirements: 8.1
        """
        if discussion_log is None:
            discussion_log = []
        
        # Compute metrics
        hit_at_10 = self.compute_hit_at_k(ranked_list, ground_truth, k=10)
        ndcg_at_10 = self.compute_ndcg_at_k(ranked_list, ground_truth, k=10)
        
        # Create evaluation result
        result = EvaluationResult(
            user_id=user_id,
            query=query,
            ranked_list=ranked_list,
            ground_truth=ground_truth,
            hit_at_10=hit_at_10,
            ndcg_at_10=ndcg_at_10,
            num_rounds=num_rounds,
            discussion_log=discussion_log
        )
        
        # Store result for later aggregation
        self.results.append(result)
        
        return result
    
    def compute_hit_at_k(
        self,
        ranked_list: List[int],
        ground_truth: List[int],
        k: int = 10
    ) -> float:
        """
        Compute Hit@K metric.
        
        Hit@K equals 1.0 if any ground truth item appears in the top-K
        recommendations, otherwise 0.0.
        
        Args:
            ranked_list: List of recommended item IDs in ranked order
            ground_truth: List of relevant item IDs
            k: Number of top items to consider (default: 10)
        
        Returns:
            1.0 if hit, 0.0 otherwise
        
        Requirements: 8.2
        """
        if not ranked_list or not ground_truth:
            return 0.0
        
        # Get top-k items from ranked list
        top_k = ranked_list[:k]
        
        # Convert ground truth to set for O(1) lookup
        ground_truth_set = set(ground_truth)
        
        # Check if any top-k item is in ground truth
        for item_id in top_k:
            if item_id in ground_truth_set:
                return 1.0
        
        return 0.0
    
    def compute_ndcg_at_k(
        self,
        ranked_list: List[int],
        ground_truth: List[int],
        k: int = 10
    ) -> float:
        """
        Compute NDCG@K using standard DCG/IDCG formula with binary relevance.
        
        NDCG (Normalized Discounted Cumulative Gain) measures ranking quality
        by comparing the actual DCG to the ideal DCG.
        
        DCG@K = sum_{i=1}^{K} rel_i / log2(i + 1)
        IDCG@K = sum_{i=1}^{min(K, |ground_truth|)} 1 / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
        
        With binary relevance: rel_i = 1 if item at position i is relevant, else 0.
        
        Args:
            ranked_list: List of recommended item IDs in ranked order
            ground_truth: List of relevant item IDs
            k: Number of top items to consider (default: 10)
        
        Returns:
            NDCG@K score between 0.0 and 1.0
        
        Requirements: 8.3
        """
        if not ranked_list or not ground_truth:
            return 0.0
        
        # Convert ground truth to set for O(1) lookup
        ground_truth_set = set(ground_truth)
        
        # Compute DCG@K
        dcg = 0.0
        for i, item_id in enumerate(ranked_list[:k]):
            if item_id in ground_truth_set:
                # Binary relevance: rel_i = 1 if relevant
                # Position is 1-indexed in the formula, so use (i + 2) for log2(i + 1)
                dcg += 1.0 / math.log2(i + 2)
        
        # Compute IDCG@K (ideal DCG with all relevant items at top)
        # Number of relevant items that could appear in top-k
        num_relevant = min(k, len(ground_truth))
        idcg = 0.0
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)
        
        # Avoid division by zero
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all evaluations.
        
        Computes mean Hit@10 and NDCG@10 across all stored evaluation results.
        
        Returns:
            Dictionary with 'hit_at_10' and 'ndcg_at_10' mean values
        
        Requirements: 8.4
        """
        if not self.results:
            return {
                'hit_at_10': 0.0,
                'ndcg_at_10': 0.0,
                'num_evaluations': 0
            }
        
        # Compute mean metrics
        total_hit = sum(r.hit_at_10 for r in self.results)
        total_ndcg = sum(r.ndcg_at_10 for r in self.results)
        num_results = len(self.results)
        
        return {
            'hit_at_10': total_hit / num_results,
            'ndcg_at_10': total_ndcg / num_results,
            'num_evaluations': num_results
        }
    
    def print_results(self) -> None:
        """
        Print aggregate metrics to console.
        
        Displays the mean Hit@10 and NDCG@10 metrics along with
        the number of evaluations performed.
        
        Requirements: 8.5
        """
        metrics = self.aggregate_metrics()
        
        print("\n" + "=" * 50)
        print("MACF Evaluation Results")
        print("=" * 50)
        print(f"Number of evaluations: {metrics['num_evaluations']}")
        print(f"Hit@10:  {metrics['hit_at_10']:.4f}")
        print(f"NDCG@10: {metrics['ndcg_at_10']:.4f}")
        print("=" * 50 + "\n")
        
        # Also log the results
        logger.info(
            f"MACF Evaluation - Hit@10: {metrics['hit_at_10']:.4f}, "
            f"NDCG@10: {metrics['ndcg_at_10']:.4f}, "
            f"Evaluations: {metrics['num_evaluations']}"
        )
    
    def clear_results(self) -> None:
        """
        Clear all stored evaluation results.
        
        Useful for resetting the evaluator between different evaluation runs.
        """
        self.results = []
    
    def get_results(self) -> List[EvaluationResult]:
        """
        Get all stored evaluation results.
        
        Returns:
            List of EvaluationResult objects from all evaluations
        """
        return self.results
