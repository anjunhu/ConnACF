"""
Defines different types of information leakage that can be tracked in MAS
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


class LeakageType(Enum):
    """Types of information that can leak from MAS (aligned with MASLeak paper)"""
    
    # Core MASLeak IP extraction targets (from paper arXiv:2505.12442)
    SYSTEM_PROMPT = "system_prompt"  # Agent system prompts and role descriptions
    TASK_INSTRUCTIONS = "task_instructions"  # Forward/backward pass instructions, ranking logic
    TOPOLOGY = "topology"  # Agent communication structure, number of agents
    
    # Additional observations (ConnaCF-specific, not in original paper)
    AGENT_PROFILE = "agent_profile"  # Demographics, preferences (observable in conversations)
    MESSAGE_CONTENT = "message_content"  # Communication patterns
    
    # Legacy names for backward compatibility
    TRUE_RATINGS = "true_ratings"  # Not used in ConnaCF (no raw ratings exposed)
    
    # Composite leakage
    FULL_IDENTITY = "full_identity"  # Complete agent reconstruction
    FULL_TOPOLOGY = "full_topology"  # Complete graph reconstruction


@dataclass
class LeakageMetrics:
    """Metrics for measuring information leakage accuracy"""
    
    leakage_type: LeakageType
    
    # Accuracy metrics
    exact_match_accuracy: float = 0.0  # Exact matches
    top_k_accuracy: Dict[int, float] = field(default_factory=dict)  # Top-k accuracy
    
    # Similarity metrics
    cosine_similarity: float = 0.0
    jaccard_similarity: float = 0.0
    edit_distance: Optional[float] = None
    
    # Statistical metrics
    correlation: float = 0.0  # For numerical data
    mae: Optional[float] = None  # Mean absolute error
    rmse: Optional[float] = None  # Root mean squared error
    
    # Confidence metrics
    confidence_scores: List[float] = field(default_factory=list)
    avg_confidence: float = 0.0
    
    # Coverage metrics
    total_items: int = 0
    recovered_items: int = 0
    recovery_rate: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_recovery_rate(self):
        """Compute recovery rate from recovered/total items"""
        if self.total_items > 0:
            self.recovery_rate = self.recovered_items / self.total_items
        return self.recovery_rate
    
    def compute_avg_confidence(self):
        """Compute average confidence from confidence scores"""
        if self.confidence_scores:
            self.avg_confidence = np.mean(self.confidence_scores)
        return self.avg_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'leakage_type': self.leakage_type.value,
            'exact_match_accuracy': self.exact_match_accuracy,
            'top_k_accuracy': self.top_k_accuracy,
            'cosine_similarity': self.cosine_similarity,
            'jaccard_similarity': self.jaccard_similarity,
            'edit_distance': self.edit_distance,
            'correlation': self.correlation,
            'mae': self.mae,
            'rmse': self.rmse,
            'avg_confidence': self.avg_confidence,
            'recovery_rate': self.recovery_rate,
            'total_items': self.total_items,
            'recovered_items': self.recovered_items,
            'metadata': self.metadata
        }


@dataclass
class LeakageObservation:
    """Single observation of leaked information"""
    
    leakage_type: LeakageType
    observed_value: Any
    ground_truth: Any
    confidence: float
    timestamp: float
    source: str  # Where the leakage came from
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_correct(self) -> bool:
        """Check if observation matches ground truth"""
        return self.observed_value == self.ground_truth
    
    def compute_similarity(self) -> float:
        """Compute similarity between observed and ground truth"""
        if isinstance(self.observed_value, (int, float)) and isinstance(self.ground_truth, (int, float)):
            # Numerical similarity
            if self.ground_truth == 0:
                return 1.0 if self.observed_value == 0 else 0.0
            return 1.0 - abs(self.observed_value - self.ground_truth) / abs(self.ground_truth)
        
        elif isinstance(self.observed_value, str) and isinstance(self.ground_truth, str):
            # String similarity (Jaccard on words)
            obs_words = set(self.observed_value.lower().split())
            gt_words = set(self.ground_truth.lower().split())
            if not obs_words and not gt_words:
                return 1.0
            if not obs_words or not gt_words:
                return 0.0
            return len(obs_words & gt_words) / len(obs_words | gt_words)
        
        elif isinstance(self.observed_value, (list, set)) and isinstance(self.ground_truth, (list, set)):
            # Set similarity
            obs_set = set(self.observed_value)
            gt_set = set(self.ground_truth)
            if not obs_set and not gt_set:
                return 1.0
            if not obs_set or not gt_set:
                return 0.0
            return len(obs_set & gt_set) / len(obs_set | gt_set)
        
        else:
            # Default: exact match
            return 1.0 if self.observed_value == self.ground_truth else 0.0
