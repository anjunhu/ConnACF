"""
Base Attacker Abstract Class

This module provides the abstract base class for all attack types in the ConnaCF
adversarial attack framework. It defines the common interface and shared functionality
for DrunkAgent, CheatAgent, and NetSafe attacks.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import logging


class BaseAttacker(ABC):
    """
    Abstract base class for all attack types.
    
    Provides common functionality and defines the attack interface with three phases:
    1. optimize() - Phase 1: Offline optimization on surrogate model
    2. inject() - Phase 2: Online injection into victim model
    3. evaluate() - Compute attack-specific metrics
    
    Attributes:
        surrogate: Surrogate model for offline optimization
        config: Attack configuration dictionary
        target_item_id: Target item to promote (if applicable)
        perturbation_budget: Maximum number of tokens to modify
        attack_cache: Cache for optimized attack payloads
        logger: Logger instance for attack execution
    """
    
    def __init__(self, surrogate_model: Optional[Any], config: Dict[str, Any]):
        """
        Initialize base attacker.
        
        Args:
            surrogate_model: Surrogate model for offline optimization (can be None for some attacks)
            config: Attack configuration dictionary containing:
                - target_item_id: Target item to promote
                - perturbation_budget: Maximum token modifications
                - attack_method: Attack type identifier
                - ... (attack-specific parameters)
        """
        self.surrogate = surrogate_model
        self.config = config
        self.target_item_id = config.get('target_item_id')
        self.perturbation_budget = config.get('perturbation_budget', 50)
        self.attack_cache = {}
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(levelname)s] {self.__class__.__name__}: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def optimize(self, *args, **kwargs) -> Any:
        """
        Phase 1: Offline optimization on surrogate model.
        
        This method should use the surrogate model to craft optimized attack payloads
        without touching the victim model. The optimization is performed offline to
        maintain black-box compliance.
        
        Returns:
            Optimized attack payload (format depends on attack type):
                - DrunkAgent: adversarial_description (str)
                - CheatAgent: (optimized_prefix, optimal_insertion_position) (Tuple[str, int])
                - NetSafe: attack-specific payload
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def inject(self, *args, **kwargs) -> Any:
        """
        Phase 2: Online injection into victim model.
        
        This method applies the optimized attack payload to the victim system.
        The injection strategy depends on the attack type:
            - DrunkAgent: Modifies dataset before agent initialization
            - CheatAgent: Injects prefix into prompts during inference
            - NetSafe: Attack-specific injection
        
        Args:
            Depends on attack type:
                - DrunkAgent: dataset, adversarial_description
                - CheatAgent: prompt, prefix, position
                - NetSafe: attack-specific arguments
        
        Returns:
            Modified data (dataset, prompt, or attack-specific)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    @abstractmethod
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute attack-specific metrics.
        
        This method evaluates the effectiveness of the attack using metrics
        appropriate for the attack type:
            - DrunkAgent: Target item rank promotion, top-K inclusion rate
            - CheatAgent: Performance degradation, target item boost
            - NetSafe: Attack-specific metrics
        
        Args:
            results: Evaluation results from victim model containing:
                - rankings: List of item rankings per user
                - metrics: Standard metrics (HR@K, NDCG@K, etc.)
                - ... (attack-specific data)
        
        Returns:
            Dictionary of attack metrics:
                {
                    'metric_name': metric_value,
                    ...
                }
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """
        Common logging functionality.
        
        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        self.logger.log(log_level, message)
    
    def cache_attack(self, key: str, payload: Any):
        """
        Cache optimized attack payload for reuse.
        
        This avoids re-running expensive optimization for repeated experiments.
        
        Args:
            key: Cache key (e.g., "drunk_target_42", "cheat_prefix_user_123")
            payload: Optimized attack payload to cache
        """
        self.attack_cache[key] = payload
        self.log(f"Cached attack payload with key: {key}", "DEBUG")
    
    def get_cached_attack(self, key: str) -> Optional[Any]:
        """
        Retrieve cached attack payload.
        
        Args:
            key: Cache key
        
        Returns:
            Cached attack payload if exists, None otherwise
        """
        payload = self.attack_cache.get(key)
        if payload is not None:
            self.log(f"Retrieved cached attack payload with key: {key}", "DEBUG")
        return payload
    
    def clear_cache(self):
        """Clear all cached attack payloads."""
        self.attack_cache.clear()
        self.log("Cleared attack cache", "DEBUG")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def __repr__(self) -> str:
        """String representation of attacker."""
        return (f"{self.__class__.__name__}("
                f"target_item_id={self.target_item_id}, "
                f"perturbation_budget={self.perturbation_budget})")
