"""
Surrogate Model Runner

This module provides a surrogate model wrapper for black-box attack optimization.
The surrogate uses Llama with General User Memories and Popular Items to maintain
strict black-box compliance (no access to victim's private data).

"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
import logging
import numpy as np


class SurrogateRunner:
    """
    Surrogate model wrapper for black-box attack optimization.
    
    Uses Llama model with General User Memories and Popular Items to optimize
    attacks offline without accessing victim's private interaction history.
    
    Black-Box Compliance:
        - Uses generic user personas (NOT victim's interaction history)
        - Uses popular items from public statistics (NOT victim's private memories)
        - Never queries victim model during optimization
    
    Attributes:
        llm: Llama model instance
        user_memories: Generic user personas
        item_memories: Popular item descriptions
        config: Configuration dictionary
        logger: Logger instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize surrogate with black-box compliant data.
        
        Args:
            config: Configuration dictionary containing:
                - surrogate_model: Model identifier (e.g., 'llama-3-8b')
                - n_users: Number of generic user personas to generate
                - top_k_popular_items: Number of popular items to use
                - use_general_user_memories: Flag to use generic personas
                - use_popular_items: Flag to use public statistics
        """
        self.config = config
        self.logger = logging.getLogger('SurrogateRunner')
        
        # Initialize Llama model
        surrogate_model_name = config.get('surrogate_model', 'llama-3-8b')
        self.llm = self._initialize_llama(surrogate_model_name)
        
        # Black-box compliance: Use generic data, NOT victim's private data
        if config.get('use_general_user_memories', True):
            n_users = config.get('n_users', 100)
            self.user_memories = self._generate_general_user_memories(n_users)
            self.logger.info(f"Generated {len(self.user_memories)} generic user personas")
        else:
            self.user_memories = []
            self.logger.warning("Not using general user memories (black-box violation risk)")
        
        if config.get('use_popular_items', True):
            top_k = config.get('top_k_popular_items', 100)
            self.item_memories = self._generate_popular_item_memories(top_k)
            self.logger.info(f"Generated {len(self.item_memories)} popular item memories")
        else:
            self.item_memories = []
            self.logger.warning("Not using popular items (black-box violation risk)")
    
    def _initialize_llama(self, model_name: str) -> Any:
        """
        Initialize Llama model via API or local loading.
        
        Args:
            model_name: Model identifier (e.g., 'llama-3-8b')
        
        Returns:
            Llama model instance (placeholder for now)
        
        Note:
            This is a placeholder implementation. In production, this would:
            - Load Llama model via API (e.g., Together AI, Replicate)
            - Or load local Llama model using transformers
            - Handle authentication and rate limiting
        """
        self.logger.info(f"Initializing Llama surrogate model: {model_name}")
        
        # Placeholder: Return a mock model for now
        # In production, replace with actual Llama initialization:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        
        class MockLlamaModel:
            """Mock Llama model for development."""
            def __init__(self, model_name):
                self.model_name = model_name
                self.logger = logging.getLogger('MockLlamaModel')
            
            def get_logits(self, prompt: str) -> torch.Tensor:
                """Mock logits generation."""
                self.logger.debug(f"Mock logits for prompt: {prompt[:50]}...")
                # Return random logits for development
                return torch.randn(100)  # 100 items
            
            def generate(self, prompt: str, max_tokens: int = 100) -> str:
                """Mock text generation."""
                self.logger.debug(f"Mock generation for prompt: {prompt[:50]}...")
                return f"Generated response for: {prompt[:30]}..."
        
        return MockLlamaModel(model_name)
    
    def _generate_general_user_memories(self, n_users: int) -> List[str]:
        """
        Generate generic user personas WITHOUT victim-specific patterns.
        
        These are generic descriptions that do not reflect any specific user's
        interaction history, maintaining black-box compliance.
        
        Args:
            n_users: Number of user personas to generate
        
        Returns:
            List of generic user descriptions
        """
        templates = [
            "A user interested in general recommendations",
            "A user who enjoys popular content",
            "A user seeking diverse recommendations",
            "A user with mainstream preferences",
            "A user exploring new content",
            "A user who values quality over quantity",
            "A user interested in trending items",
            "A user with eclectic tastes",
            "A user who prefers classic content",
            "A user open to discovering new genres"
        ]
        
        # Cycle through templates to create n_users personas
        user_memories = [templates[i % len(templates)] for i in range(n_users)]
        
        self.logger.debug(f"Generated {len(user_memories)} generic user personas")
        return user_memories
    
    def _generate_popular_item_memories(self, top_k: int) -> List[str]:
        """
        Use public interaction statistics to identify popular items.
        
        These are based on publicly available statistics (e.g., most viewed,
        most rated), NOT victim's private agent memories.
        
        Args:
            top_k: Number of popular items to include
        
        Returns:
            List of popular item descriptions
        """
        # Placeholder: In production, this would query public statistics
        # For now, return generic popular item descriptions
        item_memories = [
            f"Popular item {i}: Highly rated content with broad appeal"
            for i in range(top_k)
        ]
        
        self.logger.debug(f"Generated {len(item_memories)} popular item memories")
        return item_memories
    
    def get_loss_score(self, prompt: str, target_item_id: int) -> float:
        """
        Compute negative log-likelihood for DrunkAgent optimization.
        
        This method evaluates how well the prompt promotes the target item
        by computing the loss (lower is better for attack).
        
        Args:
            prompt: Current adversarial prompt/description
            target_item_id: Target item to promote
        
        Returns:
            Loss score (lower is better for attack)
        """
        # Construct ranking prompt
        ranking_prompt = self._construct_ranking_prompt(prompt, target_item_id)
        
        # Get model's ranking prediction
        logits = self.llm.get_logits(ranking_prompt)
        
        # Compute negative log-likelihood of target item being ranked first
        # Lower loss = target item ranked higher = better attack
        target_logit = logits[target_item_id] if target_item_id < len(logits) else logits[0]
        loss = -F.log_softmax(logits, dim=0)[target_item_id if target_item_id < len(logits) else 0]
        
        self.logger.debug(f"Loss score for target {target_item_id}: {loss.item():.4f}")
        return loss.item()
    
    def get_gradient_feedback(self, 
                             prefix: torch.Tensor, 
                             base_prompt: str, 
                             target_item_id: int) -> torch.Tensor:
        """
        Compute gradient for CheatAgent prefix tuning.
        
        This method performs gradient ascent on the prefix to maximize
        the target item's ranking.
        
        Args:
            prefix: Trainable prefix tensor
            base_prompt: Original user prompt
            target_item_id: Target item to promote
        
        Returns:
            Gradient tensor for prefix update
        """
        # Construct prompt with prefix
        full_prompt = self._inject_prefix(base_prompt, prefix)
        
        # Forward pass
        loss = self.get_loss_score(full_prompt, target_item_id)
        
        # Backward pass (placeholder - in production, use actual gradients)
        # For now, return random gradient
        loss_tensor = torch.tensor(loss, requires_grad=True)
        
        # In production, this would be:
        # loss_tensor.backward()
        # return prefix.grad
        
        # Placeholder gradient
        gradient = torch.randn_like(prefix) * 0.01
        self.logger.debug(f"Gradient feedback computed for prefix")
        return gradient
    
    def _construct_ranking_prompt(self, description: str, target_id: int) -> str:
        """
        Construct prompt for ranking evaluation.
        
        Args:
            description: Item description or user prompt
            target_id: Target item ID
        
        Returns:
            Formatted prompt for ranking
        """
        return f"Given item description: {description}\nRank this item for users. Target item: {target_id}"
    
    def _inject_prefix(self, prompt: str, prefix: torch.Tensor) -> str:
        """
        Convert prefix tensor to text and inject into prompt.
        
        Args:
            prompt: Original prompt
            prefix: Prefix tensor
        
        Returns:
            Prompt with injected prefix
        """
        # Convert prefix tensor to text tokens
        prefix_text = self._tensor_to_text(prefix)
        return f"{prefix_text} {prompt}"
    
    def _tensor_to_text(self, tensor: torch.Tensor) -> str:
        """
        Convert tensor to text tokens.
        
        Args:
            tensor: Tensor to convert
        
        Returns:
            Text representation
        
        Note:
            Placeholder implementation. In production, use tokenizer.decode()
        """
        # Placeholder: In production, use tokenizer
        # return self.tokenizer.decode(tensor.long())
        
        # For now, return placeholder text
        return f"[PREFIX_{tensor.shape[0]}]"
    
    def evaluate_candidate(self, 
                          candidate_text: str, 
                          target_item_id: int,
                          context: Optional[str] = None) -> float:
        """
        Evaluate a candidate attack text.
        
        Args:
            candidate_text: Candidate adversarial text
            target_item_id: Target item to promote
            context: Optional context for evaluation
        
        Returns:
            Score (lower is better for attack)
        """
        if context:
            full_text = f"{context}\n{candidate_text}"
        else:
            full_text = candidate_text
        
        return self.get_loss_score(full_text, target_item_id)
    
    def batch_evaluate_candidates(self, 
                                 candidates: List[str], 
                                 target_item_id: int) -> List[float]:
        """
        Evaluate multiple candidates in batch.
        
        Args:
            candidates: List of candidate texts
            target_item_id: Target item to promote
        
        Returns:
            List of scores (lower is better for attack)
        """
        scores = []
        for candidate in candidates:
            score = self.get_loss_score(candidate, target_item_id)
            scores.append(score)
        
        self.logger.debug(f"Batch evaluated {len(candidates)} candidates")
        return scores
