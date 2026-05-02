"""
MACF Configuration Module

Provides configuration management for MACF parameters, supporting loading from
YAML files or dictionaries.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class MACFConfig:
    """
    MACF configuration loaded from YAML or dictionary.
    
    Attributes:
        neighbor_count: Number of similar users to recruit as User Agents
        history_item_count: Number of history items to recruit as Item Agents
        retrieval_k: Number of candidates per retrieval tool call
        max_rounds: Maximum number of discussion rounds
        min_rounds: Minimum number of discussion rounds before allowing convergence
        top_k_recommendation: Final ranked list size
        api_batch: Batch size for embedding API calls
        chat_api_batch: Batch size for chat LLM calls
    """
    
    neighbor_count: int = 5
    history_item_count: int = 5
    retrieval_k: int = 15
    max_rounds: int = 5
    min_rounds: int = 2  # Minimum rounds before convergence allowed
    top_k_recommendation: int = 10
    api_batch: int = 25
    chat_api_batch: int = 10

    # Zero U-I edge stateful orchestrator mode (num_candidates=0 topology)
    # When True, no item agents are recruited and the orchestrator maintains
    # persistent cross-task state as the sole item-side signal substitute.
    stateful_orchestrator: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> 'MACFConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            MACFConfig instance with loaded parameters
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            
        Example:
            >>> config = MACFConfig.from_yaml('props/MACF.yaml')
            >>> print(config.neighbor_count)
            5
        """
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            if config_dict is None:
                logger.warning(f"Empty YAML file at {path}, using defaults")
                return cls()
                
            return cls.from_dict(config_dict)
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {path}: {e}")
            raise
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MACFConfig':
        """
        Load configuration from dictionary.
        
        Missing parameters will use default values. Extra parameters are ignored.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MACFConfig instance with loaded parameters
            
        Example:
            >>> config = MACFConfig.from_dict({'neighbor_count': 10, 'max_rounds': 3})
            >>> print(config.neighbor_count)
            10
            >>> print(config.retrieval_k)  # Uses default
            15
        """
        # Extract only the parameters that MACFConfig accepts
        valid_params = {
            'neighbor_count',
            'history_item_count',
            'retrieval_k',
            'max_rounds',
            'min_rounds',
            'top_k_recommendation',
            'api_batch',
            'chat_api_batch',
            'stateful_orchestrator',
        }
        
        # Filter config_dict to only include valid parameters
        filtered_dict = {
            key: value 
            for key, value in config_dict.items() 
            if key in valid_params
        }
        
        # Log any missing parameters that will use defaults
        missing_params = valid_params - set(filtered_dict.keys())
        if missing_params:
            logger.info(
                f"Using default values for missing parameters: {missing_params}"
            )
        
        # Create instance with filtered parameters
        return cls(**filtered_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if all parameters are valid, False otherwise
            
        Logs warnings for invalid parameters.
        """
        valid = True
        
        if self.neighbor_count <= 0:
            logger.warning(f"neighbor_count must be positive, got {self.neighbor_count}")
            valid = False
            
        if self.history_item_count <= 0:
            logger.warning(f"history_item_count must be positive, got {self.history_item_count}")
            valid = False
            
        if self.retrieval_k <= 0:
            logger.warning(f"retrieval_k must be positive, got {self.retrieval_k}")
            valid = False
            
        if self.max_rounds <= 0:
            logger.warning(f"max_rounds must be positive, got {self.max_rounds}")
            valid = False
            
        if self.min_rounds <= 0:
            logger.warning(f"min_rounds must be positive, got {self.min_rounds}")
            valid = False
            
        if self.min_rounds > self.max_rounds:
            logger.warning(f"min_rounds ({self.min_rounds}) cannot exceed max_rounds ({self.max_rounds})")
            valid = False
            
        if self.top_k_recommendation <= 0:
            logger.warning(f"top_k_recommendation must be positive, got {self.top_k_recommendation}")
            valid = False
            
        if self.api_batch <= 0:
            logger.warning(f"api_batch must be positive, got {self.api_batch}")
            valid = False
            
        if self.chat_api_batch <= 0:
            logger.warning(f"chat_api_batch must be positive, got {self.chat_api_batch}")
            valid = False
            
        return valid
    
    def __str__(self) -> str:
        """Return string representation of configuration."""
        return (
            f"MACFConfig(\n"
            f"  neighbor_count={self.neighbor_count},\n"
            f"  history_item_count={self.history_item_count},\n"
            f"  retrieval_k={self.retrieval_k},\n"
            f"  max_rounds={self.max_rounds},\n"
            f"  min_rounds={self.min_rounds},\n"
            f"  top_k_recommendation={self.top_k_recommendation},\n"
            f"  api_batch={self.api_batch},\n"
            f"  chat_api_batch={self.chat_api_batch}\n"
            f")"
        )
