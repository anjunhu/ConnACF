"""
Configuration for natural language agent interactions (U-U and U-I).

This module provides the InteractionConfig dataclass that controls
User-User and User-Item interaction settings in ConnaCF.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from argparse import Namespace


@dataclass
class InteractionConfig:
    """Configuration for natural language agent interactions.
    
    Controls U-U (User-User) and U-I (User-Item) interaction settings.
    Both are disabled by default.
    
    Attributes:
        enable_uu_interaction: Enable User-User friend consultation
        uu_friends_count: Number of similar users to consult (default: 3)
        uu_opinion_max_tokens: Max tokens for friend opinion generation
        enable_ui_interaction: Enable User-Item dialogue
        ui_dialogue_rounds: Number of dialogue rounds (default: 2)
        ui_pitch_max_tokens: Max tokens for item pitch
        ui_response_max_tokens: Max tokens for user response/concerns
    """
    
    # U-U Interaction Settings
    enable_uu_interaction: bool = False
    uu_friends_count: int = 3
    uu_opinion_max_tokens: int = 150
    
    # U-I Interaction Settings
    enable_ui_interaction: bool = False
    ui_dialogue_rounds: int = 2
    ui_pitch_max_tokens: int = 200
    ui_response_max_tokens: int = 150
    
    def __post_init__(self):
        """Validate configuration values."""
        self._validate()
    
    def _validate(self):
        """Validate that configuration values are valid."""
        if self.uu_friends_count < 1:
            raise ValueError(f"uu_friends_count must be positive, got {self.uu_friends_count}")
        if self.ui_dialogue_rounds < 1:
            raise ValueError(f"ui_dialogue_rounds must be positive, got {self.ui_dialogue_rounds}")
        if self.uu_opinion_max_tokens < 1:
            raise ValueError(f"uu_opinion_max_tokens must be positive, got {self.uu_opinion_max_tokens}")
        if self.ui_pitch_max_tokens < 1:
            raise ValueError(f"ui_pitch_max_tokens must be positive, got {self.ui_pitch_max_tokens}")
        if self.ui_response_max_tokens < 1:
            raise ValueError(f"ui_response_max_tokens must be positive, got {self.ui_response_max_tokens}")
    
    @classmethod
    def from_config(cls, config: dict, cli_args: Optional[Namespace] = None) -> 'InteractionConfig':
        """Create InteractionConfig from YAML config dict with CLI override support.
        
        CLI arguments take precedence over config file values (Requirement 1.5, 2.5).
        
        Args:
            config: Dictionary from YAML config file
            cli_args: Optional argparse Namespace with CLI overrides
            
        Returns:
            InteractionConfig instance with merged settings
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Start with defaults
        kwargs = {}
        
        # Load from config file
        config_mappings = {
            'enable_uu_interaction': 'enable_uu_interaction',
            'uu_friends_count': 'uu_friends_count',
            'uu_opinion_max_tokens': 'uu_opinion_max_tokens',
            'enable_ui_interaction': 'enable_ui_interaction',
            'ui_dialogue_rounds': 'ui_dialogue_rounds',
            'ui_pitch_max_tokens': 'ui_pitch_max_tokens',
            'ui_response_max_tokens': 'ui_response_max_tokens',
        }
        
        for config_key, attr_name in config_mappings.items():
            if config_key in config and config[config_key] is not None:
                kwargs[attr_name] = config[config_key]
        
        # CLI overrides (take precedence)
        if cli_args is not None:
            # Handle enable flags for U-U and U-I
            if hasattr(cli_args, 'enable_uu_interaction') and cli_args.enable_uu_interaction:
                kwargs['enable_uu_interaction'] = True
            if hasattr(cli_args, 'enable_ui_interaction') and cli_args.enable_ui_interaction:
                kwargs['enable_ui_interaction'] = True
            
            # Handle numeric arguments
            if hasattr(cli_args, 'uu_friends_count') and cli_args.uu_friends_count is not None:
                kwargs['uu_friends_count'] = cli_args.uu_friends_count
            if hasattr(cli_args, 'ui_dialogue_rounds') and cli_args.ui_dialogue_rounds is not None:
                kwargs['ui_dialogue_rounds'] = cli_args.ui_dialogue_rounds
        
        # Validate and create instance
        try:
            return cls(**kwargs)
        except ValueError as e:
            logger.warning(f"Invalid interaction config value: {e}. Using defaults.")
            return cls()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'enable_uu_interaction': self.enable_uu_interaction,
            'uu_friends_count': self.uu_friends_count,
            'uu_opinion_max_tokens': self.uu_opinion_max_tokens,
            'enable_ui_interaction': self.enable_ui_interaction,
            'ui_dialogue_rounds': self.ui_dialogue_rounds,
            'ui_pitch_max_tokens': self.ui_pitch_max_tokens,
            'ui_response_max_tokens': self.ui_response_max_tokens,
        }
    
    def __repr__(self) -> str:
        return (
            f"InteractionConfig("
            f"uu={self.enable_uu_interaction}[friends={self.uu_friends_count}], "
            f"ui={self.enable_ui_interaction}[rounds={self.ui_dialogue_rounds}])"
        )
