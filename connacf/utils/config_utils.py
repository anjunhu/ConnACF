"""
Configuration utilities for ConnaCF

Provides helper functions for loading and processing configuration,
including automatic domain-aware prompt injection.
"""

from typing import Dict, Any
import os
from connacf.model.domain_prompts import get_domain_prompts


def inject_domain_prompts(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Inject domain-appropriate prompts into config based on dataset name.
    
    This function automatically detects the domain (movies, music, games, books)
    from the dataset name and injects the appropriate prompts. Only injects
    prompts that are NOT already defined in the config, preserving any
    custom prompts the user has set.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset (e.g., "ml-100k-20-user-dense")
    
    Returns:
        Updated configuration with domain-appropriate prompts
    
    Example:
        >>> config = load_yaml("props/ConnaCF.yaml")
        >>> config = inject_domain_prompts(config, "ml-100k-20-user-dense")
        >>> # Now config has movie-specific prompts instead of CD prompts
    """
    # Get domain-appropriate prompts
    domain_prompts = get_domain_prompts(dataset_name)
    
    # Only inject prompts that aren't already in config
    # This allows users to override with custom prompts if needed
    for key, value in domain_prompts.items():
        if key not in config or not config[key]:
            config[key] = value
    
    return config


def auto_configure_prompts(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically configure prompts based on dataset_name in config.
    
    This is a convenience wrapper that extracts dataset_name from config
    and calls inject_domain_prompts.
    
    Args:
        config: Configuration dictionary (must contain 'dataset_name')
    
    Returns:
        Updated configuration with domain-appropriate prompts
    
    Example:
        >>> config = load_yaml("props/ConnaCF.yaml")
        >>> config['dataset_name'] = "ml-100k-20-user-dense"
        >>> config = auto_configure_prompts(config)
    """
    if 'dataset_name' in config:
        return inject_domain_prompts(config, config['dataset_name'])
    return config


def get_domain_from_dataset(dataset_name: str) -> str:
    """
    Extract domain type from dataset name.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Domain type ('movies', 'music', 'games', 'books')
    
    Example:
        >>> get_domain_from_dataset("ml-100k-20-user-dense")
        'movies'
        >>> get_domain_from_dataset("CDs-100user-sparse")
        'music'
    """
    from connacf.model.domain_prompts import DomainPromptMapper
    mapper = DomainPromptMapper(dataset_name)
    return mapper.domain
