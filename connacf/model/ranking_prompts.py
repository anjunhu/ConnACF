"""
Ranking-Based Prompt Templates for Multi-Candidate ConnaCF

Forward prompt templates used as DEFAULTS when not specified in config.
Config values (system_prompt_template_binary, system_prompt_template, system_prompt_template_ranking)
take precedence over these defaults.

NOTE: These are generic movie-based defaults. The domain_prompts.py module
provides automatic domain detection and mapping for different datasets.

Backward prompts are inline in ranking_handler.py.
"""

from typing import List, Dict


# ==================== Forward Prompts (Generic Defaults) ====================
# These are used as fallbacks if config doesn't specify the prompt templates.
# In practice, domain_prompts.py auto-injects domain-appropriate prompts.

FORWARD_PROMPT_BINARY = """You are a movie enthusiast. Here is your self-introduction, expressing your preferences and dislikes:
'{user_description}'

Now, you are considering whether to watch this movie:
{item_description}

Based on your preferences and dislikes, would you choose to watch this movie?

Important notes:
1. **Output Format:** Your response must be: 'Decision: [Yes/No] \n Explanation: [Your reasoning]'
2. Be specific about why this movie does or doesn't match your preferences.
3. Base your decision on facts from your self-introduction."""


FORWARD_PROMPT_PAIRWISE = """You are a movie enthusiast. Here is your self-introduction, expressing your preferences and dislikes:
'{user_description}'

Now, you are considering to select a movie from two candidate movies. The features of these two candidate movies are listed as follows:
{list_of_item_description}

Please select the movie that aligns best with your preferences. Furthermore, you must articulate why you've chosen that particular movie while rejecting the other.

Important notes:
1. **Output Format:** Your response must be: 'Choice: [Title of the selected movie] \n Explanation: [Your reasoning]'
2. You must choose exactly one of the two candidates.
3. Be specific about why one movie matches your preferences better than the other."""


FORWARD_PROMPT_RANKING = """You are a movie enthusiast. Here is your self-introduction, expressing your preferences and dislikes:
'{user_description}'

Now, you are considering to rank {num_candidates} candidate movies based on your preferences. The features of these candidate movies are listed as follows:
{list_of_item_description}

Please rank ALL {num_candidates} movies from most preferred (1st) to least preferred ({num_candidates}th) based on your preferences and dislikes.

Important notes:
1. **Output Format:** Your response must be:
   'Ranking:
   1. [Title of most preferred movie]
   2. [Title of second most preferred movie]
   ...
   {num_candidates}. [Title of least preferred movie]
   
   Explanation: [Why you ranked the top choice first and the bottom choice last]'
2. You MUST rank ALL {num_candidates} candidates - no ties allowed.
3. Be specific about the features that influenced your ranking."""


# ==================== Helper Functions ====================

def format_ranking(ranking: List[int], item_titles: Dict[int, str]) -> str:
    """Format a ranking as a numbered list for display/logging."""
    lines = []
    for i, item_id in enumerate(ranking):
        title = item_titles.get(item_id, f"Item {item_id}")
        lines.append(f"{i+1}. {title}")
    return "\n".join(lines)
