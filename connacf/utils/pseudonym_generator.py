"""
Pseudonym Generator for ConnaCF Agents

Generates human-readable, memorable pseudonyms for user and item agents
to improve log readability and maintain privacy in experiments.

Usage:
    generator = PseudonymGenerator(seed=42)
    user_name = generator.get_user_pseudonym(user_id=5)  # "Alice_Technician"
    item_name = generator.get_item_pseudonym(item_id=10)  # "Movie_Inception"
"""

import random
from typing import Dict, Optional, Tuple


class PseudonymGenerator:
    """Generate consistent, memorable pseudonyms for agents."""
    
    # First names for user agents (gender-neutral and diverse)
    FIRST_NAMES = [
        "Alex", "Blake", "Casey", "Drew", "Ellis", "Finley", "Gray", "Harper",
        "Indigo", "Jordan", "Kai", "Logan", "Morgan", "Noel", "Oakley", "Parker",
        "Quinn", "Reese", "Sage", "Taylor", "Uma", "Val", "Winter", "Xen",
        "Yael", "Zion", "Arden", "Bay", "Cedar", "Devon", "Echo", "Frost",
        "Haven", "Iris", "Jules", "Kit", "Lane", "Marley", "Nova", "Ocean",
        "Phoenix", "River", "Sky", "True", "Unity", "Wren", "Azure", "Brook",
        "Cloud", "Dawn", "Eden", "Fern", "Glen", "Jade", "Lake", "Moon",
        "North", "Onyx", "Pearl", "Rain", "Star", "Vale", "West", "Zen"
    ]
    
    # Adjectives for item agents (positive, descriptive)
    ITEM_ADJECTIVES = [
        "Classic", "Epic", "Brilliant", "Timeless", "Legendary", "Iconic",
        "Stellar", "Prime", "Elite", "Supreme", "Grand", "Noble", "Majestic",
        "Radiant", "Vivid", "Bold", "Daring", "Fierce", "Mighty", "Powerful",
        "Elegant", "Graceful", "Refined", "Polished", "Pristine", "Flawless",
        "Dynamic", "Vibrant", "Spirited", "Lively", "Energetic", "Zesty",
        "Mystical", "Enigmatic", "Cryptic", "Arcane", "Ethereal", "Cosmic",
        "Golden", "Silver", "Crimson", "Azure", "Emerald", "Sapphire",
        "Swift", "Agile", "Nimble", "Quick", "Rapid", "Fleet",
        "Wise", "Sage", "Astute", "Sharp", "Keen", "Clever",
        "Brave", "Valiant", "Gallant", "Heroic", "Fearless", "Intrepid"
    ]
    
    # Nouns for item agents (neutral objects/concepts)
    ITEM_NOUNS = [
        "Phoenix", "Dragon", "Tiger", "Eagle", "Wolf", "Falcon", "Raven",
        "Storm", "Thunder", "Lightning", "Tempest", "Cyclone", "Tornado",
        "Mountain", "Summit", "Peak", "Crest", "Ridge", "Cliff",
        "Ocean", "River", "Stream", "Cascade", "Rapids", "Tide",
        "Star", "Comet", "Nova", "Galaxy", "Nebula", "Cosmos",
        "Flame", "Blaze", "Inferno", "Ember", "Spark", "Torch",
        "Shadow", "Eclipse", "Twilight", "Dawn", "Dusk", "Horizon",
        "Crystal", "Diamond", "Gem", "Jewel", "Prism", "Shard",
        "Blade", "Arrow", "Spear", "Shield", "Crown", "Scepter",
        "Quest", "Journey", "Odyssey", "Voyage", "Adventure", "Expedition"
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize pseudonym generator.
        
        Args:
            seed: Random seed for reproducibility. If None, uses random seed.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Caches for consistent pseudonyms
        self._user_cache: Dict[int, str] = {}
        self._item_cache: Dict[int, str] = {}
        
        # Shuffle lists for variety
        self._shuffled_names = self.FIRST_NAMES.copy()
        self._shuffled_adjectives = self.ITEM_ADJECTIVES.copy()
        self._shuffled_nouns = self.ITEM_NOUNS.copy()
        
        self.rng.shuffle(self._shuffled_names)
        self.rng.shuffle(self._shuffled_adjectives)
        self.rng.shuffle(self._shuffled_nouns)
    
    def get_user_pseudonym(
        self,
        user_id: int,
        occupation: Optional[str] = None,
        gender: Optional[str] = None,
        include_id: bool = False
    ) -> str:
        """
        Generate pseudonym for a user agent.
        
        Args:
            user_id: Ordinal user ID
            occupation: User's occupation (optional)
            gender: User's gender (optional, not used to avoid stereotyping)
            include_id: Whether to include numeric ID in pseudonym
        
        Returns:
            Pseudonym like "Alex_Technician" or "Blake_42" or "Casey"
        
        Examples:
            >>> gen = PseudonymGenerator(seed=42)
            >>> gen.get_user_pseudonym(5, occupation="technician")
            'Harper_Technician'
            >>> gen.get_user_pseudonym(5, include_id=True)
            'Harper_5'
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        
        # Select name based on user_id
        name_idx = user_id % len(self._shuffled_names)
        first_name = self._shuffled_names[name_idx]
        
        # Build pseudonym
        parts = [first_name]
        
        if occupation:
            # Capitalize and clean occupation
            occ_clean = occupation.replace('_', ' ').title().replace(' ', '')
            parts.append(occ_clean)
        elif include_id:
            parts.append(str(user_id))
        
        pseudonym = '_'.join(parts)
        self._user_cache[user_id] = pseudonym
        
        return pseudonym
    
    def get_item_pseudonym(
        self,
        item_id: int,
        item_title: Optional[str] = None,
        item_type: str = "Item",
        include_id: bool = False
    ) -> str:
        """
        Generate pseudonym for an item agent.
        
        Args:
            item_id: Ordinal item ID
            item_title: Actual item title (optional, used for extraction)
            item_type: Type of item ("Movie", "CD", "Game", "Book")
            include_id: Whether to include numeric ID
        
        Returns:
            Pseudonym like "Epic_Phoenix" or "Movie_Inception" or "Classic_42"
        
        Examples:
            >>> gen = PseudonymGenerator(seed=42)
            >>> gen.get_item_pseudonym(10, item_type="Movie")
            'Stellar_Storm'
            >>> gen.get_item_pseudonym(10, item_title="Inception", item_type="Movie")
            'Movie_Inception'
        """
        if item_id in self._item_cache:
            return self._item_cache[item_id]
        
        # If we have the actual title, use it (sanitized)
        if item_title:
            # Extract key word from title (first significant word)
            title_clean = self._extract_key_word(item_title)
            pseudonym = f"{item_type}_{title_clean}"
        else:
            # Generate from adjective + noun
            adj_idx = item_id % len(self._shuffled_adjectives)
            noun_idx = (item_id // len(self._shuffled_adjectives)) % len(self._shuffled_nouns)
            
            adjective = self._shuffled_adjectives[adj_idx]
            noun = self._shuffled_nouns[noun_idx]
            
            if include_id:
                pseudonym = f"{adjective}_{noun}_{item_id}"
            else:
                pseudonym = f"{adjective}_{noun}"
        
        self._item_cache[item_id] = pseudonym
        return pseudonym
    
    def _extract_key_word(self, title: str, max_length: int = 20) -> str:
        """Extract a key word from item title for pseudonym."""
        # Remove common articles and prepositions
        stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
        
        words = title.split()
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word.lower() not in stop_words and len(clean_word) > 2:
                # Return first significant word, truncated
                return clean_word[:max_length]
        
        # Fallback: use first word
        if words:
            return ''.join(c for c in words[0] if c.isalnum())[:max_length]
        
        return "Item"
    
    def get_attacker_pseudonym(self, attacker_id: int, attacker_type: str = "User") -> str:
        """
        Generate pseudonym for an attacker agent.
        
        Args:
            attacker_id: Attacker's ID
            attacker_type: "User" or "Item"
        
        Returns:
            Pseudonym like "Attacker_Alex" or "MaliciousUser_5"
        """
        if attacker_type == "User":
            base = self.get_user_pseudonym(attacker_id)
            return f"Attacker_{base}"
        else:
            base = self.get_item_pseudonym(attacker_id)
            return f"Malicious_{base}"
    
    def get_batch_pseudonyms(
        self,
        user_ids: list,
        item_ids: list,
        user_contexts: Optional[Dict] = None,
        item_texts: Optional[list] = None,
        item_type: str = "Item"
    ) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Generate pseudonyms for a batch of users and items.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            user_contexts: Optional dict mapping user_id to context with occupation
            item_texts: Optional list mapping item_id to title
            item_type: Type of items
        
        Returns:
            Tuple of (user_pseudonyms, item_pseudonyms) dicts
        """
        user_pseudonyms = {}
        for uid in user_ids:
            occupation = None
            if user_contexts and uid in user_contexts:
                occupation = user_contexts[uid].get('role_description', {}).get('user_occupation')
            user_pseudonyms[uid] = self.get_user_pseudonym(uid, occupation=occupation)
        
        item_pseudonyms = {}
        for iid in item_ids:
            title = None
            if item_texts and iid < len(item_texts):
                title = item_texts[iid]
            item_pseudonyms[iid] = self.get_item_pseudonym(iid, item_title=title, item_type=item_type)
        
        return user_pseudonyms, item_pseudonyms
    
    def format_interaction_log(
        self,
        user_id: int,
        item_id: int,
        action: str,
        user_pseudonym: Optional[str] = None,
        item_pseudonym: Optional[str] = None
    ) -> str:
        """
        Format an interaction log entry with pseudonyms.
        
        Args:
            user_id: User ID
            item_id: Item ID
            action: Action description
            user_pseudonym: Pre-generated user pseudonym (optional)
            item_pseudonym: Pre-generated item pseudonym (optional)
        
        Returns:
            Formatted log string
        
        Example:
            >>> gen.format_interaction_log(5, 10, "selected")
            '[Harper_Technician → Stellar_Storm] selected'
        """
        if user_pseudonym is None:
            user_pseudonym = self.get_user_pseudonym(user_id)
        if item_pseudonym is None:
            item_pseudonym = self.get_item_pseudonym(item_id)
        
        return f"[{user_pseudonym} → {item_pseudonym}] {action}"


# Convenience function for quick usage
def create_pseudonym_generator(seed: Optional[int] = None) -> PseudonymGenerator:
    """Create a pseudonym generator with optional seed."""
    return PseudonymGenerator(seed=seed)
