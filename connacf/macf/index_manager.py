"""
GlobalIndexManager for MACF

Maintains global indices for MACF retrieval tools, including:
- User similarity matrix for cross-user neighbor lookup (using CF embeddings)
- Item embeddings for global candidate retrieval (using CF embeddings)
- User interaction histories for history-based retrieval

Built once at initialization from the loaded dataset.
Does NOT modify the underlying dataset.

KEY CHANGE: Uses ConnaCF's trained collaborative filtering embeddings instead of
text-based semantic embeddings. This ensures recommendations are based on learned
user-item interaction patterns, not just text similarity.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GlobalIndexManager:
    """
    Maintains global indices for MACF retrieval tools.
    
    Built once at initialization from the loaded dataset.
    Does NOT modify the underlying dataset.
    
    KEY: Uses ConnaCF's trained CF embeddings for retrieval, ensuring
    recommendations are based on collaborative filtering signals.
    
    Attributes:
        dataset: Loaded BPRDataset from RecBole
        embedding_agent: ConnaCF model with trained user_embedding and item_embedding
        user_similarity_matrix: (n_users, n_users) similarity scores between users (CF-based)
        item_embeddings: (n_items, embed_dim) CF embeddings for all items
        user_embeddings: (n_users, embed_dim) CF embeddings for all users
        user_histories: user_id -> [item_ids] mapping of interaction histories
        item_descriptions: item_id -> description mapping (can be modified for attacks)
    """
    
    def __init__(self, dataset: Any, embedding_agent: Any):
        """
        Initialize indices from dataset.
        
        Args:
            dataset: Loaded BPRDataset from RecBole
            embedding_agent: ConnaCF model with trained embeddings
                            (should have user_embedding and item_embedding nn.Embedding layers)
        """
        self.dataset = dataset
        self.embedding_agent = embedding_agent
        
        # Initialize index storage
        self.user_similarity_matrix: Optional[np.ndarray] = None  # (n_users, n_users)
        self.item_embeddings: Optional[torch.Tensor] = None       # (n_items, embed_dim) - CF embeddings
        self.user_embeddings: Optional[torch.Tensor] = None       # (n_users, embed_dim) - CF embeddings
        self.user_histories: Dict[int, List[int]] = {}            # user_id -> [item_ids]
        
        # Item descriptions (can be modified for attacks)
        self.item_descriptions: Dict[int, str] = {}               # item_id -> description
        self._original_descriptions: Dict[int, str] = {}          # Backup of original descriptions
        
        # Cache dataset properties
        self._n_users: int = 0
        self._n_items: int = 0
        self._indices_built: bool = False
        
        logger.info("GlobalIndexManager initialized (indices not yet built)")
    
    @property
    def n_users(self) -> int:
        """Number of users in the dataset."""
        return self._n_users
    
    @property
    def n_items(self) -> int:
        """Number of items in the dataset."""
        return self._n_items
    
    @property
    def indices_built(self) -> bool:
        """Whether indices have been built."""
        return self._indices_built
    
    def build_indices(self) -> None:
        """
        Build user similarity matrix and item embeddings using CF embeddings.
        
        This method:
        1. Extracts user interaction histories from the dataset
        2. Extracts CF embeddings from ConnaCF model (user_embedding, item_embedding)
        3. Computes user-user similarity based on CF embedding cosine similarity
        4. Loads item descriptions from embedding agent
        
        KEY: Uses trained CF embeddings, not text embeddings!
        
        Raises:
            RuntimeError: If dataset is not properly initialized
        """
        logger.info("Building MACF indices with CF embeddings...")
        
        # Extract dataset dimensions
        self._extract_dataset_dimensions()
        
        # Build user interaction histories
        self._build_user_histories()
        
        # Extract CF embeddings from ConnaCF model
        self._extract_cf_embeddings()
        
        # Build user similarity matrix based on CF embedding similarity
        self._build_user_similarity_matrix_cf()
        
        # Load item descriptions (for agent prompts, not retrieval)
        self._load_item_descriptions()
        
        self._indices_built = True
        logger.info(
            f"MACF indices built successfully with CF embeddings: "
            f"{self._n_users} users, {self._n_items} items, "
            f"{len(self.user_histories)} users with history, "
            f"{len(self.item_descriptions)} item descriptions"
        )
    
    def _extract_dataset_dimensions(self) -> None:
        """Extract number of users and items from dataset."""
        try:
            # RecBole datasets have num() method for field counts
            if hasattr(self.dataset, 'num'):
                self._n_users = self.dataset.num('user_id')
                self._n_items = self.dataset.num('item_id')
            elif hasattr(self.dataset, 'user_num') and hasattr(self.dataset, 'item_num'):
                self._n_users = self.dataset.user_num
                self._n_items = self.dataset.item_num
            else:
                # Fallback: try to infer from inter_feat
                if hasattr(self.dataset, 'inter_feat'):
                    user_ids = self.dataset.inter_feat['user_id'].numpy()
                    item_ids = self.dataset.inter_feat['item_id'].numpy()
                    self._n_users = int(user_ids.max()) + 1
                    self._n_items = int(item_ids.max()) + 1
                else:
                    raise RuntimeError("Cannot determine dataset dimensions")
            
            logger.info(f"Dataset dimensions: {self._n_users} users, {self._n_items} items")
            
        except Exception as e:
            logger.error(f"Error extracting dataset dimensions: {e}")
            raise RuntimeError(f"Failed to extract dataset dimensions: {e}")
    
    def _build_user_histories(self) -> None:
        """
        Build user interaction histories from dataset.
        
        Extracts user-item interactions and stores them as:
        user_histories[user_id] = [item_id_1, item_id_2, ...]
        """
        logger.info("Building user interaction histories...")
        
        self.user_histories = defaultdict(list)
        
        try:
            # RecBole datasets store interactions in inter_feat
            if hasattr(self.dataset, 'inter_feat'):
                inter_feat = self.dataset.inter_feat
                user_ids = inter_feat['user_id'].numpy()
                item_ids = inter_feat['item_id'].numpy()
                
                for user_id, item_id in zip(user_ids, item_ids):
                    # Skip padding tokens (usually 0)
                    if user_id > 0 and item_id > 0:
                        self.user_histories[int(user_id)].append(int(item_id))
            else:
                logger.warning("Dataset does not have inter_feat, user histories will be empty")
            
            # Convert defaultdict to regular dict
            self.user_histories = dict(self.user_histories)
            
            # Log statistics
            total_interactions = sum(len(h) for h in self.user_histories.values())
            avg_history_len = total_interactions / max(len(self.user_histories), 1)
            logger.info(
                f"Built histories for {len(self.user_histories)} users, "
                f"total {total_interactions} interactions, "
                f"avg {avg_history_len:.2f} items per user"
            )
            
        except Exception as e:
            logger.error(f"Error building user histories: {e}")
            self.user_histories = {}
    
    def _extract_cf_embeddings(self) -> None:
        """
        Extract collaborative filtering embeddings from ConnaCF model.
        
        Gets the trained user_embedding and item_embedding from the model.
        These embeddings capture collaborative filtering signals from training.
        """
        logger.info("Extracting CF embeddings from ConnaCF model...")
        
        try:
            # Get item embeddings from ConnaCF model
            if hasattr(self.embedding_agent, 'item_embedding'):
                self.item_embeddings = self.embedding_agent.item_embedding.weight.data.detach().cpu().clone()
                logger.info(f"Extracted item CF embeddings: shape {self.item_embeddings.shape}")
            else:
                logger.warning("No item_embedding found in model, using zeros")
                embed_dim = 64  # Default ConnaCF embedding size
                self.item_embeddings = torch.zeros(self._n_items, embed_dim)
            
            # Get user embeddings from ConnaCF model
            if hasattr(self.embedding_agent, 'user_embedding'):
                self.user_embeddings = self.embedding_agent.user_embedding.weight.data.detach().cpu().clone()
                logger.info(f"Extracted user CF embeddings: shape {self.user_embeddings.shape}")
            else:
                logger.warning("No user_embedding found in model, using zeros")
                embed_dim = self.item_embeddings.shape[1] if self.item_embeddings is not None else 64
                self.user_embeddings = torch.zeros(self._n_users, embed_dim)
            
            # Normalize embeddings for cosine similarity
            self.item_embeddings = self.item_embeddings / (
                self.item_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8
            )
            self.user_embeddings = self.user_embeddings / (
                self.user_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8
            )
            
        except Exception as e:
            logger.error(f"Error extracting CF embeddings: {e}")
            embed_dim = 64
            self.item_embeddings = torch.zeros(self._n_items, embed_dim)
            self.user_embeddings = torch.zeros(self._n_users, embed_dim)
    
    def _build_user_similarity_matrix_cf(self) -> None:
        """
        Build user-user similarity matrix based on CF embedding cosine similarity.
        
        Uses the trained user embeddings from ConnaCF for similarity computation.
        This captures collaborative filtering signals rather than just interaction overlap.
        """
        logger.info("Building user similarity matrix from CF embeddings...")
        
        if self.user_embeddings is None:
            logger.warning("No user embeddings available, falling back to Jaccard similarity")
            self._build_user_similarity_matrix_jaccard()
            return
        
        try:
            # Compute cosine similarity matrix: (n_users, n_users)
            # user_embeddings is already normalized
            self.user_similarity_matrix = torch.mm(
                self.user_embeddings, 
                self.user_embeddings.t()
            ).numpy()
            
            # Set diagonal to 1.0 (self-similarity)
            np.fill_diagonal(self.user_similarity_matrix, 1.0)
            
            logger.info(f"User similarity matrix built from CF embeddings: shape {self.user_similarity_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building CF-based similarity matrix: {e}")
            self._build_user_similarity_matrix_jaccard()
    
    def _build_user_similarity_matrix_jaccard(self) -> None:
        """
        Fallback: Build user-user similarity matrix based on interaction overlap.
        
        Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
        where A and B are the sets of items interacted with by two users.
        """
        logger.info("Building user similarity matrix (Jaccard fallback)...")
        
        # Initialize similarity matrix
        self.user_similarity_matrix = np.zeros((self._n_users, self._n_users), dtype=np.float32)
        
        # Convert histories to sets for efficient intersection/union
        user_history_sets: Dict[int, set] = {
            user_id: set(items) 
            for user_id, items in self.user_histories.items()
        }
        
        # Get list of users with non-empty histories
        users_with_history = list(user_history_sets.keys())
        n_users_with_history = len(users_with_history)
        
        if n_users_with_history == 0:
            logger.warning("No users with interaction history, similarity matrix will be zeros")
            return
        
        # Compute pairwise Jaccard similarity
        for i, user_i in enumerate(users_with_history):
            set_i = user_history_sets[user_i]
            self.user_similarity_matrix[user_i, user_i] = 1.0
            
            for j in range(i + 1, n_users_with_history):
                user_j = users_with_history[j]
                set_j = user_history_sets[user_j]
                
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                
                if union > 0:
                    similarity = intersection / union
                    self.user_similarity_matrix[user_i, user_j] = similarity
                    self.user_similarity_matrix[user_j, user_i] = similarity
        
        logger.info(f"User similarity matrix built (Jaccard): shape {self.user_similarity_matrix.shape}")
    
    def _load_item_descriptions(self) -> None:
        """
        Load item descriptions from the embedding agent.
        
        Item descriptions are stored in embedding_agent.item_text as a list
        where item_text[item_id] gives the description for that item.
        
        NOTE: These are used for agent prompts, NOT for retrieval.
        Retrieval uses CF embeddings.
        """
        logger.info("Loading item descriptions (for prompts, not retrieval)...")
        
        self.item_descriptions = {}
        self._original_descriptions = {}
        
        try:
            if self.embedding_agent is not None and hasattr(self.embedding_agent, 'item_text'):
                item_text = self.embedding_agent.item_text
                
                for item_id in range(len(item_text)):
                    desc = item_text[item_id]
                    if desc and desc != '[PAD]':
                        self.item_descriptions[item_id] = desc
                        self._original_descriptions[item_id] = desc
                
                logger.info(f"Loaded {len(self.item_descriptions)} item descriptions")
            else:
                logger.warning("No item_text available in embedding_agent")
                
        except Exception as e:
            logger.error(f"Error loading item descriptions: {e}")
    
    def get_item_description(self, item_id: int) -> str:
        """
        Get the description for an item.
        
        Args:
            item_id: The item ID
            
        Returns:
            str: Item description, or empty string if not found
        """
        return self.item_descriptions.get(item_id, "")
    
    def set_item_description(self, item_id: int, description: str) -> None:
        """
        Set the description for an item (used for attack poisoning).
        
        This modifies the description that will be used by item agents.
        The original description is preserved in _original_descriptions.
        
        Args:
            item_id: The item ID
            description: New description to set
        """
        # Preserve original if not already saved
        if item_id not in self._original_descriptions and item_id in self.item_descriptions:
            self._original_descriptions[item_id] = self.item_descriptions[item_id]
        
        self.item_descriptions[item_id] = description
        logger.debug(f"Set description for item {item_id}: {description[:50]}...")
    
    def get_original_description(self, item_id: int) -> str:
        """
        Get the original (unpoisoned) description for an item.
        
        Args:
            item_id: The item ID
            
        Returns:
            str: Original item description, or empty string if not found
        """
        return self._original_descriptions.get(item_id, self.item_descriptions.get(item_id, ""))
    
    def reset_item_descriptions(self) -> None:
        """
        Reset all item descriptions to their original values.
        
        Used to undo attack poisoning.
        """
        self.item_descriptions = self._original_descriptions.copy()
        logger.info(f"Reset {len(self.item_descriptions)} item descriptions to original values")
    
    def get_poisoned_item_ids(self) -> List[int]:
        """
        Get list of item IDs that have been modified from their original descriptions.
        
        Returns:
            List[int]: Item IDs with modified descriptions
        """
        poisoned = []
        for item_id, desc in self.item_descriptions.items():
            original = self._original_descriptions.get(item_id, "")
            if desc != original:
                poisoned.append(item_id)
        return poisoned
    
    def get_user_similarity(self, user_id: int) -> np.ndarray:
        """
        Get similarity scores for a user against all other users.
        
        Args:
            user_id: The target user ID
            
        Returns:
            np.ndarray: Array of similarity scores (n_users,)
            
        Raises:
            ValueError: If indices not built or user_id invalid
        """
        if not self._indices_built:
            raise ValueError("Indices not built. Call build_indices() first.")
        
        if user_id < 0 or user_id >= self._n_users:
            logger.warning(f"Invalid user_id {user_id}, returning zeros")
            return np.zeros(self._n_users, dtype=np.float32)
        
        return self.user_similarity_matrix[user_id].copy()
    
    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """
        Get embedding for a single item.
        
        Args:
            item_id: The item ID
            
        Returns:
            torch.Tensor: Item embedding vector (on CPU)
            
        Raises:
            ValueError: If indices not built or item_id invalid
        """
        if not self._indices_built:
            raise ValueError("Indices not built. Call build_indices() first.")
        
        if item_id < 0 or item_id >= self._n_items:
            logger.warning(f"Invalid item_id {item_id}, returning zeros")
            embed_dim = self.item_embeddings.shape[1] if self.item_embeddings is not None else 1024
            return torch.zeros(embed_dim)
        
        return self.item_embeddings[item_id].detach().cpu().clone()
    
    def get_user_history(self, user_id: int) -> List[int]:
        """
        Get interaction history for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            List[int]: List of item IDs the user has interacted with
        """
        if not self._indices_built:
            logger.warning("Indices not built, returning empty history")
            return []
        
        return self.user_histories.get(user_id, []).copy()
    
    def semantic_search_items(self, query_embedding: torch.Tensor, k: int) -> List[int]:
        """
        Find k items most similar to query embedding using CF embeddings.
        
        Uses the trained item CF embeddings for similarity computation.
        
        Args:
            query_embedding: Query embedding vector (should be CF embedding dimension)
            k: Number of items to return
            
        Returns:
            List[int]: List of k item IDs sorted by similarity (most similar first)
        """
        if not self._indices_built:
            logger.warning("Indices not built, returning empty list")
            return []
        
        if self.item_embeddings is None or self.item_embeddings.shape[0] == 0:
            logger.warning("No item embeddings available, returning empty list")
            return []
        
        if k <= 0:
            return []
        
        # Ensure query_embedding is the right shape
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Ensure both tensors are on the same device (CPU for consistency)
        query_embedding = query_embedding.detach().cpu().float()
        item_embeddings = self.item_embeddings.detach().cpu().float()
        
        # Normalize query embedding
        query_norm = query_embedding / (query_embedding.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        
        # Item embeddings are already normalized in _extract_cf_embeddings
        
        # Compute cosine similarities
        similarities = torch.mm(query_norm, item_embeddings.t()).squeeze(0)
        
        # Get top-k indices (excluding padding token at index 0)
        # Set similarity of padding token to -inf
        similarities[0] = float('-inf')
        
        # Get top-k
        k = min(k, self._n_items - 1)  # Exclude padding
        top_k_values, top_k_indices = torch.topk(similarities, k)
        
        return top_k_indices.tolist()
    
    def get_items_for_user(self, user_id: int, k: int = 15) -> List[int]:
        """
        Get k items most likely to be relevant for a user using CF embeddings.
        
        Uses the user's CF embedding to find items with high predicted affinity.
        This is the core CF-based retrieval that replaces text-based retrieval.
        
        Args:
            user_id: The user ID
            k: Number of items to return
            
        Returns:
            List[int]: List of k item IDs sorted by predicted relevance
        """
        if not self._indices_built:
            logger.warning("Indices not built, returning empty list")
            return []
        
        if self.user_embeddings is None or self.item_embeddings is None:
            logger.warning("No CF embeddings available, returning empty list")
            return []
        
        if user_id < 0 or user_id >= self._n_users:
            logger.warning(f"Invalid user_id {user_id}, returning empty list")
            return []
        
        try:
            # Get user embedding
            user_emb = self.user_embeddings[user_id].unsqueeze(0)  # (1, embed_dim)
            
            # Compute scores with all items: user_emb @ item_emb.T
            scores = torch.mm(user_emb, self.item_embeddings.t()).squeeze(0)  # (n_items,)
            
            # Exclude padding token
            scores[0] = float('-inf')
            
            # Exclude items already in user's history (we want new recommendations)
            user_history = self.user_histories.get(user_id, [])
            for item_id in user_history:
                if 0 <= item_id < len(scores):
                    scores[item_id] = float('-inf')
            
            # Get top-k
            k = min(k, self._n_items - 1 - len(user_history))
            if k <= 0:
                return []
            
            top_k_values, top_k_indices = torch.topk(scores, k)
            
            return top_k_indices.tolist()
            
        except Exception as e:
            logger.warning(f"Error getting items for user {user_id}: {e}")
            return []
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """
        Get CF embedding for a single user.
        
        Args:
            user_id: The user ID
            
        Returns:
            torch.Tensor: User CF embedding vector (on CPU)
        """
        if not self._indices_built:
            raise ValueError("Indices not built. Call build_indices() first.")
        
        if user_id < 0 or user_id >= self._n_users:
            logger.warning(f"Invalid user_id {user_id}, returning zeros")
            embed_dim = self.user_embeddings.shape[1] if self.user_embeddings is not None else 64
            return torch.zeros(embed_dim)
        
        return self.user_embeddings[user_id].detach().cpu().clone()
    
    def __repr__(self) -> str:
        """String representation of GlobalIndexManager."""
        status = "built" if self._indices_built else "not built"
        return (
            f"GlobalIndexManager("
            f"n_users={self._n_users}, "
            f"n_items={self._n_items}, "
            f"status={status}, "
            f"embedding_type=CF)"
        )
